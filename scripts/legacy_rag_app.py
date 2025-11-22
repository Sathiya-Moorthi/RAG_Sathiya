import os
import logging
import sys
from typing import List

from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    KnowledgeGraphIndex,
    Settings,
    Document
)
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in .env")
    sys.exit(1)

# Set up LlamaIndex Settings
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embedding = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 512

def load_data(directory_path: str) -> List[Document]:
    """Load data from the specified directory."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.warning(f"Directory {directory_path} did not exist. Created it.")
        return []
    
    reader = SimpleDirectoryReader(directory_path)
    documents = reader.load_data()
    logger.info(f"Loaded {len(documents)} documents.")
    return documents

def setup_vector_db(documents: List[Document]):
    """Set up the Vector Store (Postgres/Neon)."""
    logger.info("Setting up Vector Store (Postgres)...")
    
    # Create PGVectorStore
    # Note: You need to enable the vector extension in your Postgres DB: CREATE EXTENSION vector;
    
    try:
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            # Convert standard Postgres URL to SQLAlchemy format
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://")
            
            conn_str = db_url.replace("postgresql://", "postgresql+psycopg2://")
            async_conn_str = db_url.replace("postgresql://", "postgresql+asyncpg://")
            
            vector_store = PGVectorStore.from_params(
                connection_string=conn_str,
                async_connection_string=async_conn_str,
                table_name="rag_vectors",
                embed_dim=1536,
                create_engine_kwargs={"pool_pre_ping": True},
            )
        else:
            vector_store = PGVectorStore.from_params(
                database=os.getenv("POSTGRES_DB", "postgres"),
                host=os.getenv("POSTGRES_HOST", "localhost"),
                password=os.getenv("POSTGRES_PASSWORD", "password"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                table_name="rag_vectors",
                embed_dim=1536,  # text-embedding-3-small dimension
                create_engine_kwargs={"pool_pre_ping": True},
            )
        return vector_store
    except Exception as e:
        logger.error(f"Failed to connect to Postgres: {e}")
        logger.info("Falling back to simple in-memory vector store for demonstration.")
        return None

def corrective_rag_logic(query: str, retriever: BaseRetriever):
    """
    Implements a simple Corrective RAG flow:
    1. Retrieve nodes.
    2. Grade nodes for relevance.
    3. Filter irrelevant nodes.
    4. Generate response.
    """
    logger.info(f"Processing query: {query}")
    
    # 1. Retrieve
    nodes = retriever.retrieve(query)
    logger.info(f"Retrieved {len(nodes)} nodes.")
    
    if not nodes:
        return "No relevant information found in the knowledge base."

    # 2. Grade & Filter (Corrective Step)
    relevant_nodes = []
    llm = Settings.llm
    
    for node in nodes:
        # Simple grading prompt
        prompt = (
            f"Context: {node.text}\n\n"
            f"Query: {query}\n\n"
            "Is the context relevant to the query? Answer 'YES' or 'NO'."
        )
        response = llm.complete(prompt).text.strip().upper()
        
        if "YES" in response:
            relevant_nodes.append(node)
        else:
            logger.info(f"Node filtered out as irrelevant: {node.node_id}")

    # 3. Fallback / Search (Simplified)
    if not relevant_nodes:
        logger.warning("All retrieved nodes were judged irrelevant.")
        # In a full CRAG, we would trigger a web search here.
        return "The retrieved information was not relevant to your query. (Fallback to web search would happen here)"

    # 4. Generate
    # Create a temp index or just use the LLM to synthesize from relevant nodes
    context_str = "\n\n".join([n.text for n in relevant_nodes])
    final_prompt = (
        f"Context information is below.\n"
        f"---------------------\n"
        f"{context_str}\n"
        f"---------------------\n"
        f"Given the context information and not prior knowledge, answer the query.\n"
        f"Query: {query}\n"
        f"Answer: "
    )
    
    response = llm.complete(final_prompt)
    return response.text

def main():
    # 1. Load Data
    documents = load_data("./data")
    if not documents:
        logger.info("No documents found. Exiting.")
        return

    # 2. Setup Storage
    vector_store = setup_vector_db(documents)
    
    if vector_store:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    else:
        storage_context = StorageContext.from_defaults()
    
    # 3. Create Indices
    logger.info("Creating Vector Index...")
    vector_index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context
    )
    
    logger.info("Creating Knowledge Graph Index...")
    # Using SimpleGraphStore (in-memory) for the Graph DB part
    graph_store = SimpleGraphStore()
    storage_context.graph_store = graph_store
    
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=2,
        include_embeddings=True
    )

    # 4. Querying
    # We can combine retrievers or just use the vector retriever for the CRAG example
    vector_retriever = vector_index.as_retriever(similarity_top_k=3)
    
    query = "Collect all the event happened on 1950 in India ?"
    result = corrective_rag_logic(query, vector_retriever)
    print(f"\nQuery: {query}\nResult: {result}\n")
    
    query = "Tell me more about Independt movement in india"
    result = corrective_rag_logic(query, vector_retriever)
    print(f"\nQuery: {query}\nResult: {result}\n")

if __name__ == "__main__":
    main()
