import logging
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings
)
from llama_index.core.retrievers import BaseRetriever
# Import rag_config to ensure Settings.embedding is configured correctly
import rag_config
from rag_config import get_vector_store, logger

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
        logger.warning("Srored Vetors does not have this details !!!")
        # In a full CRAG, we would trigger a web search here.
        return "Fallback method: The retrieved information was not relevant to your query"

    # 4. Generate
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
    print("--- Starting RAG Retrieval ---")
    
    # 0. Verify embedding configuration is set
    import rag_config  # Ensure Settings.embedding is configured
    if Settings.embedding is None:
        logger.error("Embedding model is not configured. Please check rag_config.py")
        print("Error: Embedding model not configured. Exiting.")
        return
    
    logger.info(f"Embedding model configured: {type(Settings.embedding).__name__}")
    logger.info(f"Expected dimension: 384 (all-MiniLM-L6-v2)")
    
    # 1. Setup Storage
    vector_store = get_vector_store()
    
    if vector_store is None:
        logger.error("Could not initialize Vector Store. Check your database connection and .env file.")
        print("Error: Database connection failed. Exiting.")
        return
    
    logger.info(f"Vector store table: {vector_store.table_name}")
    logger.info(f"Vector store embed_dim: {vector_store.embed_dim}")
    
    # 2. Load Index
    logger.info("Loading index from storage...")
    try:
        if not os.path.exists("./storage"):
            logger.error("Storage directory './storage' not found.")
            print("Error: Storage not found. Please run 'python ingest.py' first to load data.")
            return

        # We load the storage context from disk (for Graph Store and docstore)
        # But we inject our Postgres vector store so it knows where to look for vectors
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage",
            vector_store=vector_store
        )
        
        # Verify embedding configuration
        if Settings.embedding is None:
            logger.error("Embedding model is not configured. Please check rag_config.py")
            print("Error: Embedding model not configured. Exiting.")
            return
        
        logger.info(f"Using embedding model: {type(Settings.embedding).__name__}")
        logger.info(f"Embedding dimension: 384 (all-MiniLM-L6-v2)")
        
        # Load the VectorStoreIndex with explicit embedding model
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=Settings.embedding  # Explicitly set embedding model
        )
        logger.info("Index loaded successfully.")
        
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        print(f"Error loading index: {e}")
        return

    # 3. Querying
    # Ensure retriever uses the correct embedding model
    vector_retriever = vector_index.as_retriever(
        similarity_top_k=3
    )
    # The retriever will use Settings.embedding automatically, but we verify it's set
    logger.info(f"Retriever configured with embedding: {type(Settings.embedding).__name__}")
    
    queries = [
        # '''Which of the following statements is ALWAYS TRUE when parallel lines are  
        #     cut by a transversal
        #     (i) corresponding angles are supplementary. 
        #     (ii) alternate interior angles are supplementary.
        #     (iii) alternate exterior angles are supplementary.
        #     (iv) interior angles on the same side of the transversal are supplementary.'''
        "how many countries are there in the world?"
    ]
    
    print(f"\nProcessing {len(queries)} queries...")
    
    for query in queries:
        print(f"\n--------------------------------------------------")
        print(f"Query: {query}")
        result = corrective_rag_logic(query, vector_retriever)
        print(f"Result: {result}")
        print(f"--------------------------------------------------\n")

if __name__ == "__main__":
    import os
    main()
