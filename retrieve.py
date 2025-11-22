import logging
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings
)
from llama_index.core.retrievers import BaseRetriever
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
        return "Fallback method Result: The retrieved information was not relevant to your query"

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
    
    # 1. Setup Storage
    vector_store = get_vector_store()
    
    if vector_store is None:
        logger.error("Could not initialize Vector Store. Check your database connection and .env file.")
        print("Error: Database connection failed. Exiting.")
        return
    
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
        
        # Load the VectorStoreIndex
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )
        logger.info("Index loaded successfully.")
        
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        print(f"Error loading index: {e}")
        return

    # 3. Querying
    vector_retriever = vector_index.as_retriever(similarity_top_k=3)
    
    queries = [
        "Who is CEO of Social Eagle?"
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
