import os
import logging
from typing import List
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    KnowledgeGraphIndex,
    Document
)
from llama_index.core.graph_stores import SimpleGraphStore
from rag_config import get_vector_store, logger

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

def main():
    # 1. Load Data
    documents = load_data("./data")
    if not documents:
        logger.info("No documents found. Exiting.")
        return

    # 2. Setup Storage
    vector_store = get_vector_store()
    
    if vector_store:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    else:
        logger.warning("Using in-memory vector store (not persistent across runs unless saved).")
        storage_context = StorageContext.from_defaults()
    
    # 3. Create Indices
    logger.info("Creating Vector Index...")
    # This will insert vectors into Postgres
    VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context
    )
    
    logger.info("Creating Knowledge Graph Index...")
    # Using SimpleGraphStore (in-memory) for the Graph DB part
    graph_store = SimpleGraphStore()
    storage_context.graph_store = graph_store
    
    KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=2,
        include_embeddings=True
    )

    # 4. Persist Storage Context (Essential for Graph Store and Index Metadata)
    logger.info("Persisting storage context to './storage'...")
    storage_context.persist(persist_dir="./storage")
    logger.info("Ingestion complete.")

if __name__ == "__main__":
    main()
