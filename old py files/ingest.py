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
# Import rag_config to ensure Settings.embedding is configured
import rag_config
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
    # 0. Verify embedding configuration
    from llama_index.core import Settings
    if Settings.embedding is None:
        logger.error("Embedding model is not configured. Please check rag_config.py")
        return
    logger.info(f"Using embedding model: {type(Settings.embedding).__name__}")
    
    # 1. Load Data
    documents = load_data("./data")
    if not documents:
        logger.info("No documents found. Exiting.")
        return

    # 2. Setup Storage with PGVector
    logger.info("Connecting to PostgreSQL vector store...")
    vector_store = get_vector_store()
    
    if vector_store:
        logger.info("Successfully connected to PostgreSQL vector store.")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    else:
        logger.warning("Failed to connect to PostgreSQL. Using in-memory vector store (not persistent).")
        storage_context = StorageContext.from_defaults()
    
    # 3. Create Vector Index (this will generate embeddings and store them in pgvector)
    logger.info("Creating Vector Index and generating embeddings...")
    logger.info("This will store embeddings in PostgreSQL pgvector table 'rag_vectors'...")
    vector_index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True
    )
    logger.info(f"Vector index created. Documents embedded and stored in pgvector.")
    
    # 4. Create Knowledge Graph Index
    logger.info("Creating Knowledge Graph Index...")
    # Using SimpleGraphStore (in-memory) for the Graph DB part
    graph_store = SimpleGraphStore()
    storage_context.graph_store = graph_store
    
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=2,
        include_embeddings=True,
        show_progress=True
    )
    logger.info("Knowledge Graph Index created.")

    # 5. Persist Storage Context (Essential for Graph Store and Index Metadata)
    logger.info("Persisting storage context to './storage'...")
    storage_context.persist(persist_dir="./storage")
    logger.info("Ingestion complete. Embeddings are now stored in PostgreSQL pgvector.")

if __name__ == "__main__":
    main()
