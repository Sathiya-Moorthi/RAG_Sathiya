"""
PDF Document Ingestion Script

This script loads PDF documents from a directory, generates embeddings using
Hugging Face all-MiniLM-L6-v2 model, and stores them in PostgreSQL pgvector.

Usage:
    python ingest_pdf.py [--data-dir ./data] [--pattern *.pdf]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Document,
    Settings
)

# Import configuration (this sets up embedding model and Settings)
from . import config as rag_config
from .config import get_vector_store, logger


def setup_logging(verbose: bool = False):
    """Configure logging level."""
    # Always show INFO level and above for this script
    level = logging.INFO if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
        force=True  # Override any existing configuration
    )
    # Also set rag_config logger to INFO level to see connection messages
    rag_config.logger.setLevel(logging.INFO)


def load_documents(
    data_dir: str,
    pattern: str = None,
    recursive: bool = True
) -> List[Document]:
    """
    Load PDF, TXT, DOCX, DOC documents from the specified directory.
    
    Args:
        data_dir: Directory path containing PDF, TXT, DOCX, DOC files
        pattern: File pattern to match (optional, loads all supported types if None)
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        List of Document objects containing PDF, TXT, DOCX, DOC files
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"ERROR: Directory '{data_dir}' does not exist!")
        logger.error(f"Please create the directory and add PDF, TXT, DOCX, or DOC files to it.")
        logger.error(f"Or specify a different directory with --data-dir flag.")
        return []
    
    if not data_path.is_dir():
        logger.error(f"{data_dir} is not a directory.")
        return []
    
    logger.info(f"Loading documents from: {data_dir}")
    logger.info(f"Supported formats: PDF, TXT, DOCX, DOC")
    logger.info(f"Recursive: {recursive}")
    
    # Use SimpleDirectoryReader which supports multiple file types
    reader = SimpleDirectoryReader(
        input_dir=str(data_path),
        input_files=None,
        exclude_hidden=True,
        filename_as_id=True,
        recursive=recursive,
        file_metadata=lambda filename: {
            "file_path": str(filename),
            "file_name": os.path.basename(filename),
            "file_type": os.path.splitext(filename)[1].lower().lstrip('.') or "unknown"
        }
    )
    
    # Load documents
    try:
        documents = reader.load_data()
        
        # Filter supported file types (PDF, TXT, DOCX, DOC)
        supported_extensions = {'.pdf', '.txt', '.docx', '.doc'}
        filtered_documents = []
        for doc in documents:
            file_name = doc.metadata.get("file_name", "").lower()
            file_path = doc.metadata.get("file_path", "").lower()
            # Check if file extension matches supported types
            if any(file_name.endswith(ext) or file_path.endswith(ext) for ext in supported_extensions):
                filtered_documents.append(doc)
        
        logger.info(f"Loaded {len(filtered_documents)} document(s) (PDF, TXT, DOCX, DOC).")
        
        if len(filtered_documents) == 0:
            logger.warning(f"No supported files (PDF, TXT, DOCX, DOC) found in {data_dir}")
        else:
            # Log file types found
            file_types = {}
            for doc in filtered_documents:
                file_type = doc.metadata.get("file_type", "unknown")
                file_types[file_type] = file_types.get(file_type, 0) + 1
            logger.info("File types loaded: " + ", ".join([f"{ext.upper()}: {count}" for ext, count in file_types.items()]))
        
        return filtered_documents
        
    except Exception as e:
        logger.error(f"Error loading PDF, TXT, DOCX, DOC documents: {e}")
        return []


def ingest_documents_to_vector_store(
    documents: List[Document],
    vector_store: Optional = None
) -> Optional[VectorStoreIndex]:
    """
    Generate embeddings and store them in PostgreSQL pgvector for PDF, TXT, DOCX, DOC documents.
    
    Args:
        documents: List of Document objects to ingest containing PDF, TXT, DOCX, DOC files
        vector_store: PGVectorStore instance (if None, will get from config)
        
    Returns:
        VectorStoreIndex instance or None if failed containing PDF, TXT, DOCX, DOC documents
    """
    if not documents:
        logger.error("No documents provided for ingestion.")
        return None
    
    # Verify embedding configuration
    if Settings.embedding is None:
        logger.error("Embedding model is not configured. Please check core/config.py")
        return None
    
    logger.info(f"Using embedding model: {type(Settings.embedding).__name__}")
    logger.info(f"Embedding dimension: 384 (all-MiniLM-L6-v2)")
    
    # Get vector store if not provided
    if vector_store is None:
        logger.info("Connecting to PostgreSQL vector store...")
        vector_store = get_vector_store()
        
        if vector_store is None:
            logger.error("Failed to connect to PostgreSQL vector store.")
            return None
    
    # Create storage context with vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create Vector Index - this generates embeddings and stores them in pgvector
    logger.info(f"Generating embeddings for {len(documents)} document(s)...")
    logger.info("Storing embeddings in PostgreSQL pgvector table 'rag_vectors'...")
    logger.info(f"Using embedding model: {type(Settings.embedding).__name__}")
    logger.info(f"Expected embedding dimension: 384 (all-MiniLM-L6-v2)")
    
    # Clear any existing embeddings from documents to ensure fresh generation
    for doc in documents:
        if hasattr(doc, 'embedding'):
            doc.embedding = None
    
    try:
        vector_index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            show_progress=True,
            embed_model=Settings.embedding  # Explicitly set embedding model
        )
        
        logger.info("Successfully generated embeddings and stored in pgvector.")
        logger.info(f"Total documents processed: {len(documents)}")
        
        return vector_index
        
    except Exception as e:
        import traceback
        logger.error(f"Error during embedding generation and storage: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("Full error traceback:")
        logger.error(traceback.format_exc())
        logger.error("\nPlease ensure:")
        logger.error("1. PostgreSQL is running and accessible")
        logger.error("2. pgvector extension is installed: CREATE EXTENSION IF NOT EXISTS vector;")
        logger.error("3. Database credentials in .env are correct")
        logger.error("4. You have CREATE TABLE permissions on the database")
        logger.error("5. Network/firewall allows connection to PostgreSQL")
        return None


def main():
    """Main function to ingest PDF, TXT, DOCX, DOC documents."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    parser = argparse.ArgumentParser(
        description="Ingest PDF, TXT, DOCX, DOC documents into PostgreSQL pgvector with Hugging Face embeddings"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(script_dir / "data"),
        help=f"Directory containing PDF files (default: {script_dir / 'data'})"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="File pattern to match (optional, loads all supported types if not specified)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search recursively in subdirectories (default: True)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    logger.info("=" * 60)
    logger.info("PDF, TXT, DOCX, DOC Document Ingestion Script")
    logger.info("Using Hugging Face all-MiniLM-L6-v2 embeddings")
    logger.info("=" * 60)
    
    # Step 1: Load all supported documents (PDF, TXT, DOCX, DOC)
    logger.info("\n[Step 1] Loading documents (PDF, TXT, DOCX, DOC)...")
    documents = load_documents(
        data_dir=args.data_dir,
        pattern=args.pattern,
        recursive=args.recursive
    )
    
    if not documents:
        logger.error("No supported documents (PDF, TXT, DOCX, DOC) found. Exiting.")
        sys.exit(1)
    
    logger.info(f"[OK] Loaded {len(documents)} document(s)")
    for i, doc in enumerate(documents, 1):
        file_name = doc.metadata.get("file_name", "Unknown")
        file_type = doc.metadata.get("file_type", "unknown").upper()
        logger.info(f"  {i}. {file_name} ({file_type})")
    
    # Step 2: Connect to PostgreSQL and get vector store
    logger.info("\n[Step 2] Connecting to PostgreSQL vector store...")
    vector_store = get_vector_store()
    
    if vector_store is None:
        logger.error("Failed to connect to PostgreSQL. Exiting.")
        sys.exit(1)
    
    logger.info("[OK] Connected to PostgreSQL vector store")
    logger.info("  Table: rag_vectors")
    logger.info("  Embedding dimension: 384")
    
    # Step 3: Generate embeddings and store in pgvector
    logger.info("\n[Step 3] Generating embeddings and storing in pgvector...")
    vector_index = ingest_documents_to_vector_store(documents, vector_store)
    
    if vector_index is None:
        logger.error("=" * 60)
        logger.error("FAILED TO INGEST DOCUMENTS")
        logger.error("=" * 60)
        logger.error("The ingestion function returned None. This usually means:")
        logger.error("1. An error occurred during embedding generation (check logs above)")
        logger.error("2. Vector store connection failed (check Step 2 logs)")
        logger.error("3. Embedding model is not configured (check rag_config.py)")
        logger.error("")
        logger.error("Please review the error messages above for details.")
        logger.error("Run with --verbose flag for more detailed logging.")
        logger.error("=" * 60)
        sys.exit(1)
    
    # Step 4: Summary
    logger.info("\n" + "=" * 60)
    logger.info("Ingestion Complete!")
    logger.info("=" * 60)
    logger.info(f"[OK] Processed: {len(documents)} document(s)")
    logger.info("[OK] Embeddings stored in PostgreSQL pgvector")
    logger.info("[OK] Table: rag_vectors")
    logger.info("[OK] Embedding model: all-MiniLM-L6-v2 (384 dimensions)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

