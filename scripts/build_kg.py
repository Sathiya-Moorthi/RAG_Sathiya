"""
Build Knowledge Graph Script

This script loads documents and builds a Knowledge Graph using LlamaIndex.
It persists the graph to disk for later retrieval.

Usage:
    python build_kg.py
"""

import os
import sys
import logging
from pathlib import Path
from llama_index.core import (
    StorageContext,
    KnowledgeGraphIndex,
    Settings,
    load_index_from_storage
)
# from llama_index.core.graph_stores import SimpleGraphStore
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.ingestion import load_documents
from core.config import get_graph_store

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def build_knowledge_graph(data_dir=None, pattern=None):
    """Builds and persists the Knowledge Graph."""
    
    # 1. Load Documents
    if data_dir is None:
        data_dir = os.path.join(Path(__file__).parent.parent, "data")
        
    logger.info(f"Loading documents from {data_dir}...")
    
    documents = load_documents(data_dir, pattern=pattern)
    if not documents:
        logger.error("No documents found. Exiting.")
        return

    # 2. Setup Storage for KG
    # We will use Neo4jGraphStore
    try:
        graph_store = get_graph_store()
        if graph_store is None:
            logger.error("Failed to initialize Neo4j Graph Store. Exiting.")
            return
            
        storage_context = StorageContext.from_defaults(graph_store=graph_store)

        # 3. Create Knowledge Graph Index
        logger.info("Building Knowledge Graph Index... (This may take a while)")
        
        # We explicitly set the embedding model from Settings (configured in rag_config)
        kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            max_triplets_per_chunk=2,
            storage_context=storage_context,
            include_embeddings=True,
            show_progress=True,
        )
        
        logger.info("Knowledge Graph built and stored in Neo4j successfully.")
        
        # 4. Persist the Index Metadata (docstore, index_store) to disk
        # The graph data itself is in Neo4j, but we need the index struct/metadata
        persist_dir = "./storage/kg_index"
        logger.info(f"Persisting Index Metadata to {persist_dir}...")
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
            
        kg_index.storage_context.persist(persist_dir=persist_dir)
        logger.info("Index metadata persisted successfully.")
        
    except Exception as e:
        logger.error(f"Failed to build Knowledge Graph: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Knowledge Graph from documents")
    parser.add_argument("--data-dir", type=str, help="Directory containing documents")
    parser.add_argument("--pattern", type=str, help="File pattern to match (e.g., *.txt)")
    
    args = parser.parse_args()
    
    build_knowledge_graph(data_dir=args.data_dir, pattern=args.pattern)
