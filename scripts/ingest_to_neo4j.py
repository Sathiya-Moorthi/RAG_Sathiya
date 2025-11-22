"""
Ingest to Neo4j Script

This script is a dedicated utility to ingest documents from the 'data/' directory
directly into the Neo4j database using LlamaIndex's KnowledgeGraphIndex.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from llama_index.core import (
    StorageContext,
    KnowledgeGraphIndex,
    Settings,
)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import get_graph_store
from core.ingestion import load_documents

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_to_neo4j(data_dir=None, pattern=None, clear_existing=False):
    """
    Ingests documents into Neo4j Knowledge Graph.
    
    Args:
        data_dir (str): Directory containing documents.
        pattern (str): File pattern to match.
        clear_existing (bool): Whether to clear the graph before ingesting.
    """
    
    # 1. Load Documents
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        
    logger.info(f"Loading documents from {data_dir}...")
    
    documents = load_documents(data_dir, pattern=pattern)
    if not documents:
        logger.error("No documents found. Exiting.")
        return

    # 2. Connect to Neo4j
    logger.info("Connecting to Neo4j...")
    graph_store = get_graph_store()
    
    if graph_store is None:
        logger.error("Failed to connect to Neo4j. Check your .env and Docker status.")
        return

    # Option to clear existing data
    if clear_existing:
        logger.warning("Clearing existing data in Neo4j...")
        try:
            # Access driver directly to run Cypher query
            with graph_store._driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("Neo4j database cleared.")
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")

    # 3. Create Knowledge Graph Index
    try:
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        
        logger.info("Ingesting data into Neo4j... (This may take time depending on data size)")
        
        # Create KG Index
        kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            max_triplets_per_chunk=2,
            storage_context=storage_context,
            include_embeddings=True,
            show_progress=True,
        )
        
        logger.info("Successfully ingested data into Neo4j.")
        
        # 4. Persist Index Metadata
        # We still need to persist the index metadata (docstore, index_store) locally
        # so we can load the index wrapper later without rebuilding everything.
        persist_dir = "./storage/kg_index"
        logger.info(f"Persisting index metadata to {persist_dir}...")
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
            
        kg_index.storage_context.persist(persist_dir=persist_dir)
        logger.info("Index metadata saved.")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Neo4j Knowledge Graph")
    parser.add_argument("--data-dir", type=str, help="Directory containing documents")
    parser.add_argument("--pattern", type=str, help="File pattern to match")
    parser.add_argument("--clear", action="store_true", help="Clear existing Neo4j data before ingestion")
    
    args = parser.parse_args()
    
    ingest_to_neo4j(
        data_dir=args.data_dir,
        pattern=args.pattern,
        clear_existing=args.clear
    )
