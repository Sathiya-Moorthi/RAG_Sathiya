"""
Knowledge Graph Retrieval Script

This script loads the persisted Knowledge Graph and retrieves answers.
"""

import os
import sys
import logging
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)
# from llama_index.core.graph_stores import SimpleGraphStore
try:
    from . import config as rag_config
    from .config import get_graph_store
except ImportError:
    import config as rag_config
    from config import get_graph_store

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def get_kg_index(persist_dir=None):
    """Loads the Knowledge Graph Index from storage."""
    if persist_dir is None:
        # Default to ../storage/kg_index relative to this script
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        persist_dir = os.path.join(base_dir, "storage", "kg_index")

    if not os.path.exists(persist_dir):
        logger.error(f"KG storage directory {persist_dir} does not exist. Please run build_kg.py first.")
        return None
        
    try:
        logger.info(f"Loading Knowledge Graph from {persist_dir}...")
        
        # Connect to Neo4j
        graph_store = get_graph_store()
        if graph_store is None:
            logger.error("Failed to connect to Neo4j.")
            return None
            
        # Load storage context with Neo4j graph store
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir,
            graph_store=graph_store
        )
        
        kg_index = load_index_from_storage(storage_context)
        return kg_index
    except Exception as e:
        logger.error(f"Error loading KG Index: {e}")
        return None

def retrieve_from_kg(query: str, kg_index=None):
    """
    Retrieves an answer from the Knowledge Graph.
    
    Args:
        query: The user's query.
        kg_index: The loaded KnowledgeGraphIndex (optional).
        
    Returns:
        The response string.
    """
    if kg_index is None:
        kg_index = get_kg_index()
        if kg_index is None:
            return "Knowledge Graph not available."

    try:
        logger.info(f"Querying Knowledge Graph: {query}")
        query_engine = kg_index.as_query_engine(
            include_text=True,
            response_mode="tree_summarize",
            embedding_mode="hybrid",
            similarity_top_k=5,
        )
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        logger.error(f"Error querying KG: {e}")
        return f"Error querying Knowledge Graph: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        print(f"Query: {query}")
        response = retrieve_from_kg(query)
        print(f"Result: {response}")
    else:
        print("Please provide a query as an argument.")
