"""
Check Neo4j Data Script

This script connects to Neo4j using the configuration from rag_config.py
and runs a simple Cypher query to count nodes and show a sample relationship.
"""

import logging
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import get_graph_store

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def check_neo4j_data():
    """Connects to Neo4j and prints statistics about the data."""
    
    logger.info("Connecting to Neo4j...")
    graph_store = get_graph_store()
    
    if not graph_store:
        logger.error("Could not connect to Neo4j. Check your .env and Docker container.")
        return

    try:
        # Access the underlying Neo4j driver from the graph_store
        driver = graph_store._driver
        
        with driver.session() as session:
            # 1. Count total nodes
            result = session.run("MATCH (n) RETURN count(n) AS total_nodes")
            total_nodes = result.single()["total_nodes"]
            print(f"\nTotal Nodes in Graph: {total_nodes}")
            
            # 2. Count total relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS total_relationships")
            total_rels = result.single()["total_relationships"]
            print(f"Total Relationships in Graph: {total_rels}")
            
            if total_nodes > 0:
                # 3. Show sample nodes (Limit 5)
                print("\nSample Nodes (First 5):")
                result = session.run("MATCH (n) RETURN labels(n) AS labels, n.id AS id LIMIT 5")
                for record in result:
                    print(f" - Labels: {record['labels']}, ID: {record.get('id', 'N/A')}")
                    
                # 4. Show sample relationships (Limit 5)
                print("\nSample Relationships (First 5):")
                result = session.run("MATCH (a)-[r]->(b) RETURN labels(a) AS start_labels, type(r) AS rel_type, labels(b) AS end_labels LIMIT 5")
                for record in result:
                    print(f" - {record['start_labels']} --[{record['rel_type']}]--> {record['end_labels']}")
            else:
                print("\nThe graph is empty. Run 'python build_kg.py' to populate it.")

    except Exception as e:
        logger.error(f"Error querying Neo4j: {e}")

if __name__ == "__main__":
    check_neo4j_data()
