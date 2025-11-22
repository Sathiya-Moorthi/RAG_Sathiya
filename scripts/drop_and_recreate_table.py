"""
Drop and Recreate Table with Correct Dimensions

This script drops any existing rag_vectors tables and prepares for recreation
with the correct 384 dimensions for all-MiniLM-L6-v2 embeddings.
"""

import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def drop_tables():
    """Drop any existing rag_vectors tables."""
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        
        db_url = os.getenv("DATABASE_URL")
        
        if db_url:
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://")
            conn = psycopg2.connect(db_url)
        else:
            conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                database=os.getenv("POSTGRES_DB")
            )
        
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Find all rag_vectors tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE '%rag_vectors%'
        """)
        tables = [row[0] for row in cur.fetchall()]
        
        if tables:
            logger.info(f"Found {len(tables)} table(s): {tables}")
            for table in tables:
                logger.info(f"Dropping table: {table}")
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
            logger.info("All tables dropped successfully.")
        else:
            logger.info("No existing tables found.")
        
        cur.close()
        conn.close()
        logger.info("Table cleanup complete. Tables will be recreated with correct dimensions on next ingestion.")
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Dropping existing rag_vectors tables")
    logger.info("="*60)
    success = drop_tables()
    sys.exit(0 if success else 1)

