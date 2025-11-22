"""
Fix Database Table Dimensions

This script checks and optionally recreates the rag_vectors table with correct dimensions
for the all-MiniLM-L6-v2 embedding model (384 dimensions).
"""

import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_and_fix_table():
    """Check table dimensions and recreate if needed."""
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
        
        # Check for any rag_vectors tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE '%rag_vectors%'
        """)
        tables = [row[0] for row in cur.fetchall()]
        logger.info(f"Found tables with 'rag_vectors': {tables}")
        
        # Check both possible table names
        table_name = 'rag_vectors'
        alt_table_name = 'data_rag_vectors'
        
        # Check if rag_vectors exists
        cur.execute("""
            SELECT EXISTS(
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
        """, (table_name,))
        table_exists = cur.fetchone()[0]
        
        # Check if data_rag_vectors exists
        cur.execute("""
            SELECT EXISTS(
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
        """, (alt_table_name,))
        alt_table_exists = cur.fetchone()[0]
        
        if table_exists:
            logger.info("Table 'rag_vectors' exists. Checking dimensions...")
            
            # Get the embedding column dimension
            cur.execute("""
                SELECT typmod 
                FROM pg_attribute 
                WHERE attrelid = 'rag_vectors'::regclass 
                AND attname = 'embedding'
            """)
            result = cur.fetchone()
            
            if result and result[0]:
                # Extract dimension from typmod (pgvector stores dimension as part of type)
                typmod = result[0]
                # For pgvector, dimension is stored in typmod
                # We can check by querying the actual column definition
                cur.execute("""
                    SELECT column_name, data_type, 
                           pg_catalog.format_type(atttypid, atttypmod) as full_type
                    FROM information_schema.columns 
                    JOIN pg_attribute ON attrelid = (SELECT oid FROM pg_class WHERE relname = 'rag_vectors')
                    WHERE table_name = 'rag_vectors' 
                    AND column_name = 'embedding'
                """)
                col_info = cur.fetchone()
                
                if col_info:
                    full_type = col_info[2]
                    logger.info(f"Current embedding column type: {full_type}")
                    
                    if '384' not in full_type or 'vector(1536)' in full_type.lower():
                        logger.warning("Table has wrong dimensions. Need to recreate with 384 dimensions.")
                        logger.info("Dropping existing table...")
                        cur.execute("DROP TABLE IF EXISTS rag_vectors CASCADE;")
                        logger.info("Table dropped. It will be recreated with correct dimensions on next ingestion.")
                    else:
                        logger.info("Table has correct dimensions (384).")
            else:
                logger.info("Could not determine dimensions. Table will be recreated if needed.")
        else:
            logger.info("Table 'rag_vectors' does not exist. It will be created with 384 dimensions on first use.")
        
        cur.close()
        conn.close()
        logger.info("Database check complete.")
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Checking and fixing table dimensions")
    logger.info("="*60)
    success = check_and_fix_table()
    sys.exit(0 if success else 1)

