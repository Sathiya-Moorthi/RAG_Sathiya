"""
Database Connection Test Script

This script tests the PostgreSQL connection and checks if pgvector extension is installed.
Run this to diagnose connection issues before running ingestion scripts.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Setup logging with UTF-8 encoding for Windows
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_basic_connection():
    """Test basic PostgreSQL connection using psycopg2."""
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        
        db_url = os.getenv("DATABASE_URL")
        
        if db_url:
            logger.info("Testing connection using DATABASE_URL...")
            # Convert postgres:// to postgresql:// if needed
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://")
            conn = psycopg2.connect(db_url)
        else:
            logger.info("Testing connection using individual credentials...")
            conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                database=os.getenv("POSTGRES_DB")
            )
        
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Check PostgreSQL version
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        logger.info(f"✓ Connected to PostgreSQL: {version.split(',')[0]}")
        
        # Check if pgvector extension exists
        cur.execute("""
            SELECT EXISTS(
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            );
        """)
        extension_exists = cur.fetchone()[0]
        
        if extension_exists:
            logger.info("✓ pgvector extension is installed")
        else:
            logger.warning("✗ pgvector extension is NOT installed")
            logger.info("Installing pgvector extension...")
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("✓ pgvector extension installed successfully")
            except Exception as e:
                logger.error(f"✗ Failed to install pgvector extension: {e}")
                logger.error("Please install manually: CREATE EXTENSION IF NOT EXISTS vector;")
                return False
        
        # Check if table exists
        cur.execute("""
            SELECT EXISTS(
                SELECT FROM information_schema.tables 
                WHERE table_name = 'rag_vectors'
            );
        """)
        table_exists = cur.fetchone()[0]
        
        if table_exists:
            logger.info("✓ Table 'rag_vectors' exists")
            # Check table structure
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'rag_vectors'
            """)
            columns = cur.fetchall()
            logger.info(f"  Columns: {', '.join([f'{col[0]} ({col[1]})' for col in columns])}")
        else:
            logger.info("ℹ Table 'rag_vectors' does not exist yet (will be created on first use)")
        
        cur.close()
        conn.close()
        
        logger.info("\n✓ All connection tests passed!")
        return True
        
    except ImportError:
        logger.error("✗ psycopg2 not installed. Installing...")
        logger.info("Run: pip install psycopg2-binary")
        return False
    except Exception as e:
        logger.error(f"✗ Connection test failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        return False


def test_pgvector_connection():
    """Test PGVectorStore connection using llama_index."""
    try:
        logger.info("\n" + "="*60)
        logger.info("Testing PGVectorStore connection...")
        logger.info("="*60)
        
        sys.path.append(str(Path(__file__).parent.parent))
        from core.config import get_vector_store
        
        vector_store = get_vector_store()
        
        if vector_store:
            logger.info("✓ PGVectorStore connection successful")
            logger.info(f"  Table: {vector_store.table_name}")
            logger.info(f"  Embedding dimension: {vector_store.embed_dim}")
            return True
        else:
            logger.error("✗ PGVectorStore connection failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ PGVectorStore test failed: {e}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all connection tests."""
    logger.info("="*60)
    logger.info("PostgreSQL Connection Diagnostic Tool")
    logger.info("="*60)
    logger.info("\nChecking environment variables...")
    
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        logger.info("✓ DATABASE_URL is set")
        # Mask password in log
        masked_url = db_url.split("@")[0].split("//")[0] + "//***:***@" + "@".join(db_url.split("@")[1:])
        logger.info(f"  URL: {masked_url}")
    else:
        logger.info("ℹ DATABASE_URL not set, checking individual credentials...")
        credentials = {
            "POSTGRES_HOST": os.getenv("POSTGRES_HOST"),
            "POSTGRES_PORT": os.getenv("POSTGRES_PORT", "5432"),
            "POSTGRES_USER": os.getenv("POSTGRES_USER"),
            "POSTGRES_PASSWORD": "***" if os.getenv("POSTGRES_PASSWORD") else None,
            "POSTGRES_DB": os.getenv("POSTGRES_DB")
        }
        
        missing = [k for k, v in credentials.items() if k != "POSTGRES_PASSWORD" and not v]
        if missing:
            logger.error(f"✗ Missing required credentials: {', '.join(missing)}")
            logger.error("Please set these in your .env file")
            return False
        else:
            logger.info("✓ All PostgreSQL credentials are set")
            for k, v in credentials.items():
                logger.info(f"  {k}: {v}")
    
    logger.info("\n" + "="*60)
    logger.info("Step 1: Testing basic PostgreSQL connection...")
    logger.info("="*60)
    basic_test = test_basic_connection()
    
    if not basic_test:
        logger.error("\n✗ Basic connection test failed. Please fix the issues above.")
        return False
    
    logger.info("\n" + "="*60)
    logger.info("Step 2: Testing PGVectorStore connection...")
    logger.info("="*60)
    pgvector_test = test_pgvector_connection()
    
    if not pgvector_test:
        logger.error("\n✗ PGVectorStore test failed. Please check the configuration.")
        return False
    
    logger.info("\n" + "="*60)
    logger.info("✓ All tests passed! You can now run ingestion scripts.")
    logger.info("="*60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

