import os
import logging
import sys
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in .env")
    sys.exit(1)

# Set up LlamaIndex Settings
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embedding = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 512

def get_vector_store():
    """Set up the Vector Store (Postgres/Neon)."""
    logger.info("Setting up Vector Store (Postgres)...")
    
    try:
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            # Convert standard Postgres URL to SQLAlchemy format
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://")
            
            conn_str = db_url.replace("postgresql://", "postgresql+psycopg2://")
            async_conn_str = db_url.replace("postgresql://", "postgresql+asyncpg://")
            
            vector_store = PGVectorStore.from_params(
                connection_string=conn_str,
                async_connection_string=async_conn_str,
                table_name="rag_vectors",
                embed_dim=1536,
                create_engine_kwargs={"pool_pre_ping": True},
            )
        else:
            vector_store = PGVectorStore.from_params(
                database=os.getenv("POSTGRES_DB", "postgres"),
                host=os.getenv("POSTGRES_HOST", "localhost"),
                password=os.getenv("POSTGRES_PASSWORD", "password"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                table_name="rag_vectors",
                embed_dim=1536,
                create_engine_kwargs={"pool_pre_ping": True},
            )
        return vector_store
    except Exception as e:
        logger.error(f"Failed to connect to Postgres: {e}")
        return None
