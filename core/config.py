import os
import logging
import sys
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore

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
Settings.llm = OpenAI(model="gpt-4.1-mini", temperature=0)
Settings.embedding = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.chunk_size = 512

def get_vector_store():
    """Set up the Vector Store (Postgres/Neon) using credentials from .env."""
    logger.info("Setting up Vector Store (Postgres/Neon)...")
    
    try:
        db_url = os.getenv("DATABASE_URL")
        
        if db_url:
            logger.info("Using DATABASE_URL from .env file")
            # Convert standard Postgres URL to SQLAlchemy format
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://")
            
            conn_str = db_url.replace("postgresql://", "postgresql+psycopg2://")
            async_conn_str = db_url.replace("postgresql://", "postgresql+asyncpg://")
            
            vector_store = PGVectorStore.from_params(
                connection_string=conn_str,
                async_connection_string=async_conn_str,
                table_name="rag_vectors",
                embed_dim=384,  # all-MiniLM-L6-v2 embedding dimension
                create_engine_kwargs={"pool_pre_ping": True},
            )
        else:
            # Use individual credentials from .env
            logger.info("Using individual POSTGRES credentials from .env file")
            postgres_host = os.getenv("POSTGRES_HOST")
            postgres_user = os.getenv("POSTGRES_USER")
            postgres_password = os.getenv("POSTGRES_PASSWORD")
            postgres_db = os.getenv("POSTGRES_DB")
            postgres_port = os.getenv("POSTGRES_PORT", "5432")
            
            if not all([postgres_host, postgres_user, postgres_password, postgres_db]):
                logger.error("Missing required PostgreSQL credentials in .env file.")
                logger.error("Required: POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB")
                return None
            
            vector_store = PGVectorStore.from_params(
                database=postgres_db,
                host=postgres_host,
                password=postgres_password,
                port=postgres_port,
                user=postgres_user,
                table_name="rag_vectors",
                embed_dim=384,  # all-MiniLM-L6-v2 embedding dimension
                create_engine_kwargs={"pool_pre_ping": True},
            )
        
        logger.info(f"Successfully connected to PostgreSQL vector store.")
        logger.info(f"Table: rag_vectors, Embedding dimension: 384 (all-MiniLM-L6-v2)")
        return vector_store
    except Exception as e:
        import traceback
        logger.error(f"Failed to connect to Postgres: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("Full error traceback:")
        logger.error(traceback.format_exc())
        logger.error("\nPlease ensure:")
        logger.error("1. PostgreSQL is running and accessible")
        logger.error("2. pgvector extension is installed: CREATE EXTENSION IF NOT EXISTS vector;")
        logger.error("3. Database credentials in .env are correct")
        logger.error("4. Network/firewall allows connection to PostgreSQL")
        return None

def get_graph_store():
    """Set up the Graph Store (Neo4j) using credentials from .env."""
    logger.info("Setting up Graph Store (Neo4j)...")
    
    try:
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            logger.error("Missing required Neo4j credentials in .env file.")
            logger.error("Required: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD")
            return None
            
        graph_store = Neo4jGraphStore(
            username=neo4j_user,
            password=neo4j_password,
            url=neo4j_uri,
            database=neo4j_database,
        )
        
        logger.info(f"Successfully connected to Neo4j graph store at {neo4j_uri}")
        return graph_store
        
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        return None

if __name__ == "__main__":
    try:
        from llama_index.core import Settings
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        print("All imports successful. Environment is set up correctly.")
    except ImportError as e:
        print(f"Import error: {e}")
