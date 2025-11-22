"""
Retrieve with Web Fallback Script

This script retrieves data from PGVectorStore. If no relevant data is found,
it falls back to web search to get answers using RAG (Retrieval-Augmented Generation).

Usage:
    python retrieve_with_web_fallback.py [--query "your question"] [--threshold 0.7]
"""

import os
import sys
import argparse
import logging
from typing import List, Optional, Tuple

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    Settings,
    Document
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RetrieverQueryEngine

# Import configuration (this sets up embedding model and Settings)
from . import config as rag_config
from .config import get_vector_store, logger

# Try to import web search capabilities
try:
    from duckduckgo_search import DDGS
    WEB_SEARCH_AVAILABLE = True
    WEB_SEARCH_TYPE = "ddgs"
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    WEB_SEARCH_TYPE = None

if not WEB_SEARCH_AVAILABLE:
    logger.warning("Web search tools not available. Install with: pip install duckduckgo-search")


def setup_logging(verbose: bool = False):
    """Configure logging level."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
        force=True
    )
    rag_config.logger.setLevel(logging.INFO)


def check_relevance(node: NodeWithScore, query: str, threshold: float = 0.7) -> bool:
    """
    Check if a retrieved node is relevant to the query.
    
    Args:
        node: Retrieved node with score
        query: User query
        threshold: Relevance threshold (default: 0.7)
        
    Returns:
        True if relevant, False otherwise
    """
    # For vector similarity, lower distance means higher relevance
    # We'll consider it relevant if score is above threshold
    # Note: Similarity scores vary by vector store, adjust threshold accordingly
    
    # Simple relevance check using LLM
    llm = Settings.llm
    prompt = (
        f"Context: {node.text[:500]}\n\n"
        f"Query: {query}\n\n"
        "Is this context relevant to answering the query? Answer 'YES' or 'NO' only."
    )
    
    try:
        response = llm.complete(prompt).text.strip().upper()
        is_relevant = "YES" in response
        
        # Also check similarity score if available
        if hasattr(node, 'score') and node.score is not None:
            # For cosine similarity, higher is better; for distance, lower is better
            # Assuming similarity score (higher = more relevant)
            score_relevant = node.score >= threshold if node.score <= 1.0 else True
            return is_relevant and score_relevant
        
        return is_relevant
    except Exception as e:
        logger.warning(f"Error checking relevance: {e}. Assuming relevant.")
        return True


def web_search_query(query: str, num_results: int = 3) -> List[Document]:
    """
    Perform web search and return results as documents.
    
    Args:
        query: Search query
        num_results: Number of search results to return
        
    Returns:
        List of Document objects from web search
    """
    if not WEB_SEARCH_AVAILABLE:
        logger.error("Web search is not available. Please install required packages.")
        return []
    
    try:
        logger.info(f"Performing web search for: {query}")
        documents = []
        
        if WEB_SEARCH_TYPE == "ddgs" and WEB_SEARCH_AVAILABLE:
            # Use duckduckgo_search library directly
            from duckduckgo_search import DDGS
            
            try:
                results = DDGS().text(query, max_results=num_results)
                
                if results:
                    for result in results:
                        title = result.get('title', '')
                        snippet = result.get('body', '')
                        url = result.get('href', '')
                        
                        text = f"{title}\n\n{snippet}"
                        doc = Document(
                            text=text,
                            metadata={
                                "source": "web_search",
                                "url": url,
                                "title": title,
                                "query": query
                            }
                        )
                        documents.append(doc)
                        logger.info(f"Found web result: {title}")
                else:
                    logger.warning("DDGS returned no results.")
            except Exception as e:
                logger.error(f"DDGS search failed: {e}")
                return []
        else:
            logger.error("Web search is not available. Please install duckduckgo-search.")
            return []
        
        logger.info(f"Retrieved {len(documents)} web search results")
        return documents
        
    except Exception as e:
        import traceback
        logger.error(f"Error during web search: {e}")
        logger.error(traceback.format_exc())
        logger.error("Make sure duckduckgo-search is installed: pip install duckduckgo-search")
        return []


def check_query_topic(query: str) -> bool:
    """
    Check if the query is related to education, research, or academic topics.
    
    Args:
        query: User query
        
    Returns:
        True if relevant, False otherwise
    """
    llm = Settings.llm
    prompt = (
        f"You are a query classifier for an educational and research assistant.\n"
        f"Your task is to determine if the following query is related to:\n"
        f"1. Education (school, college, university topics)\n"
        f"2. Academic Research\n"
        f"3. General knowledge suitable for students\n"
        f"4. Technical or scientific concepts\n\n"
        f"If the query is about entertainment, gaming, sports, gossip, news, or irrelevant casual chat, answer 'NO'.\n"
        f"Otherwise, answer 'YES'.\n\n"
        f"Query: {query}\n"
        f"Answer (YES/NO):"
    )
    
    try:
        response = llm.complete(prompt).text.strip().upper()
        return "YES" in response
    except Exception as e:
        logger.warning(f"Error checking query topic: {e}. Defaulting to allow.")
        return True


def retrieve_with_fallback(
    query: str,
    vector_store=None,
    relevance_threshold: float = 0.7,
    min_results: int = 1
) -> Tuple[str, str]:
    """
    Retrieve answer with fallback to web search.
    
    Args:
        query: User query
        vector_store: PGVectorStore instance
        relevance_threshold: Minimum relevance score (0-1)
        min_results: Minimum number of relevant results needed
        
    Returns:
        Tuple of (answer, source) where source is "vector_store" or "web_search"
    """
    logger.info(f"Processing query: {query}")
    
    # Step 1: Try to retrieve from vector store
    if vector_store is None:
        logger.info("Connecting to PostgreSQL vector store...")
        vector_store = get_vector_store()
        
        if vector_store is None:
            logger.warning("Failed to connect to vector store. Falling back to web search.")
            
            # Check topic before web search
            if not check_query_topic(query):
                logger.info("Query rejected: Not educational/research related.")
                return (
                    "I am an AI assistant developed for educational and research purposes. "
                    "Your query appears to be outside this scope (e.g., gaming, entertainment, sports). "
                    "Please ask a question related to education, research, or academic topics.",
                    "topic_restriction"
                )
                
            source = "web_search"
            documents = web_search_query(query)
            
            if not documents:
                return "I couldn't retrieve information from the knowledge base or the web. Please check your connection.", source
            
            # Generate answer from web results
            context = "\n\n".join([doc.text for doc in documents])
            llm = Settings.llm
            prompt = (
                f"Context information from web search:\n"
                f"---------------------\n"
                f"{context}\n"
                f"---------------------\n"
                f"Given the context information above, answer the query.\n"
                f"Query: {query}\n"
                f"Answer: "
            )
            answer = llm.complete(prompt).text
            return answer.strip(), source
    
    try:
        # Create storage context and index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Verify embedding configuration
        if Settings.embedding is None:
            logger.error("Embedding model not configured. Falling back to web search.")
            
            # Check topic before web search
            if not check_query_topic(query):
                return (
                    "I am an AI assistant developed for educational and research purposes. "
                    "Your query appears to be outside this scope. "
                    "Please ask a question related to education, research, or academic topics.",
                    "topic_restriction"
                )
                
            documents = web_search_query(query)
            if not documents:
                return "Configuration error: Embedding model not set up.", "error"
            context = "\n\n".join([doc.text for doc in documents])
            llm = Settings.llm
            prompt = (
                f"Context: {context}\n\n"
                f"Query: {query}\n\n"
                f"Answer: "
            )
            return llm.complete(prompt).text.strip(), "web_search"
        
        logger.info(f"Using embedding model: {type(Settings.embedding).__name__}")
        
        # Load index from vector store
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=Settings.embedding
        )
        
        # Create retriever
        retriever = vector_index.as_retriever(
            similarity_top_k=5,
            embed_model=Settings.embedding
        )
        
        # Retrieve nodes
        logger.info("Retrieving from vector store...")
        nodes = retriever.retrieve(query)
        logger.info(f"Retrieved {len(nodes)} nodes from vector store")
        
        # Check relevance of retrieved nodes
        relevant_nodes = []
        for node in nodes:
            if check_relevance(node, query, relevance_threshold):
                relevant_nodes.append(node)
                logger.info(f"Found relevant node: {node.node_id[:50]}...")
        
        # Step 2: Determine if we have enough relevant results
        if len(relevant_nodes) >= min_results:
            logger.info(f"Found {len(relevant_nodes)} relevant results in vector store. Using local data.")
            
            # Generate answer from vector store results
            context_str = "\n\n".join([n.text for n in relevant_nodes])
            llm = Settings.llm
            prompt = (
                f"Context information from knowledge base:\n"
                f"---------------------\n"
                f"{context_str}\n"
                f"---------------------\n"
                f"Given the context information above and not prior knowledge, answer the query.\n"
                f"Query: {query}\n"
                f"Answer: "
            )
            
            answer = llm.complete(prompt).text
            return answer.strip(), "vector_store"
        
        else:
            # Step 3: Fallback to web search
            logger.warning(f"Only found {len(relevant_nodes)} relevant results (minimum: {min_results})")
            logger.info("Falling back to web search...")
            
            # Check topic before web search
            if not check_query_topic(query):
                logger.info("Query rejected: Not educational/research related.")
                return (
                    "I am an AI assistant developed for educational and research purposes. "
                    "Your query appears to be outside this scope (e.g., gaming, entertainment, sports). "
                    "Please ask a question related to education, research, or academic topics.",
                    "topic_restriction"
                )
            
            # Get web search results
            web_documents = web_search_query(query, num_results=5)
            
            if not web_documents:
                logger.warning("Web search returned no results. Using available vector store results.")
                if relevant_nodes:
                    context_str = "\n\n".join([n.text for n in relevant_nodes])
                    llm = Settings.llm
                    prompt = (
                        f"Context: {context_str}\n\n"
                        f"Query: {query}\n\n"
                        f"Note: Limited information available. Answer based on what is provided.\n"
                        f"Answer: "
                    )
                    return llm.complete(prompt).text.strip(), "vector_store_limited"
                else:
                    return "I couldn't find relevant information in the knowledge base or through web search. Please rephrase your query or provide more context.", "no_results"
            
            # Combine vector store and web results
            combined_context = []
            if relevant_nodes:
                combined_context.append("From knowledge base:")
                combined_context.append("\n".join([n.text for n in relevant_nodes]))
            
            combined_context.append("\nFrom web search:")
            combined_context.append("\n\n".join([doc.text for doc in web_documents]))
            
            context_str = "\n\n".join(combined_context)
            
            # Generate answer from combined sources
            llm = Settings.llm
            prompt = (
                f"Context information:\n"
                f"---------------------\n"
                f"{context_str}\n"
                f"---------------------\n"
                f"Given the context information above, answer the query.\n"
                f"Query: {query}\n"
                f"Answer: "
            )
            
            answer = llm.complete(prompt).text
            return answer.strip(), "web_search"
            
    except Exception as e:
        import traceback
        logger.error(f"Error during retrieval: {e}")
        logger.error(traceback.format_exc())
        
        # Final fallback to web search
        logger.info("Attempting web search as final fallback...")
        documents = web_search_query(query)
        
        if documents:
            context = "\n\n".join([doc.text for doc in documents])
            llm = Settings.llm
            prompt = (
                f"Context from web search:\n{context}\n\n"
                f"Query: {query}\n\n"
                f"Answer: "
            )
            return llm.complete(prompt).text.strip(), "web_search_fallback"
        else:
            return f"An error occurred: {str(e)}. Could not retrieve information.", "error"


def main():
    """Main function for interactive query processing."""
    parser = argparse.ArgumentParser(
        description="Retrieve data from PGVectorStore with web search fallback"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query to process (if not provided, will run interactively)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Relevance threshold for vector store results (default: 0.7)"
    )
    parser.add_argument(
        "--min-results",
        type=int,
        default=1,
        help="Minimum number of relevant results needed before using vector store (default: 1)"
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
    logger.info("Retrieve with Web Fallback RAG System")
    logger.info("=" * 60)
    
    # Verify embedding configuration
    if Settings.embedding is None:
        logger.error("Embedding model is not configured. Please check core/config.py")
        sys.exit(1)
    
    logger.info(f"Embedding model: {type(Settings.embedding).__name__}")
    logger.info(f"Embedding dimension: 384 (all-MiniLM-L6-v2)")
    logger.info(f"Relevance threshold: {args.threshold}")
    logger.info(f"Minimum results: {args.min_results}")
    
    if not WEB_SEARCH_AVAILABLE:
        logger.warning("Web search fallback is not available.")
        logger.info("Install with: pip install duckduckgo-search")
    
    # Get vector store
    logger.info("\nConnecting to PostgreSQL vector store...")
    vector_store = get_vector_store()
    
    if vector_store:
        logger.info(f"[OK] Connected to vector store")
        logger.info(f"Table: {vector_store.table_name}")
        logger.info(f"Embedding dimension: {vector_store.embed_dim}")
    else:
        logger.warning("Warning: Could not connect to vector store. Only web search will be available.")
    
    # Process queries
    if args.query:
        # Single query mode
        logger.info(f"\nProcessing query: {args.query}")
        answer, source = retrieve_with_fallback(
            args.query,
            vector_store,
            args.threshold,
            args.min_results
        )
        
        print("\n" + "=" * 60)
        print(f"Query: {args.query}")
        print(f"Source: {source}")
        print("=" * 60)
        print(f"Answer: {answer}")
        print("=" * 60)
    else:
        # Interactive mode
        print("\nInteractive mode. Enter queries (type 'exit' or 'quit' to stop):")
        print("=" * 60)
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                answer, source = retrieve_with_fallback(
                    query,
                    vector_store,
                    args.threshold,
                    args.min_results
                )
                
                print("\n" + "=" * 60)
                print(f"Source: {source}")
                print("=" * 60)
                print(f"Answer: {answer}")
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"Error: {e}")


if __name__ == "__main__":
    main()

