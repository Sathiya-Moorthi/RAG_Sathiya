"""
Compare RAG vs Knowledge Graph Script

This script retrieves answers from both the Vector Store (RAG) and the Knowledge Graph,
compares them using an LLM, and provides a synthesized accurate result.
"""

import os
import sys
import logging
import argparse
from llama_index.core import Settings

# Import our modules
from . import config as rag_config
from .retriever import retrieve_with_fallback, get_vector_store
from .knowledge_graph import retrieve_from_kg, get_kg_index

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_and_synthesize(query: str, vector_store, kg_index):
    """
    Queries both RAG and KG, then compares and synthesizes the answer.
    """
    
    # 1. Get Vector RAG Result
    logger.info("--- Getting Vector RAG Result ---")
    rag_response, rag_source = retrieve_with_fallback(query, vector_store=vector_store)
    logger.info(f"Vector RAG Source: {rag_source}")
    
    # 2. Get Knowledge Graph Result
    logger.info("--- Getting Knowledge Graph Result ---")
    kg_response = retrieve_from_kg(query, kg_index=kg_index)
    
    # 3. Compare and Synthesize using LLM
    logger.info("--- Comparing and Synthesizing ---")
    
    llm = Settings.llm
    
    prompt = (
        f"You are an expert assistant tasked with providing the most accurate answer to a user's query.\n"
        f"You have two sources of information:\n\n"
        f"SOURCE 1 (Vector RAG): {rag_response}\n\n"
        f"SOURCE 2 (Knowledge Graph): {kg_response}\n\n"
        f"User Query: {query}\n\n"
        f"Task:\n"
        f"1. Analyze both answers.\n"
        f"2. Identify any discrepancies or conflicts.\n"
        f"3. Synthesize a final, comprehensive, and accurate answer based on both sources.\n"
        f"4. If one source is clearly hallucinating or irrelevant, prioritize the other.\n"
        f"5. Provide the Final Answer followed by a brief Explanation of your synthesis logic.\n\n"
        f"Format:\n"
        f"Final Answer: [Your detailed answer here]\n\n"
        f"Explanation: [Why you chose this answer]"
    )
    
    comparison_result = llm.complete(prompt).text
    return comparison_result, rag_response, kg_response, rag_source

def main():
    parser = argparse.ArgumentParser(description="Compare RAG and Knowledge Graph results")
    parser.add_argument("--query", type=str, help="The query to ask")
    args = parser.parse_args()
    
    if not args.query:
        # Interactive mode
        print("Entering interactive mode...")
        
        # Load resources once
        logger.info("Loading resources...")
        vector_store = get_vector_store()
        kg_index = get_kg_index()
        
        while True:
            try:
                query = input("\nEnter Query (or 'q' to quit): ").strip()
                if query.lower() == 'q':
                    break
                if not query:
                    continue
                    
                final_result, rag_res, kg_res, rag_src = compare_and_synthesize(query, vector_store, kg_index)
                
                print("\n" + "="*80)
                print(f"VECTOR RAG ANSWER [Source: {rag_src}]:\n{rag_res}")
                print("-" * 40)
                print(f"KNOWLEDGE GRAPH ANSWER [Source: Neo4j Graph]:\n{kg_res}")
                print("="*80)
                print(f"SYNTHESIZED RESULT:\n{final_result}")
                print("="*80)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"An error occurred: {e}")
    else:
        # Single query mode
        vector_store = get_vector_store()
        kg_index = get_kg_index()
        final_result, rag_res, kg_res, rag_src = compare_and_synthesize(args.query, vector_store, kg_index)
        
        print("\n" + "="*80)
        print(f"VECTOR RAG ANSWER [Source: {rag_src}]:\n{rag_res}")
        print("-" * 40)
        print(f"KNOWLEDGE GRAPH ANSWER [Source: Neo4j Graph]:\n{kg_res}")
        print("="*80)
        print(f"SYNTHESIZED RESULT:\n{final_result}")
        print("="*80)

if __name__ == "__main__":
    main()
