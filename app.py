import streamlit as st
import os
import sys
import logging
import time
from pathlib import Path

# Add the current directory to python path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.config import get_vector_store, get_graph_store
from core.retriever import retrieve_with_fallback
from core.knowledge_graph import get_kg_index, retrieve_from_kg
from core.comparator import compare_and_synthesize
from core.ingestion import ingest_documents_to_vector_store
from llama_index.core import Settings, SimpleDirectoryReader

# Configure page settings
st.set_page_config(
    page_title="RAG vs Knowledge Graph Battle",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def load_css():
    with open("assets/style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize Resources (Cached)
@st.cache_resource
def initialize_resources():
    """Initialize connections to Vector DB and Neo4j."""
    with st.spinner("Connecting to Knowledge Bases..."):
        try:
            vector_store = get_vector_store()
            kg_index = get_kg_index()
            return vector_store, kg_index
        except Exception as e:
            st.error(f"Failed to initialize resources: {e}")
            return None, None

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    st.markdown("**Backend Status:**")
    
    vector_store, kg_index = initialize_resources()
    
    if vector_store:
        st.success("✅ Vector Store (Postgres) Connected")
    else:
        st.error("❌ Vector Store Disconnected")
        
    if kg_index:
        st.success("✅ Knowledge Graph (Neo4j) Connected")
    else:
        st.error("❌ Knowledge Graph Disconnected")
        
    st.markdown("---")
    st.markdown("### 📂 Upload Documents")
    st.info("Upload a file to add it to the Vector Knowledge Base.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx", "doc"])
    
    if uploaded_file:
        if st.button("Ingest File"):
            if not vector_store:
                st.error("❌ Vector Store not connected. Cannot ingest.")
            else:
                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    try:
                        # Save uploaded file to data directory
                        data_dir = Path("data")
                        data_dir.mkdir(exist_ok=True)
                        save_path = data_dir / uploaded_file.name
                        
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Load and Ingest
                        reader = SimpleDirectoryReader(input_files=[str(save_path)])
                        documents = reader.load_data()
                        
                        # Add metadata
                        for doc in documents:
                            doc.metadata["file_name"] = uploaded_file.name
                        
                        ingest_documents_to_vector_store(documents, vector_store)
                        st.success(f"✅ Successfully ingested {uploaded_file.name}!")
                        
                    except Exception as e:
                        st.error(f"Error during ingestion: {e}")
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This tool compares answers from a **Vector RAG** system and a **Knowledge Graph** "
        "to provide a synthesized, accurate result."
    )
    st.markdown("Created for **RAG Battle** Project.")

# Main UI
st.markdown('<h1 style="text-align: center; margin-bottom: 0.5rem;">🌌 <span class="custom-header">GraphRAG Nexus</span> 🌌</h1>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #A0AEC0; margin-bottom: 30px; font-size: 1.1rem;'>Bridging Vector Search and Knowledge Graphs for Ultimate Truth</p>", 
    unsafe_allow_html=True
)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask me anything... e.g., 'How many countries are there in the world?'"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if not vector_store or not kg_index:
            st.error("❌ Databases are not connected. Please check your configuration.")
            response_html = "❌ Databases are not connected. Please check your configuration."
            st.session_state.messages.append({"role": "assistant", "content": response_html})
        else:
            try:
                status_placeholder = st.empty()
                status_placeholder.markdown("**🛰️ Scanning Vector Database & Knowledge Graph...**")
                
                start_time = time.time()
                final_result, rag_res, kg_res, rag_src = compare_and_synthesize(prompt, vector_store, kg_index)
                end_time = time.time()
                
                status_placeholder.empty()
                
                # Create the detailed response HTML
                # Note: Indentation is stripped to prevent Markdown from interpreting it as code blocks
                response_html = f"""
<div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px;">
<div class="result-card rag-card" style="flex: 1; min-width: 300px; margin-bottom: 0;">
<div class="card-header">
<span>📚 Vector RAG</span>
<span class="source-tag">{rag_src}</span>
</div>
<div class="result-text" style="white-space: pre-wrap; font-size: 0.9rem;">{rag_res}</div>
</div>
<div class="result-card kg-card" style="flex: 1; min-width: 300px; margin-bottom: 0;">
<div class="card-header">
<span>🕸️ Knowledge Graph</span>
<span class="source-tag">Neo4j</span>
</div>
<div class="result-text" style="white-space: pre-wrap; font-size: 0.9rem;">{kg_res}</div>
</div>
</div>
<div class="result-card synth-card">
<div class="card-header">
<span>✨ Synthesized Insight</span>
<span class="source-tag">GPT-4o</span>
</div>
<div class="synth-text" style="white-space: pre-wrap;">{final_result}</div>
<div style="margin-top: 15px; font-size: 0.75rem; color: #6B7280; text-align: right; border-top: 1px solid #333; padding-top: 5px;">
⚡ Processing Time: {end_time - start_time:.2f}s
</div>
</div>
"""
                
                st.markdown(response_html, unsafe_allow_html=True)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_html})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {e}"})

# Footer
st.markdown(
    """
    <div class="footer">
        GraphRAG Nexus | Powered by LlamaIndex, Neo4j & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
