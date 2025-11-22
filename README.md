# 🌌 GraphRAG Nexus: RAG vs Knowledge Graph Battle

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)](https://streamlit.io/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10.0-purple.svg)](https://www.llamaindex.ai/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-018bff.svg)](https://neo4j.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791.svg)](https://www.postgresql.org/)

**GraphRAG Nexus** is an advanced Retrieval-Augmented Generation (RAG) system designed to compare and synthesize answers from two powerful knowledge sources:
1.  **Vector RAG**: Embeddings stored in PostgreSQL (pgvector) for semantic search.
2.  **Knowledge Graph**: Structured relationships stored in Neo4j for context-aware retrieval.

This project demonstrates the strengths of combining statistical vector search with structured knowledge graphs to provide highly accurate, hallucination-resistant answers for educational and research purposes.

---

## ✨ Features

*   **⚔️ Dual Retrieval Engine**: Simultaneously queries a Vector Store and a Knowledge Graph.
*   **🧠 Smart Synthesis**: Uses GPT-4o to analyze both answers, identify discrepancies, and generate a final, authoritative response.
*   **🌐 Web Search Fallback**: Automatically searches the web (via DuckDuckGo) if local knowledge is insufficient.
*   **🛡️ Topic Guardrails**: Restricts queries to educational and research topics, filtering out irrelevant content (gaming, entertainment, etc.).
*   **📂 Interactive Ingestion**: Upload PDF, TXT, or DOCX files directly via the UI to expand the knowledge base.
*   **💬 Continuous Chat**: Conversational interface with history tracking.
*   **📊 Visual Comparison**: Side-by-side view of Vector RAG vs. Knowledge Graph results.

---

## 🚀 Architecture

1.  **Frontend**: Built with **Streamlit** for a responsive, chat-based UI.
2.  **Orchestration**: **LlamaIndex** manages document indexing, retrieval, and query engines.
3.  **Vector Store**: **PostgreSQL** with `pgvector` extension stores document embeddings (Hugging Face `all-MiniLM-L6-v2`).
4.  **Graph Store**: **Neo4j** stores entities and relationships extracted from documents.
5.  **LLM**: **OpenAI GPT-4o** handles reasoning, synthesis, and topic classification.

---

## 🛠️ Prerequisites

*   **Python 3.10+**
*   **Docker & Docker Compose** (for Neo4j)
*   **PostgreSQL** (Local or Cloud) with `pgvector` extension enabled.
*   **OpenAI API Key**

---

## 🏁 Quick Start

See [QUICKSTART.md](QUICKSTART.md) for a detailed, step-by-step guide to setting up and running the project.

### Brief Overview

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/rag-battle.git
    cd rag-battle
    ```

2.  **Set up Environment**:
    Create a `.env` file:
    ```env
    OPENAI_API_KEY=sk-...
    POSTGRES_URL=postgresql://user:pass@localhost:5432/dbname
    NEO4J_URL=bolt://localhost:7687
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=password
    ```

3.  **Start Neo4j**:
    ```bash
    docker-compose up -d
    ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

---

## 📁 Project Structure

```
RAG-Battle/
├── app.py                 # Main Streamlit application
├── core/                  # Core logic modules
│   ├── config.py          # Configuration & Database connections
│   ├── ingestion.py       # Document processing & Vector indexing
│   ├── knowledge_graph.py # Neo4j Graph retrieval logic
│   ├── retriever.py       # Vector retrieval & Web fallback logic
│   └── comparator.py      # RAG vs Graph comparison & synthesis
├── assets/                # CSS and static assets
├── data/                  # Directory for uploaded documents
├── docker-compose.yml     # Neo4j Docker configuration
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
