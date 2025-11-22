# 🏁 Quick Start Guide for GraphRAG Nexus

This guide will help you set up and run **GraphRAG Nexus** on your local machine.

---

## 1. Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
*   **Docker Desktop**: [Download Docker](https://www.docker.com/products/docker-desktop/) (Required for Neo4j)
*   **PostgreSQL**: [Download PostgreSQL](https://www.postgresql.org/download/)
    *   *Note*: You must enable the `pgvector` extension on your database.
*   **Git**: [Download Git](https://git-scm.com/downloads)

---

## 2. Clone the Repository

```bash
git clone https://github.com/your-username/rag-battle.git
cd rag-battle
```

---

## 3. Environment Setup

### Create Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

1.  Create a new file named `.env` in the root directory.
2.  Copy the following content into `.env` and fill in your actual credentials:

```env
# OpenAI API Key
OPENAI_API_KEY=sk-your-openai-api-key-here

# PostgreSQL Configuration (Vector Store)
POSTGRES_DB=rag_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_postgres_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Neo4j Configuration (Knowledge Graph)
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=my_neo4j_password
```

---

## 4. Database Setup

### PostgreSQL (Vector Store)

1.  Open pgAdmin or psql.
2.  Create a new database named `rag_db` (or whatever matches your `.env`).
3.  Run the following SQL command to enable `pgvector`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Neo4j (Knowledge Graph)

We use Docker Compose to run Neo4j.

1.  Start the Neo4j container:
    ```bash
    docker-compose up -d
    ```
2.  Wait a few moments for it to start. You can access the Neo4j Browser at `http://localhost:7474`.
    *   **Default Login**: `neo4j` / `neo4j` (You may need to set a new password, update `.env` accordingly).
    *   *Note*: The `docker-compose.yml` sets the password to `my_neo4j_password` by default.

---

## 5. Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application should automatically open in your default web browser at `http://localhost:8501`.

---

## 6. Using the App

1.  **Upload Documents**: Use the sidebar to upload PDF, TXT, or DOCX files (e.g., textbooks, research papers).
2.  **Ingest**: Click "Ingest File" to process the document. This creates embeddings in PostgreSQL and extracts entities for Neo4j.
3.  **Ask Questions**: Use the chat interface to ask questions related to your documents.
    *   *Example*: "What are the key findings in the provided research paper?"
    *   *Topic Check*: The system will reject queries about unrelated topics (gaming, sports, etc.).
4.  **Analyze Results**: View the side-by-side comparison of what the Vector Search found versus what the Knowledge Graph retrieved.

---

## ❓ Troubleshooting

*   **"Connection refused"**: Ensure Docker is running and PostgreSQL service is active.
*   **"ImportError"**: Double-check that you have activated your virtual environment (`venv`).
*   **"Extension 'vector' does not exist"**: Make sure you installed PostgreSQL 15+ and the `pgvector` extension is installed on your OS.

---
