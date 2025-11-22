# RAG Application with Corrective RAG and Graph/Vector DB

This application implements a RAG system using LlamaIndex, Postgres (Neon) for Vector Storage, and a simple Knowledge Graph.

## Prerequisites

1.  **Python 3.10+**
2.  **Postgres Database (Neon)** with `pgvector` extension enabled.
    *   Run `CREATE EXTENSION vector;` in your database.
3.  **OpenAI API Key**

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment**:
    *   Edit `.env` and fill in your OpenAI API Key and Postgres credentials.
    *   Ensure `POSTGRES_HOST`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` are set correctly.

3.  **Add Data**:
    *   Place your text book files (txt, pdf, etc.) in the `data/` directory.
    *   A sample `textbook.txt` is provided.

## Running the Application

### 1. Ingest Data (Load & Index)
This script loads data from `data/`, creates embeddings in Postgres, and builds the Knowledge Graph.
```bash
python ingest.py
```

### 2. Retrieve Data (Query)
This script loads the indices and runs the Corrective RAG queries.
```bash
python retrieve.py
```

## Features

*   **Vector Store**: Uses Postgres (via `pgvector`) to store embeddings.
*   **Knowledge Graph**: Builds a simple Knowledge Graph from the documents.
*   **Corrective RAG**: Implements a "Retrieve-Grade-Generate" loop.
    *   Retrieves documents.
    *   Uses LLM to grade relevance.
    *   Filters irrelevant documents.
    *   Generates answer.
