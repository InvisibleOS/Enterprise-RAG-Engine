# Enterprise RAG Engine

A production-ready **Retrieval-Augmented Generation (RAG)** system built to eliminate AI hallucinations by grounding answers in internal data.

[**Live Demo Here**](https://enterprise-rag-engine.streamlit.app/)

Unlike standard tutorial implementations that rely on heavy frameworks, this project utilizes a **Pure Python** architecture with raw SDKs for maximum control, lower latency, and zero dependency bloat.

## Key Features

* **Zero-Framework Architecture:** Built without LangChain or LlamaIndex to demonstrate first-principles understanding of Vector Search and LLM orchestration.
* **Manual Ingestion Pipeline:** Implements a custom sliding-window chunking algorithm (1000 chars / 200 overlap) to preserve semantic context.
* **Serverless Vector Search:** Uses **Pinecone Serverless** for scalable, low-latency embedding retrieval.
* **Dual-Model Architecture:**
    * **Embeddings:** Google `text-embedding-004` (768 Dimensions) for semantic search.
    * **Generation:** Google `Gemini 1.5 Flash` for high-speed, cost-effective answer synthesis.
* **Source Citations:** Every answer includes a "View Sources" dropdown, allowing users to audit the AI's reasoning against the original PDF text.

## Tech Stack

| Component | Technology | Reasoning |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Rapid prototyping, clean state management. |
| **LLM** | Gemini 1.5 Flash | High throughput, large context window, free tier availability. |
| **Vector DB** | Pinecone | Managed serverless infrastructure (no Docker required). |
| **Orchestration** | **Pure Python** | Avoided LangChain/LlamaIndex to prevent dependency bloat and "black box" abstraction leakage. |
| **Parsing** | pypdf | Robust, lightweight PDF text extraction. |

## Architecture

The system follows a standard RAG ETL (Extract, Transform, Load) pipeline:

1.  **Ingest:** PDF documents are parsed into raw text.
2.  **Chunk:** Text is sliced into overlapping windows to ensure boundary context isn't lost.
3.  **Embed:** Chunks are sent to Google's Embedding API to generate 768-dimensional vectors.
4.  **Upsert:** Vectors + Metadata (original text) are batched (100 at a time) and pushed to Pinecone.
5.  **Retrieve:** User queries are embedded and compared against the database using **Cosine Similarity**.
6.  **Synthesize:** Top 3 matches are injected into a strict system prompt for the LLM to generate the final answer.

## How to Run Locally

*Note: You can skip installation and test the live app directly via the link above.*

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/enterprise-rag-engine.git](https://github.com/yourusername/enterprise-rag-engine.git)
cd enterprise-rag-engine
````

### 2\. Set Up Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it (Windows)
.\.venv\Scripts\activate

# Activate it (Mac/Linux)
source .venv/bin/activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Configure Secrets

Create a `.streamlit/secrets.toml` file (or input keys directly in the UI sidebar):

```toml
GOOGLE_API_KEY = "your_google_key"
PINECONE_API_KEY = "your_pinecone_key"
```

### 5\. Launch Application

```bash
python -m streamlit run app.py
```

## 🧠 Why "Hard Mode" (Pure Python)?

Most RAG tutorials use LangChain. While powerful, I found that framework abstractions often obscure the underlying logic and introduce "dependency hell."

By building this manually, I gained direct control over:

  * **Chunking Strategy:** Writing the sliding window logic ensured I understood exactly how data was being segmented.
  * **API Latency:** Direct SDK calls allowed for easier debugging and optimization of network requests.

  * **Error Handling:** Custom try/except blocks provided clearer visibility into failure states (e.g., Pinecone connection timeouts) than generic framework errors.

