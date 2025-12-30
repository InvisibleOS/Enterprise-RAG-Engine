# Enterprise RAG Engine

### High-Precision Document Intelligence & Analysis Platform

[Live Demonstration](https://enterprise-rag-engine.streamlit.app/)

Enterprise RAG Engine is a production-grade **Retrieval-Augmented Generation** system designed for high-stakes corporate environments where accuracy is non-negotiable. The platform ingests internal documentation—such as technical manuals, legal contracts, and compliance policies—and enables users to query this knowledge base with verifiable accuracy.

Unlike rapid-prototyping implementations that rely on third-party abstraction layers like the LangChain API, this platform is architected using a **Pure Python** approach. It relies purely on the raw **Google Gemini API** and SDKs to ensure deterministic behavior, minimize latency, and eliminate the security and maintenance risks associated with dependency bloat.

## Core Capabilities

- **Deterministic Ingestion Pipeline:** Utilizes a sliding-window segmentation algorithm (1000 characters with 200-character overlap) to strictly preserve semantic context across clause boundaries, preventing information fragmentation.

- **Serverless Vector Infrastructure:** Integrated with Pinecone for scalable, milliseconds-latency embedding retrieval without the operational overhead of container management.

- **Hallucination Mitigation:** Enforces a strict evidence-based generation model. The reasoning engine must cite specific paragraphs from the source text for every claim generated, providing an audit trail for compliance verification.

- **Dual-Model Architecture:**

   - **Semantic Search:** Google text-embedding-004 (768 Dimensions) for high-fidelity vector representation.

   - **Synthesis:** Google Gemini 1.5 Flash for high-throughput, cost-efficient answer generation.

## Strategic Engineering Decisions

A core differentiator of this platform is the deliberate exclusion of orchestration frameworks such as LangChain or LlamaIndex. While such frameworks offer rapid prototyping speed, they introduce significant technical debt in production environments.

This platform utilizes Raw Google GenAI and Pinecone Clients to achieve specific engineering outcomes:

**1. Elimination of Abstraction Leaks**

Frameworks often obscure underlying API interactions, making it difficult to debug specific failure modes (e.g., determining if a timeout occurred at the embedding layer or the vector database layer). By using raw SDKs, the system maintains total visibility into the retrieval process, allowing for precise tuning of top_k parameters based on actual data performance.

**2. Dependency Stability & Security**

The AI ecosystem evolves rapidly. Frameworks frequently introduce breaking changes or dependency conflicts. A Pure Python approach ensures the application relies only on stable, official vendor SDKs, significantly reducing the risk of upstream breakage and reducing the container image size by removing unused transitive dependencies.

**3. Granular Error Handling**

Direct integration enables the implementation of custom resilience patterns. The system features a bespoke exponential backoff mechanism to handle API rate limits (HTTP 429) and service interruptions gracefully. This logic is often difficult to configure or observe when wrapped in high-level chain abstractions.

**4. Latency Optimization**

By removing the overhead of chain serialization, callback handlers, and intermediate parsing steps common in frameworks, the application achieves lower latency in the Request-to-Response loop, a critical metric for real-time user interaction.

## System Architecture

The application follows a specialized ETL (Extract, Transform, Load) pipeline optimized for document processing:

1. **Ingestion:** PDF documents are parsed into raw text strings.

2. **Segmentation:** Text is processed via the sliding-window algorithm to maintain semantic integrity.

3. **Vectorization:** Segments are passed to the Embedding API to generate dense vectors.

4. **Indexing:** Vectors and Metadata (Source Text) are batched and upserted to the vector database.

5. **Retrieval:** User queries are embedded and compared using Cosine Similarity to retrieve the most relevant context windows.

6. **Synthesis:** Retrieved contexts are injected into a strict system prompt for the Large Language Model to generate the final answer.

## Technical Stack

| Component | Technology | Role                                |
|-----------|------------|-------------------------------------|
| Frontend  | Streamlit  | User interface and state management.|
| Reasoning Engine | Gemini 1.5 Flash | Context synthesis and answer generation. |
| Memory | Pinecone | Managed vector storage and similarity search. |
| Orchestration | Pure Python | Request handling, error logic, and API integration. |
| Parsing | pypdf | Document text extraction. |

## Local Deployment

*Note: The application is deployed and accessible via the provided demonstration link.*

**1. Repository Setup**

```git
git clone https://github.com/InvisibleOS/Enterprise-RAG-Engine.git
cd Enterprise-RAG-Engine
```


**2. Environment Configuration**

```py
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate
```


**3. Dependency Installation**

```py
pip install -r requirements.txt
```


**4. Security Configuration**

Create a `.streamlit/secrets.toml` file to manage API keys securely:

```
GOOGLE_API_KEY = "your_google_key"
PINECONE_API_KEY = "your_pinecone_key"
```


**5. Application Launch**

```py
python -m streamlit run app.py
```
