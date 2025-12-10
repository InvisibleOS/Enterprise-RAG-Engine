import os
import time
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

class EnterpriseBrain:
    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        self.client = genai.Client(api_key=self.google_api_key)
        self.model_name = "gemini-flash-latest"
        self.index_name = "rag-resume-project"
        self.pc = None
        self.index = None

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def _extract_embedding(self, response) -> list:
        if not response:
            raise RuntimeError("Embedding API returned no response")
        emb_list = getattr(response, "embeddings", None)
        if not emb_list:
            # some clients return data or candidates; try common alternatives
            emb_list = getattr(response, "data", None) or getattr(response, "vectors", None)
        if not emb_list:
            raise RuntimeError("Embedding API returned no embeddings")

        first = emb_list[0]
        # possible shapes: object with .values, dict with 'embedding', or plain list
        if hasattr(first, "values"):
            return list(first.values)
        if isinstance(first, dict) and ("embedding" in first or "vector" in first):
            val = first.get("embedding") or first.get("vector")
            if val is None:
                raise RuntimeError("Embedding entry missing data")
            return list(val)
        if isinstance(first, (list, tuple)):
            return list(first)
        # fallback: try to coerce
        try:
            return list(first)
        except Exception:
            raise RuntimeError("Unable to parse embedding from response")

    def _extract_text_from_ai_response(self, resp) -> str:
        if not resp:
            return ""
        if hasattr(resp, "text") and resp.text:
            return resp.text
        if hasattr(resp, "message") and resp.message:
            return str(resp.message)
        candidates = getattr(resp, "candidates", None) or getattr(resp, "outputs", None)
        if candidates and len(candidates) > 0:
            first = candidates[0]
            if hasattr(first, "content"):
                return first.content
            if isinstance(first, dict) and "content" in first:
                return first["content"]
        return str(resp)

    def ingest_pdf(self, pdf_path: str, pinecone_api_key: str) -> str:
        try:
            self.pc = Pinecone(api_key=pinecone_api_key)
            existing_indexes = [i.name for i in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768, 
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
            self.index = self.pc.Index(self.index_name)

            reader = PdfReader(pdf_path)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text: full_text += text + "\n"

            text_chunks = self._chunk_text(full_text)
            vectors_to_upsert = []
            
            for i, chunk_text in enumerate(text_chunks):
                response = self.client.models.embed_content(
                    model="text-embedding-004",
                    contents=chunk_text,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                embedding = self._extract_embedding(response)
                metadata = {"text": chunk_text, "id": str(i)}
                vectors_to_upsert.append((f"vec_{i}", embedding, metadata))

            batch_size = 100
            if not self.index:
                raise RuntimeError("Vector index not initialized")
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i : i + batch_size]
                # guard: ensure vectors are in the expected shape (id, vector, metadata)
                safe_batch = []
                for vid, vec, meta in batch:
                    if not isinstance(vec, (list, tuple)):
                        raise RuntimeError(f"Invalid vector for {vid}")
                    safe_batch.append((vid, list(vec), meta))
                self.index.upsert(vectors=safe_batch)
            return f"Success: Manually processed {len(text_chunks)} chunks."
        except Exception as e:
            return f"Error: {str(e)}"

    def ask(self, query: str) -> dict:
        if not self.index:
            return {"result": "Please upload a document first."}
        try:
            response = self.client.models.embed_content(
                model="text-embedding-004",
                contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            query_vector = self._extract_embedding(response)
            search_results = self.index.query(vector=query_vector, top_k=3, include_metadata=True)
            
            context_text = ""
            source_docs = []
            for match in search_results.get('matches', []):
                meta = match.get('metadata', {}) or {}
                text = meta.get('text', '')
                score = match.get('score', None)
                context_text += f"---\n{text}\n"
                class Document:
                    def __init__(self, page_content, metadata):
                        self.page_content = page_content
                        self.metadata = metadata
                doc = Document(page_content=text, metadata={'score': score})
                source_docs.append(doc)

            prompt = f"Answer the user's question using ONLY the context below.\n\nContext:\n{context_text}\n\nQuestion: {query}"
            ai_response = self.client.models.generate_content(model=self.model_name, contents=prompt)
            text_result = self._extract_text_from_ai_response(ai_response)
            return {"result": text_result, "source_documents": source_docs}
        except Exception as e:
            return {"result": f"Error: {str(e)}"}