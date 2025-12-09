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
        self.model_name = "gemini-1.5-flash"
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
                embedding = response.embeddings[0].values
                metadata = {"text": chunk_text, "id": str(i)}
                vectors_to_upsert.append((f"vec_{i}", embedding, metadata))

            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i : i + batch_size]
                self.index.upsert(vectors=batch)
            return f"Success: Manually processed {len(text_chunks)} chunks."
        except Exception as e:
            return f"Error: {str(e)}"

    def ask(self, query: str) -> dict:
        if not self.index: return {"result": "Please upload a document first."}
        try:
            response = self.client.models.embed_content(
                model="text-embedding-004",
                contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            query_vector = response.embeddings[0].values
            search_results = self.index.query(vector=query_vector, top_k=3, include_metadata=True)
            
            context_text = ""
            source_docs = []
            for match in search_results['matches']:
                text = match['metadata']['text']
                score = match['score']
                context_text += f"---\n{text}\n"
                class MockDoc: pass
                doc = MockDoc()
                doc.page_content = text
                doc.metadata = {'score': score}
                source_docs.append(doc)

            prompt = f"Answer the user's question using ONLY the context below.\n\nContext:\n{context_text}\n\nQuestion: {query}"
            ai_response = self.client.models.generate_content(model=self.model_name, contents=prompt)
            return {"result": ai_response.text, "source_documents": source_docs}
        except Exception as e:
            return {"result": f"Error: {str(e)}"}