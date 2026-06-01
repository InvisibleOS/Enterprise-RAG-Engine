import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add the workspace directory to the path so python can import rag_engine
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rag_engine import EnterpriseBrain

class TestEnterpriseBrain(unittest.TestCase):
    def test_chunk_text(self):
        brain = EnterpriseBrain(google_api_key="dummy_google_key")
        # Text length is 1500 chars. With chunk_size=1000 and overlap=200:
        # First chunk: 0 to 1000
        # Next start = 1000 - 200 = 800
        # Second chunk: 800 to 1500 (length 700)
        text = "a" * 1500
        chunks = brain._chunk_text(text, chunk_size=1000, overlap=200)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 1000)
        self.assertEqual(len(chunks[1]), 700)

    @patch('rag_engine.Pinecone')
    @patch('rag_engine.PdfReader')
    def test_ingest_pdf_empty_text(self, mock_pdf_reader, mock_pinecone):
        # Setup mocks
        mock_pc = MagicMock()
        mock_pinecone.return_value = mock_pc
        mock_pc.list_indexes.return_value = []
        
        mock_reader = MagicMock()
        mock_pdf_reader.return_value = mock_reader
        # Mock pdf having no pages or empty text
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader.pages = [mock_page]

        brain = EnterpriseBrain(google_api_key="dummy_google_key")
        result = brain.ingest_pdf("dummy.pdf", pinecone_api_key="dummy_pinecone_key")
        
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("No text could be extracted", result)

    @patch('rag_engine.Pinecone')
    @patch('rag_engine.PdfReader')
    def test_ingest_pdf_success(self, mock_pdf_reader, mock_pinecone):
        # Setup Pinecone mock
        mock_pc = MagicMock()
        mock_pinecone.return_value = mock_pc
        
        # Mock index list (already exists)
        mock_index_info = MagicMock()
        mock_index_info.name = "rag-resume-project"
        mock_pc.list_indexes.return_value = [mock_index_info]
        
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        
        # Mock PdfReader
        mock_reader = MagicMock()
        mock_pdf_reader.return_value = mock_reader
        
        mock_page_1 = MagicMock()
        mock_page_1.extract_text.return_value = "Page 1 Content " * 100 # 1500 chars
        mock_page_2 = MagicMock()
        mock_page_2.extract_text.return_value = "Page 2 Content " * 100 # 1500 chars
        mock_reader.pages = [mock_page_1, mock_page_2]

        brain = EnterpriseBrain(google_api_key="dummy_google_key")
        
        # Mock Google GenAI client
        mock_client = MagicMock()
        brain.client = mock_client
        
        # Mock embed_content response
        mock_embedding_response = MagicMock()
        mock_emb = MagicMock()
        mock_emb.values = [0.1] * 768
        mock_embedding_response.embeddings = [mock_emb]
        mock_client.models.embed_content.return_value = mock_embedding_response

        # Execute ingestion
        result = brain.ingest_pdf("dummy.pdf", pinecone_api_key="dummy_pinecone_key")
        
        # Verify success and check status
        self.assertTrue(result.startswith("Success:"), f"Expected success but got: {result}")
        
        # Verify that Pinecone upsert was called
        self.assertTrue(mock_index.upsert.called)
        
        # Retrieve calls to upsert to check metadata contains page numbers
        upsert_args = mock_index.upsert.call_args
        vectors = upsert_args[1].get('vectors') or upsert_args[0][0]
        
        # We should have vectors with page metadata
        self.assertTrue(len(vectors) > 0)
        # Check first vector metadata contains correct page index (0 for first page chunks)
        first_vec_metadata = vectors[0][2]
        self.assertEqual(first_vec_metadata['page'], 0)
        
        # Last vector metadata (from page 2) should contain page index 1
        last_vec_metadata = vectors[-1][2]
        self.assertEqual(last_vec_metadata['page'], 1)

    @patch('rag_engine.Pinecone')
    def test_ask(self, mock_pinecone):
        # Setup Pinecone index query mock
        mock_pc = MagicMock()
        mock_pinecone.return_value = mock_pc
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        
        # Mock Pinecone query results
        mock_index.query.return_value = {
            "matches": [
                {
                    "metadata": {"text": "This is page 1 content", "page": 0},
                    "score": 0.9,
                    "id": "vec_0"
                },
                {
                    "metadata": {"text": "This is page 2 content", "page": 1},
                    "score": 0.8,
                    "id": "vec_2"
                }
            ]
        }
        
        brain = EnterpriseBrain(google_api_key="dummy_google_key")
        brain.index = mock_index
        
        # Mock Google GenAI client
        mock_client = MagicMock()
        brain.client = mock_client
        
        # Mock embedding response for query
        mock_embedding_response = MagicMock()
        mock_emb = MagicMock()
        mock_emb.values = [0.1] * 768
        mock_embedding_response.embeddings = [mock_emb]
        mock_client.models.embed_content.return_value = mock_embedding_response
        
        # Mock generate_content response
        mock_generate_response = MagicMock()
        mock_generate_response.text = "This is the generated answer."
        mock_client.models.generate_content.return_value = mock_generate_response

        # Execute query
        result = brain.ask("dummy query")
        
        # Verify response
        self.assertEqual(result["result"], "This is the generated answer.")
        self.assertEqual(len(result["source_documents"]), 2)
        
        # Verify page numbers in source documents
        self.assertEqual(result["source_documents"][0].metadata["page"], 0)
        self.assertEqual(result["source_documents"][0].page_content, "This is page 1 content")
        self.assertEqual(result["source_documents"][1].metadata["page"], 1)
        self.assertEqual(result["source_documents"][1].page_content, "This is page 2 content")

if __name__ == '__main__':
    unittest.main()
