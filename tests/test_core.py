
import unittest
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock

# Ensure modules are in path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.document_loader import extract_text, extract_from_multiple
from modules.rag_pipeline import RAGPipeline
from modules.script_generator import generate_script, _parse_script
from modules.voice_generator import generate_audio
from modules.audio_utils import merge_audio_segments

class TestDocumentLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_extract_txt(self):
        file_path = os.path.join(self.test_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("Hello world.\nThis is a test document.")
        
        text, metadata = extract_text(file_path)
        self.assertIn("Hello world", text)
        self.assertEqual(metadata["type"], "TXT")

    def test_extract_multiple(self):
        fp1 = os.path.join(self.test_dir, "test1.txt")
        fp2 = os.path.join(self.test_dir, "test2.txt")
        with open(fp1, "w") as f: f.write("Doc 1 content")
        with open(fp2, "w") as f: f.write("Doc 2 content")
        
        combined, meta = extract_from_multiple([fp1, fp2])
        self.assertIn("Doc 1 content", combined)
        self.assertIn("Doc 2 content", combined)
        self.assertEqual(len(meta), 2)

class TestRAGPipeline(unittest.TestCase):
    def test_chunking(self):
        pipeline = RAGPipeline(chunk_size=50, chunk_overlap=0)
        text = "This is a long sentence that should ideally be split into multiple chunks because it exceeds the limit."
        chunks = pipeline._chunk_text(text)
        self.assertTrue(len(chunks) > 0)
        
    def test_ingest_and_retrieve(self):
        # Patch sentence-transformers at the source
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock()}):
            # Force small chunks to ensure we get multiple chunks
            pipeline = RAGPipeline(chunk_size=10, chunk_overlap=0)
            # Mock the internal model directly since it's lazy loaded
            mock_model = MagicMock()
            mock_model.encode.return_value = import_numpy().random.rand(2, 384).astype('float32')
            pipeline._model = mock_model
            
            text = "Snippet one.\n\nSnippet two."
            # We need to mock faiss too since it's imported inside ingest
            with patch.dict('sys.modules', {'faiss': MagicMock()}):
                count = pipeline.ingest(text)
                self.assertEqual(count, 2)
                
                # Test retrieve
                # Mock index search
                pipeline.index = MagicMock()
                pipeline.index.search.return_value = (import_numpy().array([[0.1]]), import_numpy().array([[0]]))
                
                results = pipeline.retrieve("query", k=1)
                self.assertEqual(len(results), 1)

class TestScriptGenerator(unittest.TestCase):
    def test_parse_script(self):
        raw = """
        HOST_A: Welcome to the show.
        HOST_B: Thanks for having me.
        This is a continuation line.
        HOST_A: Let's start.
        """
        parsed = _parse_script(raw, "podcast")
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0], ("HOST_A", "Welcome to the show."))
        self.assertEqual(parsed[1], ("HOST_B", "Thanks for having me. This is a continuation line."))
        self.assertEqual(parsed[2], ("HOST_A", "Let's start."))

    def test_generate_script_mock(self):
        # Patch HF_TOKEN to avoid ValueError
        with patch('modules.script_generator.HF_TOKEN', "dummy_token"):
            mock_client = MagicMock()
            mock_client.chat_completion.return_value.choices[0].message.content = "NARRATOR: This is a generated story."
            
            with patch.dict('sys.modules', {'huggingface_hub': MagicMock(InferenceClient=MagicMock(return_value=mock_client))}):
                script = generate_script("context", "narration")
                self.assertEqual(len(script), 1)
                self.assertEqual(script[0], ("NARRATOR", "This is a generated story."))

class TestVoiceGenerator(unittest.TestCase):
    def test_generate_audio(self):
        # Mock edge_tts module since it's imported inside the function
        mock_edge = MagicMock()
        
        # Async mock for Communicate.save
        async def mock_save(*args, **kwargs):
            pass
            
        mock_communicate = MagicMock()
        mock_communicate.save = mock_save
        mock_edge.Communicate.return_value = mock_communicate
        
        with patch.dict('sys.modules', {'edge_tts': mock_edge}):
            script = [("HOST_A", "Hello testing.")]
            # We need to mock the internal loop handling if necessary, or just run it.
            # generate_audio handles the loop.
            
            paths = generate_audio(script)
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].endswith(".mp3"))
            self.assertTrue(os.path.dirname(paths[0]).startswith(tempfile.gettempdir()))

def import_numpy():
    import numpy
    return numpy

if __name__ == '__main__':
    unittest.main()
