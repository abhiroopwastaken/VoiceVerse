
import unittest
import sys
import os
from unittest.mock import patch, MagicMock, PropertyMock

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Define a real exception for mocking
class MockGradioError(Exception):
    pass

# Mock gradio module with the Error class
mock_gradio = MagicMock()
mock_gradio.Error = MockGradioError
with patch.dict('sys.modules', {'gradio': mock_gradio}):
    import app

class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        # Reset global state
        app.rag_pipeline = None

    def test_process_documents(self):
        with patch.object(app, 'extract_from_multiple') as mock_extract, \
             patch.object(app, 'RAGPipeline') as mock_rag_cls:
            
            # Setup mocks
            mock_extract.return_value = ("Combined text content.", [{"filename": "test.txt", "type": "TXT", "word_count": 100, "pages": 1}])
            mock_pipeline = MagicMock()
            mock_pipeline.ingest.return_value = 5 # 5 chunks
            mock_rag_cls.return_value = mock_pipeline
            
            # Test input
            files = [MagicMock(name="/path/to/test.txt")]
            # mock name attribute
            type(files[0]).name = PropertyMock(return_value="/path/to/test.txt")
            # Also handle simple attribute access if logic uses it differently
            files[0].name = "/path/to/test.txt"
            
            # Run function
            summary, preview = app.process_documents(files)
            
            # Assertions
            self.assertIn("100 words", summary)
            self.assertIn("5 chunks", summary)
            self.assertEqual(preview, "Combined text content.")
            self.assertIsNotNone(app.rag_pipeline)

    def test_generate_content(self):
        with patch.object(app, 'generate_script') as mock_gen_script, \
             patch.object(app, 'generate_audio') as mock_gen_audio, \
             patch.object(app, 'merge_audio_segments') as mock_merge, \
             patch.object(app, 'cleanup_temp_files') as mock_cleanup, \
             patch.object(app, 'get_audio_duration') as mock_duration:
                 
            # Setup state
            app.rag_pipeline = MagicMock()
            app.rag_pipeline.get_relevant_context.return_value = "Context"
            
            # Setup mocks
            mock_gen_script.return_value = [("HOST_A", "Hello")]
            mock_gen_audio.return_value = ["/tmp/seg1.mp3"]
            mock_merge.return_value = "/tmp/final.mp3" 
            mock_duration.return_value = 10.5
            
            # Run function
            audio_path, script_display, status = app.generate_content("Podcast", "", 0, 0)
            
            # Verify calls
            mock_gen_script.assert_called_once()
            mock_gen_audio.assert_called_once()
            mock_merge.assert_called_once()
            
            # The result of generate_content is whatever merge_audio_segments returns
            self.assertEqual(audio_path, "/tmp/final.mp3")
            self.assertIn("Host A", script_display)
            self.assertIn("10.5 seconds", status)

if __name__ == '__main__':
    unittest.main()
