
import sys
import os
import os.path

# Add current directory to path
sys.path.append(os.path.abspath(os.curdir))

from app import process_documents
import gradio as gr

def test_process():
    print("Starting integration test for process_documents...")
    try:
        # Test with direct text input
        text_input = "This is a test document for VoiceVerse. It should be processed correctly by the RAG pipeline."
        files = []
        
        print("Testing direct text processing...")
        summary, content_summary = process_documents(files, text_input)
        
        print("\n--- Summary ---")
        print(summary)
        print("\n--- Content Summary ---")
        print(content_summary)
        
        if "Total: 16 words" in summary and "chunks" in summary:
            print("\nSUCCESS: Text processing verified.")
        else:
            print("\nFAILURE: Summary output format unexpected.")
            
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_process()
