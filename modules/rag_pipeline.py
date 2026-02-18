"""
RAG Pipeline Module
===================
Chunking, embedding, FAISS indexing, and retrieval.
"""

import numpy as np
from typing import List, Tuple, Optional


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline using sentence-transformers + FAISS.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self._model = None
    
    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model
    
    def ingest(self, text: str) -> int:
        """
        Chunk text, compute embeddings, and build FAISS index.
        
        Returns:
            Number of chunks created.
        """
        import faiss
        
        # 1. Chunk the text
        self.chunks = self._chunk_text(text)
        
        if not self.chunks:
            raise ValueError("No text chunks could be created from the document.")
        
        # 2. Compute embeddings
        self.embeddings = self.model.encode(
            self.chunks,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        
        # 3. Build FAISS index (Inner Product for normalized vectors = cosine sim)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))
        
        return len(self.chunks)
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve top-k most relevant chunks for a query.
        """
        if self.index is None:
            raise ValueError("No documents ingested. Call ingest() first.")
        
        k = min(k, len(self.chunks))
        
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)
        
        scores, indices = self.index.search(query_emb, k)
        
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
        
        return results
    
    def get_full_context(self, max_chunks: int = 10) -> str:
        """
        Return combined text from all (or top-N) chunks for script generation.
        """
        selected = self.chunks[:max_chunks]
        return "\n\n".join(selected)
    
    def get_relevant_context(self, style: str, k: int = 8) -> str:
        """
        Retrieve context relevant to the desired content style.
        Uses style-specific queries to get the most suitable chunks.
        """
        style_queries = {
            "podcast": "What are the key discussion points, interesting facts, and debatable topics?",
            "narration": "What is the main narrative, key events, and important details?",
            "debate": "What are the controversial points, different perspectives, and arguments?",
            "lecture": "What are the key concepts, definitions, examples, and educational takeaways?",
            "storytelling": "What are the interesting characters, events, conflicts, and story elements?",
        }
        
        query = style_queries.get(style.lower(), "What are the main topics and key information?")
        
        retrieved = self.retrieve(query, k=k)
        return "\n\n".join(retrieved)
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks using paragraph-aware splitting.
        """
        # First split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) + 2 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If a single paragraph is too long, split it by sentences
                if len(para) > self.chunk_size:
                    sentences = self._split_sentences(para)
                    sub_chunk = ""
                    for sent in sentences:
                        if len(sub_chunk) + len(sent) + 1 > self.chunk_size:
                            if sub_chunk:
                                chunks.append(sub_chunk.strip())
                            sub_chunk = sent
                        else:
                            sub_chunk += " " + sent if sub_chunk else sent
                    if sub_chunk:
                        current_chunk = sub_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Add overlapping context
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-self.chunk_overlap:]
                overlapped.append(prev_tail + " " + chunks[i])
            chunks = overlapped
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]
