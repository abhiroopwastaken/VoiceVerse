"""
RAG Pipeline Module
===================
Chunking (Semantic), embedding, FAISS indexing, and retrieval.
"""

import numpy as np
from typing import List, Tuple, Optional


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline using sentence-transformers + FAISS.
    """
    
    def __init__(self):
        from config import CHUNK_SIZE, CHUNK_OVERLAP, SEMANTIC_CHUNK_THRESHOLD, TOP_K_RETRIEVAL, RERANKER_TOP_K
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self.semantic_threshold = SEMANTIC_CHUNK_THRESHOLD
        self.initial_top_k = TOP_K_RETRIEVAL
        self.rerank_top_k = RERANKER_TOP_K
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self._model = None
    
    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from config import EMBEDDING_MODEL
            print(f"DEBUG: Loading embedding model: {EMBEDDING_MODEL}...")
            # Using BGE-M3 or similar high-quality model
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            print("DEBUG: Embedding model loaded.")
        return self._model
    
    def ingest(self, text: str) -> int:
        """
        Chunk text, compute embeddings, and build FAISS index.
        
        Returns:
            Number of chunks created.
        """
        import faiss
        
        # 1. Chunk the text (using semantic chunking)
        self.chunks = self._chunk_text_semantic(text)
        
        if not self.chunks:
            raise ValueError("No text chunks could be created from the document.")
        
        # 2. Compute embeddings
        print(f"DEBUG: Computing embeddings for {len(self.chunks)} chunks...")
        self.embeddings = self.model.encode(
            self.chunks,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        print("DEBUG: Embeddings computed.")
        
        # 3. Build FAISS index (Inner Product for normalized vectors = cosine sim)
        print("DEBUG: Building FAISS index...")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))
        print("DEBUG: FAISS index built.")
        
        return len(self.chunks)
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve top-k chunks using FAISS + Reranker.
        """
        if self.index is None:
            raise ValueError("No documents ingested. Call ingest() first.")
        
        # 1. Retrieval (Fetch more candidates for reranking)
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)
        
        scores, indices = self.index.search(query_emb, self.initial_top_k)
        
        candidate_indices = indices[0]
        candidates = []
        for idx in candidate_indices:
            if 0 <= idx < len(self.chunks):
                candidates.append(self.chunks[idx])
        
        if not candidates:
            return []

        # 2. Reranking (API)
        try:
            reranked_chunks = self._rerank(query, candidates, top_k=self.rerank_top_k)
            return reranked_chunks
        except Exception as e:
            print(f"Reranker failed: {e}. Falling back to standard FAISS results.")
            return candidates[:self.rerank_top_k]

    def _rerank(self, query: str, chunks: List[str], top_k: int) -> List[str]:
        """
        Re-rank candidates using a local Cross-Encoder model.
        Uses cross-encoder/ms-marco-MiniLM-L-6-v2 — fast, CPU-friendly, no API needed.
        Returns results ordered by their ORIGINAL DOCUMENT POSITION for narrative flow.
        """
        try:
            from sentence_transformers import CrossEncoder

            if not hasattr(self, '_cross_encoder') or self._cross_encoder is None:
                print("Loading local Cross-Encoder reranker (one-time)...")
                self._cross_encoder = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    max_length=512,
                )

            # Score all (query, chunk) pairs
            pairs = [[query, chunk] for chunk in chunks]
            scores = self._cross_encoder.predict(pairs)

            # Pair with original chunk indices (for position ordering)
            scored = []
            for chunk, score in zip(chunks, scores):
                # Find original position in self.chunks for document-order sorting
                try:
                    orig_pos = self.chunks.index(chunk)
                except ValueError:
                    orig_pos = 99999  # unknown position → end
                scored.append((chunk, float(score), orig_pos))

            # Keep top_k by relevance score
            top = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

            # Re-order kept chunks by original document position (narrative coherence)
            top_ordered = sorted(top, key=lambda x: x[2])

            return [chunk for chunk, score, pos in top_ordered]

        except Exception as e:
            print(f"Local reranker failed: {e}. Falling back to FAISS order.")
            return chunks[:top_k]



    
    def get_full_context(self, max_chunks: int = 10) -> str:
        """
        Return combined text from all (or top-N) chunks for script generation.
        """
        selected = self.chunks[:max_chunks]
        return "\n\n".join(selected)
    
    def get_relevant_context(self, style: str, k: int = 8, custom_focus: str = "") -> str:
        """
        Retrieve context relevant to the desired content style using multi-query retrieval.
        Runs multiple queries from different angles to maximise document coverage.
        Uses doc-size-aware k so small docs don't over-retrieve.
        """
        total_chunks = len(self.chunks)

        # ── Doc-size-aware k scaling ─────────────────────────────
        # Small doc (<10 chunks) → retrieve fewer per query to avoid repetition.
        # Large doc (>30 chunks) → retrieve more per query for better coverage.
        if total_chunks <= 5:
            effective_k = max(2, total_chunks)        # tiny doc: retrieve almost all
        elif total_chunks <= 15:
            effective_k = min(k, max(4, total_chunks // 2))  # small doc: ~half
        elif total_chunks <= 30:
            effective_k = k                           # medium doc: use default k
        else:
            effective_k = min(k + 4, total_chunks)   # large doc: a bit more
        # ─────────────────────────────────────────────────────────
        # Style-specific query angles to cover different facets of the document
        style_queries = {
            "podcast": [
                "What are the most interesting concepts, surprising facts, and debatable ideas?",
                "What examples, experiments, and case studies are mentioned?",
                "What are the key arguments, conclusions, and implications?",
            ],
            "narration": [
                "What is the main story, key developments, and narrative arc?",
                "What vivid details, examples, and turning points are described?",
                "What is the conclusion and broader significance?",
            ],
            "debate": [
                "What are the main arguments, counterarguments, and opposing views?",
                "What evidence, examples, and data support different perspectives?",
                "What are the unresolved questions and areas of disagreement?",
            ],
            "lecture": [
                "What are the key concepts, definitions, and theories explained?",
                "What examples, studies, and experiments illustrate the ideas?",
                "What are the practical applications and educational takeaways?",
            ],
            "storytelling": [
                "What are the most vivid scenes, characters, and dramatic moments?",
                "What conflicts, discoveries, and turning points drive the narrative?",
                "What is the deeper meaning or revelation?",
            ],
        }
        
        if custom_focus:
            queries = [custom_focus] + style_queries.get(style.lower(), [])[:2]
        else:
            queries = style_queries.get(style.lower(), ["What are the main topics and key information?"])
        
        # Run all queries and collect unique chunks
        seen = set()
        all_chunks = []
        for query in queries:
            try:
                retrieved = self.retrieve(query, k=effective_k)
                for chunk in retrieved:
                    key = chunk[:100]  # dedup by start of chunk
                    if key not in seen:
                        seen.add(key)
                        all_chunks.append(chunk)
            except Exception as e:
                print(f"Multi-query retrieval warning for '{query[:40]}': {e}")
        
        # If we got very little, fall back to full doc context
        if len(all_chunks) < 3:
            return self.get_full_context(max_chunks=15)
        
        return "\n\n---\n\n".join(all_chunks)

    
    @property
    def fast_model(self):
        """Lazy-load a small, fast model for semantic splitting logic."""
        if not hasattr(self, '_fast_model') or self._fast_model is None:
            from sentence_transformers import SentenceTransformer
            print("DEBUG: Loading fast model for semantic chunking (all-MiniLM-L6-v2)...")
            # "all-MiniLM-L6-v2" is extremely fast and good enough for determining sentence similarity breaks
            self._fast_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("DEBUG: Fast model loaded.")
        return self._fast_model

    def _chunk_text_semantic(self, text: str) -> List[str]:
        """
        Split text into chunks based on semantic similarity between sentences.
        Uses a fast model for the structural analysis to maintain performance.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []
            
        # Get using the FAST model for structure analysis
        embeddings = self.fast_model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        embeddings = embeddings.cpu().numpy()
        
        # Calculate cosine similarity between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i+1])
            similarities.append(sim)
            
        chunks = []
        current_chunk_sentences = [sentences[0]]
        
        for i in range(len(similarities)):
            sim = similarities[i]
            sent = sentences[i+1]
            
            # Current chunk length estimation
            current_len = sum(len(s) for s in current_chunk_sentences)
            
            # Refined Splitting Logic:
            # 1. HARD LIMIT: If current Chunk + Sentence > Max Size -> Split
            # 2. SOFT MERGE: If current Chunk is too small (< 200 chars) -> Keep appending (Merge)
            # 3. SEMANTIC SPLIT: If Sim < Threshold -> Split (only if chunk isn't tiny)
            
            is_hard_limit = (current_len + len(sent) > self.chunk_size)
            is_tiny_chunk = (current_len < 300) 
            is_topic_shift = (sim < self.semantic_threshold)
            
            if is_hard_limit:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sent]
            elif is_topic_shift and not is_tiny_chunk:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sent]
            else:
                current_chunk_sentences.append(sent)
                
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
        return chunks

    def _cosine_similarity(self, vec_a, vec_b):
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        return dot_product / (norm_a * norm_b + 1e-10)

    def _split_sentences(self, text: str) -> List[str]:
        """Robuster sentence splitter."""
        import re
        # Split on .?! followed by space or end of string, keeping the punctuation
        parts = re.split(r'([.?!])\s+', text)
        sentences = []
        current_sent = ""
        for part in parts:
            if part in ['.', '?', '!']:
                current_sent += part
                if current_sent.strip():
                    sentences.append(current_sent.strip())
                current_sent = ""
            else:
                current_sent += part
        
        if current_sent.strip():
            sentences.append(current_sent.strip())
            
        return sentences
