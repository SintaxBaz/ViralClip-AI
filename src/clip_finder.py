# src/clip_finder.py
from sentence_transformers import SentenceTransformer, util

class ClipFinder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the semantic + keyword search engine for transcripts.
        """
        self.embedder = SentenceTransformer(model_name)
        self.texts = None
        self.embeddings = None
        self.transcription_segments = None

    def build_index(self, transcription_segments):
        """
        Builds embeddings for all transcript segments.
        """
        self.transcription_segments = transcription_segments
        self.texts = [seg["text"] for seg in transcription_segments]
        self.embeddings = self.embedder.encode(self.texts, convert_to_tensor=True)

    def search_segments(self, transcription_segments, query, top_k=5):
        """
        Searches transcript segments for a query (keyword + semantic).
        
        Args:
            transcription_segments (list[dict]): transcript segments with "text", "start", "end".
            query (str): Search keyword or phrase.
            top_k (int): Number of results to return.

        Returns:
            list[dict]: Matching transcript segments with scores.
        """
        # Ensure index exists
        if self.embeddings is None:
            self.build_index(transcription_segments)

        results = []

        # --- Step 1: Exact matches ---
        exact_matches = []
        for i, text in enumerate(self.texts):
            if query.lower() in text.lower():
                exact_matches.append({
                    "score": 1.0,
                    "text": text,
                    "start": self.transcription_segments[i]["start"],
                    "end": self.transcription_segments[i]["end"]
                })

        results.extend(exact_matches)

        # --- Step 2: Semantic matches ---
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)[0]

        for hit in hits:
            idx = hit["corpus_id"]

            # skip if already an exact match
            if any(idx == self.texts.index(m["text"]) for m in exact_matches):
                continue

            results.append({
                "score": float(hit["score"]),
                "text": self.texts[idx],
                "start": self.transcription_segments[idx]["start"],
                "end": self.transcription_segments[idx]["end"]
            })

            if len(results) >= top_k:
                break

        return results
