from sentence_transformers import SentenceTransformer, util

# Load embedding model (fast + lightweight)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(transcription_segments):
    """
    Builds an embedding index for the transcript segments.
    """
    texts = [seg["text"] for seg in transcription_segments]
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    return texts, embeddings

def search_segments(query, texts, embeddings, transcription_segments, top_k=5):
    """
    Hybrid search:
    1. Return exact keyword matches first (if any).
    2. Fill remaining slots with semantic matches.
    """
    results = []

    # Step 1: Exact matches
    exact_matches = []
    for i, text in enumerate(texts):
        if query.lower() in text.lower():
            exact_matches.append({
                "score": 1.0,  # force exact matches to top
                "text": text,
                "start": transcription_segments[i]["start"],
                "end": transcription_segments[i]["end"]
            })

    results.extend(exact_matches)

    # Step 2: Semantic matches
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)[0]

    for hit in hits:
        idx = hit["corpus_id"]
        # Skip if this was already an exact match
        if any(idx == texts.index(m["text"]) for m in exact_matches):
            continue
        results.append({
            "score": float(hit["score"]),
            "text": texts[idx],
            "start": transcription_segments[idx]["start"],
            "end": transcription_segments[idx]["end"]
        })
        if len(results) >= top_k:  # stop once we have enough results
            break

    return results
