"""
retriever.py
-------------
Retrieves top-k relevant text chunks from the FAISS vector store
for a given query. Supports optional re-ranking using a Cross-Encoder.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Tuple


# =============== Load Vector Store ===============
def load_vector_store(output_dir: str = "vector_store"):
    """
    Loads FAISS index and metadata from disk.
    """
    index = faiss.read_index(f"{output_dir}/research_index.faiss")
    metadata = np.load(f"{output_dir}/metadata.npy", allow_pickle=True)
    print(f"[INFO] Loaded FAISS index and metadata from '{output_dir}/'.")
    return index, metadata


# =============== Load Embedding Model ===============
def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads the same embedding model used in embedder.py.
    """
    print(f"[INFO] Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


# =============== Retrieve Top-k ===============
def retrieve_top_k(
    query: str,
    index,
    metadata,
    embedder,
    k: int = 5
) -> List[Dict]:
    """
    Retrieves top-k chunks from the FAISS index given a text query.
    """
    query_vec = embedder.encode([query], convert_to_numpy=True)
    scores, indices = index.search(query_vec, k)

    retrieved = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            chunk_info = metadata[idx]  # ✅ fixed
            retrieved.append({
                "rank": i + 1,
                "score": float(scores[0][i]),
                "text": chunk_info["text"],
                "section": chunk_info["section"],
                "page": chunk_info["page"]
            })
    return retrieved


# =============== Optional Re-ranking ===============
def rerank_chunks(query: str, retrieved: List[Dict], model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> List[Dict]:
    """
    Re-ranks retrieved chunks using a Cross-Encoder (semantic re-ranking).
    """
    print(f"[INFO] Re-ranking {len(retrieved)} chunks...")
    cross_encoder = CrossEncoder(model_name)
    pairs = [(query, chunk["text"]) for chunk in retrieved]
    scores = cross_encoder.predict(pairs)

    for i, s in enumerate(scores):
        retrieved[i]["rerank_score"] = float(s)

    retrieved = sorted(retrieved, key=lambda x: x["rerank_score"], reverse=True)
    return retrieved


# =============== Pretty Print ===============
def show_results(retrieved: List[Dict], n: int = 3):
    for i, item in enumerate(retrieved[:n]):
        print(f"\n🔹 Rank {item['rank']}, Score: {item.get('rerank_score', item['score']):.4f}")
        print(f"Section: {item['section']} (Page {item['page']})")
        print(item['text'][:400], "...")


# =============== Test Run ===============
if __name__ == "__main__":
    # Load index and model
    index, metadata = load_vector_store("vector_store")
    embedder = load_embedder()

    # Query
    query = "What is the proposed method in the paper?"
    print(f"\n[QUERY] {query}")

    retrieved = retrieve_top_k(query, index, metadata, embedder, k=10)
    show_results(retrieved, 3)

    # Optional: re-rank results for better accuracy
    reranked = rerank_chunks(query, retrieved)
    show_results(reranked, 3)

