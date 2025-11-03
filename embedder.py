"""
embedder.py
------------
Generates embeddings for research paper chunks using SentenceTransformers.
Stores them locally in FAISS for fast retrieval during summarization.
"""

import os
import faiss
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer


# =============== Load Embedding Model ===============
def load_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads a SentenceTransformer model for text embeddings.
    """
    print(f"[INFO] Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)
    print("[INFO] Model loaded successfully.")
    return model


# =============== Create Embeddings ===============
def generate_embeddings(model, chunks: List[Dict]) -> np.ndarray:
    """
    Generates vector embeddings for each text chunk.
    """
    texts = [c["text"] for c in chunks]
    print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    print("[INFO] Embeddings generated.")
    return embeddings


# =============== Store in FAISS ===============
def store_embeddings(embeddings: np.ndarray, chunks: List[Dict], output_dir: str = "vector_store"):
    """
    Stores embeddings in a FAISS index along with metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(output_dir, "research_index.faiss"))

    # Save metadata
    meta_path = os.path.join(output_dir, "metadata.npy")
    np.save(meta_path, np.array(chunks, dtype=object))
    print(f"[INFO] Stored {len(chunks)} embeddings and metadata in '{output_dir}/'.")


# =============== Load Function (for Retrieval) ===============
def load_vector_store(output_dir: str = "vector_store"):
    """
    Loads FAISS index and metadata for retrieval.
    """
    index = faiss.read_index(os.path.join(output_dir, "research_index.faiss"))
    metadata = np.load(os.path.join(output_dir, "metadata.npy"), allow_pickle=True)
    print(f"[INFO] Loaded FAISS index and metadata from '{output_dir}/'.")
    return index, metadata


# =============== Wrapper for App Integration ===============
def create_vector_store(chunks: List[Dict], output_dir: str = "vector_store"):
    """
    Creates embeddings for chunks and saves them to FAISS for fast retrieval.

    Args:
        chunks: List of text chunks.
        output_dir: Directory to store FAISS index and metadata.

    Returns:
        tuple: (FAISS index, metadata)
    """
    model = load_model()
    embeddings = generate_embeddings(model, chunks)
    store_embeddings(embeddings, chunks, output_dir)
    index, metadata = load_vector_store(output_dir)
    return index, metadata
