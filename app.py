"""
Enhanced Research Summarizer API (Flask + RAG + Gemini)
-------------------------------------------------------
Now API-only (no HTML), compatible with React frontend.

Endpoints:
- /upload          → Upload & process PDF
- /summarize       → Query-based summarization
- /citations       → Extract citations
- /compare         → Compare abstracts
- /papers          → List uploaded PDFs
- /delete/<file>   → Delete PDF
- /cluster         → Cluster topics
"""

import os
import re
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from dotenv import load_dotenv

# Optional imports
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher

# Local imports
from extractor import extract_pdf_sections
from chunker import create_chunks
from embedder import create_vector_store
from retriever import load_vector_store, load_embedder, retrieve_top_k, rerank_chunks
from summarizer import SummarizerLLM

# ==================== Setup ====================

load_dotenv()
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend (http://localhost:3000 by default)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is missing. Please add it to your .env file.")

ALLOWED_EXTENSIONS = {"pdf"}


# ==================== Helpers ====================

def allowed_file(filename: str) -> bool:
    """Check if uploaded file is a valid PDF."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_citations(text: str):
    """Extract numeric and author-year style citations using regex."""
    # Normalize dashes and spaces from PDF or MS Word text
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u00A0", " ")

    pattern = r"""
        # Numeric citations like [1], [12–14]
        \[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]|
        
        # Author-year citations like (Benson, 2001)
        \([A-Z][A-Za-zÀ-ÿ\-\s&\.]+,\s?\d{4}[a-z]?(?:,\s?p{1,2}\.?\s?\d+(?:-\d+)?)?\)|

        # Et al. citations like (Benson et al., 2001)
        \([A-Z][A-Za-zÀ-ÿ\-\s&\.]+\s+et\s+al\.,\s?\d{4}[a-z]?\)
    """

    matches = re.findall(pattern, text, flags=re.VERBOSE)
    citations = sorted(set(m.strip() for m in matches if m.strip()))
    return citations


def compare_abstracts(original: str, generated: str) -> float:
    """Compare abstract and generated summary using lexical similarity."""
    ratio = SequenceMatcher(None, original, generated).ratio()
    return round(ratio * 100, 2)


# ==================== Routes ====================

@app.route("/")
def root():
    """API health check."""
    return jsonify({
        "status": "✅ Research Summarizer API running",
        "endpoints": ["/upload", "/summarize", "/citations", "/compare", "/papers", "/delete/<filename>", "/cluster"]
    })


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle PDF upload, chunking, and vector store creation."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            sections = extract_pdf_sections(filepath)
            if not sections:
                return jsonify({"error": "No extractable text found in PDF."}), 400

            chunks = create_chunks(sections)
            create_vector_store(chunks, VECTOR_STORE_PATH)

            return jsonify({
                "message": "✅ File processed successfully.",
                "filename": filename,
                "chunks": len(chunks)
            }), 200

        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return jsonify({"error": "Invalid file type. Only PDF allowed."}), 400


@app.route("/summarize", methods=["POST"])
def summarize():
    """Run query-based summarization using RAG + Gemini."""
    data = request.get_json(force=True)
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query is required."}), 400

    try:
        index, metadata = load_vector_store(VECTOR_STORE_PATH)
        embedder = load_embedder()

        retrieved = retrieve_top_k(query, index, metadata, embedder, k=15)
        retrieved = rerank_chunks(query, retrieved)

        summarizer = SummarizerLLM(use_gemini=True)
        summary = summarizer.summarize(query, retrieved)

        citations = extract_citations(summary)

        return jsonify({
            "query": query,
            "summary": summary,
            "context_count": len(retrieved),
            "citations_found": citations
        }), 200
    except Exception as e:
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500


@app.route("/citations", methods=["POST"])
def citations():
    """Extract citations from uploaded paper or summary text."""
    data = request.get_json(force=True)
    text = data.get("text", "")
    citations = extract_citations(text)
    return jsonify({"citations": citations, "count": len(citations)})


@app.route("/compare", methods=["POST"])
def compare():
    """Compare original abstract vs AI-generated summary."""
    data = request.get_json(force=True)
    original = data.get("original", "")
    generated = data.get("generated", "")
    score = compare_abstracts(original, generated)
    return jsonify({"similarity_percent": score})


@app.route("/papers", methods=["GET"])
def list_papers():
    """Dashboard: list all uploaded research papers."""
    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".pdf")]
    return jsonify({"uploaded_papers": files})


@app.route("/delete/<filename>", methods=["DELETE"])
def delete_paper(filename):
    """Delete a specific uploaded research paper."""
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"message": f"🗑️ Deleted {filename}"}), 200
    return jsonify({"error": "File not found"}), 404


@app.route("/cluster", methods=["GET"])
def cluster_topics():
    """Cluster text chunks into topics and show top keywords."""
    try:
        index, metadata = load_vector_store(VECTOR_STORE_PATH)
        texts = [m["text"] for m in metadata]

        if not texts:
            return jsonify({"error": "No text data found for clustering."}), 400

        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        X = vectorizer.fit_transform(texts)

        n_clusters = min(5, len(texts))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)

        top_terms = []
        terms = vectorizer.get_feature_names_out()
        for i in range(n_clusters):
            center = kmeans.cluster_centers_[i]
            top_indices = center.argsort()[-8:][::-1]
            top_terms.append([terms[j] for j in top_indices])

        return jsonify({"clusters": top_terms}), 200

    except Exception as e:
        return jsonify({"error": f"Clustering failed: {str(e)}"}), 500


# ==================== Run ====================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 API running at http://127.0.0.1:{port}")
    app.run(debug=True, host="0.0.0.0", port=port)
