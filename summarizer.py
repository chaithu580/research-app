"""
summarizer.py (Enhanced Gemini + RAG version)
---------------------------------------------
Generates research summaries or query-based answers using RAG (Retriever-Augmented Generation).
Includes enhancements:
 - Citation Extraction (Regex + NLP)
 - Abstract Comparison (Semantic similarity)
 - Query-based Summarization
 - Dashboard Metadata Hooks
 - Topic Clustering & Keyword Visualization
"""

import os
import re
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from retriever import load_vector_store, load_embedder, retrieve_top_k, rerank_chunks

# Optional imports
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    import spacy
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    spacy = None
    util = None

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    import matplotlib.pyplot as plt
except ImportError:
    KMeans = None

# ==========================================================
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# ==========================================================


# ==========================================================
#  LLM Summarizer Class
# ==========================================================
class SummarizerLLM:
    def __init__(
        self,
        use_gemini: bool = True,
        model_name: str = "gemini-2.5-flash-lite",
        local_model: str = "facebook/bart-large-cnn",
    ):
        self.use_gemini = use_gemini
        if use_gemini:
            if not genai:
                raise ImportError("Install Google Gemini SDK: pip install google-generativeai")
            if not GEMINI_API_KEY:
                raise ValueError("❌ GEMINI_API_KEY not set in .env file.")
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(model_name)
            print(f"[INFO] Using Google Gemini model: {model_name}")
        else:
            if not pipeline:
                raise ImportError("Install transformers: pip install transformers")
            print(f"[INFO] Loading local HuggingFace model: {local_model}")
            self.model = pipeline("text2text-generation", model=local_model, max_new_tokens=512)

    # ----------------------------------------------------------
    def summarize(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generate a concise summary or answer using retrieved context.
        """
        context_text = "\n\n".join([c["text"] for c in context_chunks])
        prompt = f"""
You are an expert research assistant.
Using the following extracted context, answer the query clearly and concisely.

Query:
{query}

Context:
{context_text}

Your summary should focus on:
- Research problem & motivation
- Methodology or approach
- Key findings or results
- Overall contribution

Important:
- Add in-text citations (e.g., “(Smith et al., 2020)” or “[1]”) wherever relevant to support the information.
- Ensure citations correspond to the sources mentioned in the provided context.
- Do NOT include citations in the abstract section if one is generated.
"""

        if self.use_gemini:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        else:
            result = self.model(prompt)[0]["generated_text"]
            return result.strip()


# ==========================================================
#  Enhancements
# ==========================================================

import re
from typing import List, Tuple

# Helper: normalize unicode quirks commonly produced by PDF/text extraction
def _normalize_text_for_citations(text: str) -> str:
    # convert non-breaking spaces to normal spaces
    text = text.replace('\u00A0', ' ')
    # normalize different dashes to hyphen
    text = text.replace('\u2013', '-')  # en-dash
    text = text.replace('\u2014', '-')  # em-dash
    # normalize smart quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    return text

def extract_citations(text: str, return_spans: bool = False) -> List:
    """
    Extract citations from academic text.

    Supports:
    ✅ Numeric citations: [1], [2,3], [2–4]
    ✅ (Author, 2020), (Author & Author, 2021)
    ✅ Inline: Author (2020), Author and Author (2020)
    ✅ et al. formats: (Smith et al., 2020), Smith et al. (2020)
    ✅ Bibliography lines: Benson, P. (2001)
    """

    def _normalize_text_for_citations(t: str) -> str:
        return re.sub(r'\s+', ' ', t.replace("\n", " ").replace("\r", " ")).strip()

    text = _normalize_text_for_citations(text)

    # 1) Numeric citations [2], [2-4], [2–4], [2,3]
    numeric_pattern = r"""
        \[\s*\d+(?:\s*(?:[,;-]\s*|\s*-\s*|\s*–\s*)\d+)*(?:\s*(?:,\s*\d+)*)?\s*\]
    """

    # 2) (Author, Year...), multi-author allowed
    author_year_pattern = r"""
        \(
            \s*
            (?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`\.&\-\s]+?)
            (?:\s*(?:;|and|&)\s*[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`\.&\-\s]+?)*
            \s*,\s*
            \d{4}[a-z]?
            (?:\s*,\s*(?:pp?\.?\s*)?\d+(?:\s*-\s*\d+)?)?
            (?:\s*;\s*[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`\.&\-\s]+?,\s*\d{4}[a-z]?
                (?:\s*,\s*(?:pp?\.?\s*)?\d+(?:\s*-\s*\d+)?)? )*
            \s*
        \)
    """

    # 3) Bibliography style: Benson, P. (2001)
    bib_entry_pattern = r"""
        (?:^|\n|\r|\. )
        \s*
        [A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+
        (?:,\s*[A-Z]\.)+
        (?:\s*(?:&|and)\s*[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+,\s*[A-Z]\.)*
        \s*\(\s*\d{4}[a-z]?\s*\)
    """

    # 4) Inline: Littlewood (1996)
    author_year_inline_pattern = r"""
        \b
        [A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+
        (?:\s+(?:and|&)\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+)?
        \s*\(\s*\d{4}[a-z]?\s*\)
    """

    # 5) Inline et al.: Smith et al. (2020)
    author_et_al_inline_pattern = r"""
        \b
        [A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+
        \s+et\s+al\.?\s*
        \(\s*\d{4}[a-z]?\s*\)
    """

    # 6) Parenthetical et al.: (Smith et al., 2020)
    author_et_al_parenthetical_pattern = r"""
        \(
            \s*
            [A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+
            \s+et\s+al\.?
            \s*,?\s*
            \d{4}[a-z]?
            (?:\s*,\s*(?:pp?\.?\s*)?\d+(?:\s*-\s*\d+)?)?
            \s*
        \)
    """

    # Compile all citation patterns
    combined = f"""
        (?:{numeric_pattern})
        | (?:{author_year_pattern})
        | (?:{bib_entry_pattern})
        | (?:{author_year_inline_pattern})
        | (?:{author_et_al_inline_pattern})
        | (?:{author_et_al_parenthetical_pattern})
    """

    regex = re.compile(combined, re.VERBOSE | re.UNICODE | re.IGNORECASE)

    matches: List[Tuple[str,int,int]] = []
    for m in regex.finditer(text):
        matched = m.group(0).strip()
        if len(matched) < 3:
            continue
        matches.append((matched, m.start(), m.end()))

    def _clean(s: str) -> str:
        return re.sub(r'\s+', ' ', s).strip()

    unique = {}
    for s, a, b in matches:
        key = _clean(s)
        if key not in unique:
            unique[key] = (key, a, b)

    results = list(unique.values())
    results.sort(key=lambda t: t[1])

    return results if return_spans else [t[0] for t in results]

def compare_abstracts(original: str, generated: str) -> Dict[str, float]:
    """
    Compare similarity between original abstract and generated summary.
    """
    if not util:
        print("[WARN] sentence-transformers not installed, skipping similarity comparison.")
        return {"similarity": 0.0, "keyword_overlap": 0.0}

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    emb1, emb2 = embedder.encode(original, convert_to_tensor=True), embedder.encode(generated, convert_to_tensor=True)
    similarity = float(util.cos_sim(emb1, emb2))
    overlap = len(set(original.lower().split()) & set(generated.lower().split())) / len(set(original.split()))
    return {"similarity": round(similarity * 100, 2), "keyword_overlap": round(overlap * 100, 2)}


def cluster_topics(texts: List[str], n_clusters: int = 5):
    """
    Cluster topics from multiple paper summaries.
    """
    if not KMeans:
        print("[WARN] sklearn/matplotlib not installed, skipping clustering.")
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    terms = vectorizer.get_feature_names_out()
    top_terms = {}
    for i in range(n_clusters):
        cluster_terms = X[labels == i].toarray().sum(axis=0)
        top_words = [terms[t] for t in cluster_terms.argsort()[-10:]]
        top_terms[f"Cluster {i+1}"] = top_words
    return top_terms


def visualize_clusters(topics: Dict[str, List[str]]):
    """
    Simple word cluster visualization.
    """
    if not plt:
        print("[WARN] matplotlib not installed, skipping visualization.")
        return
    plt.figure(figsize=(8, 4))
    for i, (cluster, words) in enumerate(topics.items()):
        plt.barh([f"{cluster}"], [len(words)])
    plt.title("Topic Clusters by Word Count")
    plt.show()


# ==========================================================
#  Main RAG Summarization with Enhancements
# ==========================================================
def generate_research_summary(
    query: str,
    top_k: int = 5,
    rerank: bool = True,
    use_gemini: bool = True,
    original_abstract: str = None,
    dashboard_mode: bool = False,
):
    """
    Full RAG pipeline: retrieve → rerank → summarize → enhance.
    """
    print("[INFO] Loading vector store and embedder...")
    index, metadata = load_vector_store("vector_store")
    embedder = load_embedder()

    print(f"[INFO] Retrieving top {top_k} chunks...")
    retrieved = retrieve_top_k(query, index, metadata, embedder, k=top_k)
    if rerank:
        print("[INFO] Reranking chunks...")
        retrieved = rerank_chunks(query, retrieved)

    summarizer = SummarizerLLM(use_gemini=use_gemini)
    summary = summarizer.summarize(query, retrieved)

    # ---- ✨ Enhancements ----
    # 1️⃣ Citation Extraction
    citations = extract_citations(summary)
    print(f"[INFO] Citations found: {len(citations)}")
    for i, cite in enumerate(citations, 1):
        print(f"{i}. {cite}")

    # 2️⃣ Abstract Comparison (optional)
    comparison = {"similarity": None, "keyword_overlap": None}
    if original_abstract:
        comparison = compare_abstracts(original_abstract, summary)

    # 3️⃣ Dashboard Metadata
    dashboard_data = {
        "query": query,
        "summary_length": len(summary),
        "citation_count": len(citations),
        "similarity": comparison["similarity"],
        "keyword_overlap": comparison["keyword_overlap"],
    }

    # 4️⃣ Output Display
    print("\n🧠 === RESEARCH SUMMARY ===\n")
    print(summary)
    print("\n📚 Citations:", citations)
    if original_abstract:
        print(f"\n📊 Abstract Similarity: {comparison['similarity']}% | Keyword Overlap: {comparison['keyword_overlap']}%")

    # 5️⃣ Dashboard Mode (return structured data)
    if dashboard_mode:
        return {"summary": summary, "metadata": dashboard_data, "citations": citations}

    return summary

# ==========================================================
#  Topic Clustering Utility (For Dashboard)
# ==========================================================
def analyze_topic_clusters(all_summaries: List[str]):
    topics = cluster_topics(all_summaries)
    print("\n🌐 === TOPIC CLUSTERS ===")
    for cluster, words in topics.items():
        print(f"{cluster}: {', '.join(words)}")
    visualize_clusters(topics)
    return topics
