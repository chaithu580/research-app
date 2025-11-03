"""
chunker.py
-----------
Splits extracted research paper sections into overlapping text chunks.
Each chunk includes metadata (section name, page number, chunk ID).
Optimized for use with embedding models like sentence-transformers.
"""

import re
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

def normalize_whitespace(text: str) -> str:
    """Cleans and normalizes whitespace and line breaks."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def create_chunks(sections: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Splits each section into chunks with overlap.
    Uses LangChain's RecursiveCharacterTextSplitter for robustness.

    Args:
        sections: List of dicts from extractor.py (section, content, page)
        chunk_size: max character length per chunk (default 1000)
        overlap: number of characters to overlap between consecutive chunks (default 200)

    Returns:
        List of chunks with metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []
    for sec in sections:
        text = normalize_whitespace(sec["content"])
        section_chunks = splitter.split_text(text)

        for idx, chunk in enumerate(section_chunks):
            chunks.append({
                "chunk_id": f"{sec['section']}_{sec['page']}_{idx}",
                "text": chunk,
                "section": sec["section"],
                "page": sec["page"]
            })

    print(f"[INFO] Created {len(chunks)} chunks from {len(sections)} sections.")
    return chunks


def preview_chunks(chunks: List[Dict], n: int = 3):
    """Prints the first n chunks for inspection."""
    for i, ch in enumerate(chunks[:n]):
        print(f"\n🧩 Chunk {i+1}:")
        print(f"Section: {ch['section']} | Page: {ch['page']}")
        print(ch['text'][:400], "...")

