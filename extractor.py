"""
extractor.py
-------------
Extracts text and metadata from research papers (PDFs).
Uses PyMuPDF (fitz) for accurate layout-based text extraction.
Outputs structured text with section-level metadata.
"""

import fitz  # PyMuPDF
import re
import os

def extract_text_from_pdf(pdf_path):
    """
    Extracts raw text (page-wise) from a PDF file.
    Returns a list of pages: [{"page_number": int, "text": str}]
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    pages = []

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text("text")
        pages.append({
            "page_number": page_number + 1,
            "text": text.strip()
        })
    
    doc.close()
    return pages


def clean_text(text):
    """
    Cleans up extracted text by:
      - removing multiple spaces
      - fixing newlines
      - stripping reference numbers or figure captions (optional)
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]+\]', '', text)  # remove inline citations like [12]
    text = text.replace('–', '-')
    return text.strip()


def segment_sections(pages):
    """
    Segments text into sections based on headings.
    Uses simple regex for scientific paper section headers like:
      'ABSTRACT', 'INTRODUCTION', 'METHODS', etc.
    Returns a list of dicts: [{"section": str, "content": str, "page": int}]
    """
    section_pattern = re.compile(
        r'(?i)\b(abstract|introduction|background|related work|'
        r'methodology|methods|experiments|results|discussion|conclusion|references)\b'
    )

    sections = []
    current_section = {"section": "Unknown", "content": "", "page": 1}

    for page in pages:
        text = clean_text(page["text"])
        # Split by common section headers
        parts = section_pattern.split(text)
        for i in range(1, len(parts), 2):
            section_name = parts[i].strip().title()
            section_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if section_text:
                sections.append({
                    "section": section_name,
                    "content": section_text,
                    "page": page["page_number"]
                })
    if not sections:
        # fallback: whole paper as one section
        joined_text = " ".join([clean_text(p["text"]) for p in pages])
        sections = [{"section": "Full_Paper", "content": joined_text, "page": 1}]
    
    return sections


def extract_pdf_sections(pdf_path):
    """
    Full pipeline: extract → clean → segment.
    Returns structured text for downstream embedding & chunking.
    """
    pages = extract_text_from_pdf(pdf_path)
    sections = segment_sections(pages)
    print(f"[INFO] Extracted {len(sections)} sections from {pdf_path}")
    return sections

