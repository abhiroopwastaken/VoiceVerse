"""
Document Loader Module
======================
Extracts text from PDF, DOCX, and TXT files.
"""

import os
import tempfile
from typing import Tuple


def extract_text(file_path: str) -> Tuple[str, dict]:
    """
    Extract text from a document file.
    
    Returns:
        Tuple of (extracted_text, metadata_dict)
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext == ".docx":
        return _extract_docx(file_path)
    elif ext == ".txt":
        return _extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Please upload PDF, DOCX, or TXT.")


def _extract_pdf(file_path: str) -> Tuple[str, dict]:
    """Extract text from PDF using PyMuPDF."""
    import fitz  # PyMuPDF
    
    doc = fitz.open(file_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    
    text = "\n\n".join(pages)
    text = _clean_text(text)
    
    metadata = {
        "filename": os.path.basename(file_path),
        "type": "PDF",
        "pages": len(pages),
        "word_count": len(text.split()),
    }
    return text, metadata


def _extract_docx(file_path: str) -> Tuple[str, dict]:
    """Extract text from DOCX using python-docx."""
    from docx import Document
    
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    text = _clean_text(text)
    
    metadata = {
        "filename": os.path.basename(file_path),
        "type": "DOCX",
        "pages": max(1, len(paragraphs) // 20),  # estimate
        "word_count": len(text.split()),
    }
    return text, metadata


def _extract_txt(file_path: str) -> Tuple[str, dict]:
    """Extract text from TXT with encoding detection."""
    import chardet
    
    with open(file_path, "rb") as f:
        raw = f.read()
    
    detected = chardet.detect(raw)
    encoding = detected.get("encoding", "utf-8") or "utf-8"
    
    try:
        text = raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        text = raw.decode("utf-8", errors="ignore")
    
    text = _clean_text(text)
    
    metadata = {
        "filename": os.path.basename(file_path),
        "type": "TXT",
        "pages": 1,
        "word_count": len(text.split()),
    }
    return text, metadata


def _clean_text(text: str) -> str:
    """Clean extracted text - remove excessive whitespace, fix encoding artifacts."""
    import re
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    # Remove null bytes
    text = text.replace('\x00', '')
    return text.strip()


def extract_from_multiple(file_paths: list) -> Tuple[str, list]:
    """
    Extract text from multiple files and combine.
    
    Returns:
        Tuple of (combined_text, list_of_metadata)
    """
    all_text = []
    all_meta = []
    
    for fp in file_paths:
        try:
            text, meta = extract_text(fp)
            if text:
                all_text.append(f"--- Document: {meta['filename']} ---\n{text}")
                all_meta.append(meta)
        except Exception as e:
            all_meta.append({
                "filename": os.path.basename(fp),
                "type": "ERROR",
                "error": str(e),
            })
    
    combined = "\n\n".join(all_text)
    return combined, all_meta
