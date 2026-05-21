"""
PDF Loading Module for the RAG AI Assistant.

Uses pypdf directly (instead of LangChain's PyPDFLoader) to avoid
triggering the heavy langchain_community import chain that pulls in
sentence_transformers, tensorflow, etc.

HOW PDF LOADING WORKS:
======================
1. Streamlit gives us a file-like object (BytesIO) when a user uploads a PDF
2. pypdf.PdfReader reads the file directly from memory (no temp file needed)
3. We extract text page-by-page and create LangChain Document objects
4. Each Document contains: page_content (the text) + metadata (source, page)
"""

from typing import List, BinaryIO
from pypdf import PdfReader
from langchain_core.documents import Document


def load_pdf_from_file(uploaded_file: BinaryIO, filename: str) -> List[Document]:
    """
    Extract text from a single uploaded PDF file.

    Args:
        uploaded_file: File-like object from Streamlit's file_uploader.
        filename: Original name of the PDF (used in metadata for citations).

    Returns:
        List of Document objects — one per page — with metadata.

    Raises:
        RuntimeError: If PDF extraction fails.
    """
    try:
        # Read PDF directly from the file-like object (no temp file needed)
        reader = PdfReader(uploaded_file)
        documents = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            # Skip empty pages
            if not text.strip():
                continue

            doc = Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_num,
                    "total_pages": len(reader.pages),
                },
            )
            documents.append(doc)

        return documents

    except Exception as e:
        raise RuntimeError(f"Error loading PDF '{filename}': {e}")


def load_multiple_pdfs(uploaded_files: list) -> List[Document]:
    """
    Extract text from multiple uploaded PDF files.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        Combined list of Documents from all PDFs, each with source metadata.
    """
    all_documents = []

    for uploaded_file in uploaded_files:
        # Reset file pointer so we read from the beginning
        uploaded_file.seek(0)
        docs = load_pdf_from_file(uploaded_file, uploaded_file.name)
        all_documents.extend(docs)

    return all_documents
