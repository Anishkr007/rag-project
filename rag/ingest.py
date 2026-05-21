"""
Document Ingestion Pipeline for the RAG AI Assistant.

Orchestrates the full pipeline: PDF → Text → Chunks → Embeddings → Pinecone.

HOW THE INGESTION PIPELINE WORKS:
==================================
1. LOAD:   PDFs are parsed page-by-page into Document objects
2. CHUNK:  Large pages are split into smaller overlapping pieces
3. EMBED:  Each chunk is converted into a 1536-dim vector by OpenAI
4. STORE:  Vectors + metadata are upserted into Pinecone

This only needs to run once per set of PDFs. After ingestion, the vectors
live in Pinecone and can be searched any number of times.
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from utils.pdf_loader import load_multiple_pdfs
from utils.chunking import split_documents
from utils.helpers import get_env_var
from rag.pinecone_db import get_vector_store, get_or_create_index, get_embeddings_model


def ingest_pdfs(
    uploaded_files: list,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    namespace: str = "",
) -> dict:
    """
    Run the full ingestion pipeline on uploaded PDF files.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between chunks.
        namespace: Pinecone namespace for isolation.

    Returns:
        Dict with stats: num_files, num_pages, num_chunks, status.
    """
    stats = {
        "num_files": len(uploaded_files),
        "num_pages": 0,
        "num_chunks": 0,
        "status": "processing",
        "error": None,
    }

    try:
        # ----- Step 1: Load PDFs -----
        documents = load_multiple_pdfs(uploaded_files)
        stats["num_pages"] = len(documents)

        if not documents:
            stats["status"] = "error"
            stats["error"] = "No text could be extracted from the uploaded PDFs."
            return stats

        # ----- Step 2: Split into chunks -----
        chunks = split_documents(documents, chunk_size, chunk_overlap)
        stats["num_chunks"] = len(chunks)

        if not chunks:
            stats["status"] = "error"
            stats["error"] = "Text splitting produced no chunks."
            return stats

        # ----- Step 3 & 4: Embed + Store in Pinecone -----
        # PineconeVectorStore.from_documents handles embedding + upserting
        # It requires the index_name, not the index object
        index_name = get_env_var("PINECONE_INDEX_NAME", "rag-assistant")
        get_or_create_index(index_name=index_name) # Ensure it exists
        embeddings = get_embeddings_model()

        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=index_name,
            text_key="text",
            namespace=namespace,
        )

        stats["status"] = "success"
        return stats

    except Exception as e:
        stats["status"] = "error"
        stats["error"] = str(e)
        return stats


def ingest_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    namespace: str = "",
) -> dict:
    """
    Ingest pre-loaded Document objects (for advanced use cases).

    Args:
        documents: Pre-loaded LangChain Documents.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between chunks.
        namespace: Pinecone namespace.

    Returns:
        Dict with ingestion stats.
    """
    stats = {"num_docs": len(documents), "num_chunks": 0, "status": "processing"}

    try:
        chunks = split_documents(documents, chunk_size, chunk_overlap)
        stats["num_chunks"] = len(chunks)

        index_name = get_env_var("PINECONE_INDEX_NAME", "rag-assistant")
        get_or_create_index(index_name=index_name) # Ensure it exists
        embeddings = get_embeddings_model()

        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=index_name,
            text_key="text",
            namespace=namespace,
        )

        stats["status"] = "success"
        return stats

    except Exception as e:
        stats["status"] = "error"
        stats["error"] = str(e)
        return stats
