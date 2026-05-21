"""
Document Retriever Module for the RAG AI Assistant.

Creates a retriever that searches Pinecone for chunks most relevant to a query.

HOW RETRIEVAL WORKS:
====================
1. The user's question is converted into an embedding vector
2. Pinecone performs Approximate Nearest Neighbor (ANN) search
3. It returns the top-k most similar document chunks
4. These chunks become the "context" the LLM uses to answer

Retrieval Quality Tips:
- top_k=5 is a good default (balances context quality vs. token usage)
- Cosine similarity works best for semantic search
- The chunk size during ingestion affects retrieval quality
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag.pinecone_db import get_vector_store, get_or_create_index, get_embeddings_model


def create_retriever(
    top_k: int = 5,
    namespace: str = "",
    search_type: str = "similarity",
) -> BaseRetriever:
    """
    Create a LangChain retriever backed by Pinecone.

    Args:
        top_k: Number of top results to return (default 5).
        namespace: Pinecone namespace to search in.
        search_type: "similarity" or "mmr" (Maximal Marginal Relevance).
            - "similarity": Returns the most similar documents.
            - "mmr": Returns diverse yet relevant documents (reduces redundancy).

    Returns:
        A LangChain retriever object you can call with .invoke(query).
    """
    vector_store = get_vector_store(namespace=namespace)

    # Create retriever with search parameters
    search_kwargs = {"k": top_k}

    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    return retriever


def similarity_search(
    query: str,
    top_k: int = 5,
    namespace: str = "",
) -> List[Document]:
    """
    Perform a direct similarity search on Pinecone.

    This is a lower-level alternative to using the retriever.
    Useful when you need more control over the search process.

    Args:
        query: The search query text.
        top_k: Number of results to return.
        namespace: Pinecone namespace.

    Returns:
        List of Document objects sorted by similarity (most similar first).
    """
    vector_store = get_vector_store(namespace=namespace)
    results = vector_store.similarity_search(query, k=top_k)
    return results


def similarity_search_with_scores(
    query: str,
    top_k: int = 5,
    namespace: str = "",
) -> List[tuple]:
    """
    Perform similarity search and return scores alongside documents.

    Scores range from 0 to 1 (for cosine similarity):
    - 1.0 = perfect match
    - 0.0 = completely unrelated

    Args:
        query: The search query text.
        top_k: Number of results to return.
        namespace: Pinecone namespace.

    Returns:
        List of (Document, score) tuples.
    """
    vector_store = get_vector_store(namespace=namespace)
    results = vector_store.similarity_search_with_score(query, k=top_k)
    return results
