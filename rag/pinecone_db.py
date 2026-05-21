"""
Pinecone Vector Database Module for the RAG AI Assistant.

Handles initialization of the Pinecone client, index creation, and
providing a LangChain-compatible PineconeVectorStore.

HOW PINECONE STORES VECTORS:
=============================
1. Each text chunk is converted to a numerical vector (embedding) by OpenAI
2. Pinecone stores these vectors in a high-dimensional index
3. When you search, your query is also converted to a vector
4. Pinecone uses cosine similarity to find the closest matching vectors
5. It returns the most similar chunks along with their metadata

Think of it like a library:
- Each book page is a vector (point in space)
- Similar pages are located close together
- Searching = finding the nearest neighbors to your query point
"""

import os
from typing import Optional

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from utils.helpers import get_env_var


# =========================================================================
# Pinecone Client Initialization
# =========================================================================

def get_pinecone_client() -> Pinecone:
    """
    Create and return a Pinecone client instance.

    Reads PINECONE_API_KEY from environment variables.

    Returns:
        Authenticated Pinecone client.
    """
    api_key = get_env_var("PINECONE_API_KEY")
    return Pinecone(api_key=api_key)


def get_or_create_index(
    client: Optional[Pinecone] = None,
    index_name: Optional[str] = None,
    dimension: int = 1024,  # user's Pinecone index is set to 1024
    metric: str = "cosine",
) -> object:
    """
    Get an existing Pinecone index or create a new one.

    Args:
        client: Pinecone client (created if None).
        index_name: Name of the index (defaults to env var PINECONE_INDEX_NAME).
        dimension: Vector dimension (1024 to match user's existing index).
        metric: Distance metric — "cosine", "euclidean", or "dotproduct".

    Returns:
        A Pinecone Index object ready for operations.
    """
    if client is None:
        client = get_pinecone_client()

    if index_name is None:
        index_name = get_env_var("PINECONE_INDEX_NAME", "rag-assistant")

    # Check if the index already exists
    existing_indexes = [idx.name for idx in client.list_indexes()]

    if index_name not in existing_indexes:
        # Create a new serverless index
        cloud = get_env_var("PINECONE_CLOUD", "aws")
        region = get_env_var("PINECONE_REGION", "us-east-1")

        client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"[OK] Created Pinecone index: {index_name}")
    else:
        print(f"[OK] Using existing Pinecone index: {index_name}")

    return client.Index(index_name)


# =========================================================================
# LangChain Vector Store
# =========================================================================

def get_embeddings_model() -> OpenAIEmbeddings:
    """
    Create an OpenAI embeddings model instance.

    Uses text-embedding-3-small truncated to 1024 dimensions to match
    the existing Pinecone index dimension.
    
    Returns:
        Configured OpenAIEmbeddings instance.
    """
    api_key = get_env_var("OPENAI_API_KEY")
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1024,
        openai_api_key=api_key,
    )


def get_vector_store(
    index_name: Optional[str] = None,
    embeddings: Optional[OpenAIEmbeddings] = None,
    namespace: str = "",
) -> PineconeVectorStore:
    """
    Create a LangChain PineconeVectorStore for document storage and retrieval.

    This is the bridge between LangChain and Pinecone — it lets you use
    Pinecone as a retriever in LangChain chains.

    Args:
        index_name: Name of the Pinecone index.
        embeddings: Embeddings model (created if None).
        namespace: Pinecone namespace for multi-tenant isolation.

    Returns:
        PineconeVectorStore instance.
    """
    if index_name is None:
        index_name = get_env_var("PINECONE_INDEX_NAME", "rag-assistant")
        get_or_create_index(index_name=index_name) # Ensure it exists first

    if embeddings is None:
        embeddings = get_embeddings_model()

    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        text_key="text",
        namespace=namespace,
    )
