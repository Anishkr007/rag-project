"""
Helper Utility Functions for the RAG AI Assistant.

Provides:
- Environment variable loading and validation
- Source document formatting for UI display
- Text utilities (truncation, token estimation)
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv


# =========================================================================
# Environment Management
# =========================================================================

def load_environment() -> None:
    """
    Load environment variables from .env file.
    Call this once at application startup before accessing any env vars.
    """
    load_dotenv()


def validate_environment() -> Dict[str, bool]:
    """
    Check that all required environment variables are set.

    Returns:
        Dict with variable names → True/False, plus "all_required_set" key.
    """
    required = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    optional = ["PINECONE_INDEX_NAME", "PINECONE_CLOUD", "PINECONE_REGION"]

    status = {}
    for var in required:
        status[var] = bool(os.getenv(var))
    for var in optional:
        status[var] = bool(os.getenv(var))

    status["all_required_set"] = all(status[v] for v in required)
    return status


def get_env_var(name: str, default: Optional[str] = None) -> str:
    """
    Retrieve an environment variable, raising an error if missing.

    Args:
        name: Variable name.
        default: Fallback value (if None and var missing, raises ValueError).
    """
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(
            f"Environment variable '{name}' is not set. "
            f"Please add it to your .env file."
        )
    return value


# =========================================================================
# Source Document Formatting
# =========================================================================

def format_source_documents(docs: list) -> List[Dict[str, str]]:
    """
    Format retrieved LangChain Documents into a clean list for UI display.

    Deduplicates by (source, page) and provides a content preview.

    Args:
        docs: List of LangChain Document objects.

    Returns:
        List of dicts with keys: source, page, content.
    """
    sources = []
    seen = set()

    for doc in docs:
        meta = doc.metadata
        source = meta.get("source", "Unknown")
        page = meta.get("page", "N/A")
        key = f"{source}::p{page}"

        if key in seen:
            continue
        seen.add(key)

        preview = doc.page_content[:250].strip()
        if len(doc.page_content) > 250:
            preview += "..."

        sources.append({
            "source": source,
            "page": str(page),
            "content": preview,
        })

    return sources


# =========================================================================
# Text Utilities
# =========================================================================

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis."""
    return text if len(text) <= max_length else text[: max_length - 3] + "..."


def estimate_token_count(text: str) -> int:
    """Rough token estimate (1 token ≈ 4 chars). Use tiktoken for precision."""
    return len(text) // 4
