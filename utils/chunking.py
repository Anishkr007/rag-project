"""
Text Chunking Module for the RAG AI Assistant.

Custom implementation of recursive character text splitting that avoids
importing langchain_text_splitters (which triggers a broken import chain
through sentence_transformers → tensorflow → h5py → numpy crash).

HOW CHUNKING WORKS:
===================
Why chunk?  LLMs have token limits, and searching small focused chunks is more
accurate than searching entire pages.  Overlap ensures we don't lose context
at chunk boundaries.

Strategy:
  1. Try splitting on paragraph breaks ("\\n\\n")
  2. If chunks are still too big, split on single newlines ("\\n")
  3. Then on sentences (". ")
  4. Then on spaces (" ")
  5. Finally, character-by-character as a last resort
"""

from typing import List, Optional
from langchain_core.documents import Document


def _split_text_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str],
) -> List[str]:
    """
    Recursively split text using a hierarchy of separators.

    Tries the first separator; if any resulting chunk is still too large,
    falls back to the next separator for that chunk.
    """
    if not text:
        return []

    # Pick the best separator (first one found in the text)
    separator = separators[-1]  # fallback: split char-by-char
    for sep in separators:
        if sep == "":
            separator = sep
            break
        if sep in text:
            separator = sep
            break

    # Split the text
    if separator:
        splits = text.split(separator)
    else:
        splits = list(text)

    # Merge splits into chunks of the right size
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for piece in splits:
        piece_len = len(piece) + (len(separator) if current_chunk else 0)

        if current_length + piece_len > chunk_size and current_chunk:
            # Save current chunk
            merged = separator.join(current_chunk)
            chunks.append(merged)

            # Keep overlap by retaining trailing pieces
            overlap_length = 0
            overlap_pieces: List[str] = []
            for p in reversed(current_chunk):
                if overlap_length + len(p) > chunk_overlap:
                    break
                overlap_pieces.insert(0, p)
                overlap_length += len(p) + len(separator)

            current_chunk = overlap_pieces
            current_length = sum(len(p) for p in current_chunk) + len(separator) * max(0, len(current_chunk) - 1)

        current_chunk.append(piece)
        current_length += piece_len

    # Don't forget the last chunk
    if current_chunk:
        merged = separator.join(current_chunk)
        chunks.append(merged)

    # If any chunk is still too large and we have more separators, recurse
    remaining_separators = separators[separators.index(separator) + 1:] if separator in separators else []
    if remaining_separators:
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                final_chunks.extend(
                    _split_text_recursive(chunk, chunk_size, chunk_overlap, remaining_separators)
                )
            else:
                final_chunks.append(chunk)
        return final_chunks

    return chunks


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks, preserving metadata.

    Each resulting chunk carries:
      - All original metadata (source filename, page number, etc.)
      - An added 'chunk_index' field for ordering

    Args:
        documents: List of LangChain Document objects (e.g., from pdf_loader).
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between chunks.
        separators: Custom separator hierarchy.

    Returns:
        List of chunked Document objects.
    """
    if not documents:
        return []

    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    all_chunks: List[Document] = []
    chunk_index = 0

    for doc in documents:
        text_chunks = _split_text_recursive(
            doc.page_content, chunk_size, chunk_overlap, separators
        )

        for text_chunk in text_chunks:
            text_chunk = text_chunk.strip()
            if not text_chunk:
                continue

            new_doc = Document(
                page_content=text_chunk,
                metadata={**doc.metadata, "chunk_index": chunk_index},
            )
            all_chunks.append(new_doc)
            chunk_index += 1

    return all_chunks
