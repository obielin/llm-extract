"""
Text chunker for long documents.

Splits text into overlapping chunks that fit within LLM context windows,
preserving sentence and paragraph boundaries where possible.
"""

from __future__ import annotations

import re

from llm_extract.models import DocumentChunk


def chunk_text(
    text: str,
    chunk_size: int = 8000,
    overlap: int = 500,
) -> list[DocumentChunk]:
    """
    Split text into overlapping chunks.

    Prefers splitting at paragraph boundaries, then sentence boundaries,
    then word boundaries. Never splits mid-word.

    Args:
        text:       Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap:    Characters of overlap between adjacent chunks

    Returns:
        List of DocumentChunk objects
    """
    text = text.strip()
    if not text:
        return []

    # If text fits in one chunk, return it directly
    if len(text) <= chunk_size:
        return [
            DocumentChunk(
                text=text,
                chunk_index=0,
                total_chunks=1,
                char_start=0,
                char_end=len(text),
            )
        ]

    chunks = _split_into_chunks(text, chunk_size, overlap)
    total = len(chunks)

    return [
        DocumentChunk(
            text=chunk_text,
            chunk_index=i,
            total_chunks=total,
            char_start=char_start,
            char_end=char_start + len(chunk_text),
        )
        for i, (chunk_text, char_start) in enumerate(chunks)
    ]


def _split_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, int]]:
    """
    Returns list of (chunk_text, char_start_position) tuples.
    """
    chunks: list[tuple[str, int]] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Try to find a good split point: paragraph > sentence > word
            split_point = _find_split_point(text, start, end)
        else:
            split_point = end

        chunk = text[start:split_point].strip()
        if chunk:
            chunks.append((chunk, start))

        # Move forward, backing up by overlap amount
        start = split_point - overlap
        if start <= chunks[-1][1] if chunks else start <= 0:
            # Avoid infinite loop — force progress
            start = split_point

    return chunks


def _find_split_point(text: str, start: int, end: int) -> int:
    """
    Find the best split point near `end`, searching backwards from it.
    Priority: paragraph break > sentence end > word boundary.
    """
    window = text[start:end]

    # Try paragraph break (double newline)
    para_match = list(re.finditer(r"\n\n", window))
    if para_match:
        # Use the last paragraph break in the window
        last_match = para_match[-1]
        return start + last_match.end()

    # Try sentence end (. ! ? followed by space or newline)
    sentence_match = list(re.finditer(r"[.!?]\s", window))
    if sentence_match:
        last_match = sentence_match[-1]
        return start + last_match.end()

    # Try word boundary (space)
    space_match = list(re.finditer(r"\s", window))
    if space_match:
        last_match = space_match[-1]
        return start + last_match.start()

    # No good split point — cut at end (might split mid-word, but unavoidable)
    return end
