"""
Embedding module initialization.

This module provides text embedding functionality using sentence-transformers.
"""

from backend.embedding.embedding_service import (
    EmbeddingService,
    get_embedding_service,
    embed_text,
    embed_texts,
    embed_query,
    get_sentence_transformer,
    get_embedding_dimension
)

__all__ = [
    "EmbeddingService",
    "get_embedding_service",
    "embed_text",
    "embed_texts",
    "embed_query",
    "get_sentence_transformer",
    "get_embedding_dimension"
]
