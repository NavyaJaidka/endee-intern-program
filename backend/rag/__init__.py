"""
RAG (Retrieval-Augmented Generation) module initialization.

This module implements the complete RAG pipeline including:
- Vector similarity search
- Context formatting
- LLM integration
"""

from backend.rag.rag_pipeline import (
    RAGPipeline,
    RAGResponse,
    RetrievedDocument,
    LLMWrapper,
    get_rag_pipeline,
    query_with_rag
)

__all__ = [
    "RAGPipeline",
    "RAGResponse",
    "RetrievedDocument",
    "LLMWrapper",
    "get_rag_pipeline",
    "query_with_rag"
]
