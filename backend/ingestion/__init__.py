"""
Ingestion module initialization.

This module handles document processing for various file formats including
PDF, text files, markdown, JSON, and GitHub repositories.
"""

from backend.ingestion.document_processor import (
    DocumentProcessor,
    Document,
    DocumentChunk,
    process_file,
    process_pdf,
    process_text,
    chunk_text
)

__all__ = [
    "DocumentProcessor",
    "Document",
    "DocumentChunk",
    "process_file",
    "process_pdf",
    "process_text",
    "chunk_text"
]
