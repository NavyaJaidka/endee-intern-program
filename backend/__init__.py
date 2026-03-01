"""
Backend module initialization.

This module provides the complete backend for the AI Research & Code Copilot,
including:
- Core configuration and logging
- Vector store (Endee client)
- Embedding generation
- Document ingestion
- RAG pipeline
- AI Agents
- FastAPI application
"""

__version__ = "1.0.0"

from backend.main import app
from backend.core import settings, get_settings
from backend.vectorstore import EndeeClient, get_endee_client
from backend.embedding import EmbeddingService, get_embedding_service
from backend.ingestion import DocumentProcessor
from backend.rag import RAGPipeline, get_rag_pipeline
from backend.agents import (
    RetrievalAgent,
    AnalysisAgent,
    RecommendationAgent,
    AgentPipeline,
    get_agent_pipeline
)

__all__ = [
    "app",
    "settings",
    "get_settings",
    "EndeeClient",
    "get_endee_client",
    "EmbeddingService",
    "get_embedding_service",
    "DocumentProcessor",
    "RAGPipeline",
    "get_rag_pipeline",
    "RetrievalAgent",
    "AnalysisAgent",
    "RecommendationAgent",
    "AgentPipeline",
    "get_agent_pipeline"
]
