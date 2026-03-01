"""
AI Research & Code Copilot - Main Application.

This is the main FastAPI application that serves as the backend
for the AI Research & Code Copilot system.

The application provides:
- REST API for document ingestion
- Vector search via Endee
- RAG-powered question answering
- Multi-agent AI system

Usage:
    uvicorn backend.main:app --reload --port 8000

Or with custom settings:
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from backend.core.logging import setup_logging, get_logger
from backend.core.config import settings
from backend.api import router
from backend.vectorstore import get_endee_client

# Setup logging
setup_logging(
    log_level=settings.log_level,
    log_file=settings.log_file
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting AI Research & Code Copilot Backend")
    logger.info(f"Environment: {settings.llm_provider}")
    logger.info(f"Endee URL: {settings.endee_base_url}")
    logger.info(f"Embedding model: {settings.embedding_model}")
    
    # Test Endee connection
    try:
        endee_client = get_endee_client()
        if endee_client.health_check():
            logger.info("Endee vector database connected successfully")
            
            # Ensure default index exists
            try:
                endee_client.ensure_index_exists(
                    settings.endee_index_name,
                    settings.endee_dimension
                )
                logger.info(f"Index '{settings.endee_index_name}' ready")
            except Exception as e:
                logger.warning(f"Could not create default index: {e}")
        else:
            logger.warning("Endee vector database not responding")
    except Exception as e:
        logger.warning(f"Could not connect to Endee: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Research & Code Copilot Backend")


# Create FastAPI app
app = FastAPI(
    title="AI Research & Code Copilot API",
    description="""
    ## Overview
    
    This is the backend API for an AI-powered Research & Code Copilot system.
    It provides advanced document processing, vector search, and AI-powered
    question answering capabilities.
    
    ## Features
    
    - **Document Ingestion**: Upload PDFs, text files, markdown, JSON, or GitHub repos
    - **Vector Search**: Fast similarity search using Endee vector database
    - **RAG Pipeline**: Retrieval-Augmented Generation for question answering
    - **AI Agents**: Three specialized agents (Retrieval, Analysis, Recommendation)
    
    ## Endee Vector Database
    
    This system uses Endee as the core vector database for storing and searching
    embeddings. Endee is a high-performance open-source vector database.
    
    See: https://github.com/endee-io/endee
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API routes
app.include_router(router, prefix="/api/v1", tags=["API"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI Research & Code Copilot API",
        "version": "1.0.0",
        "description": "Advanced AI-powered research and code assistant",
        "docs": "/docs",
        "health": "/api/v1/health",
        "status": "/api/v1/status"
    }


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# Run the application
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
