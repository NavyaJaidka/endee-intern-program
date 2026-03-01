"""
Pydantic models for the AI Research & Code Copilot API.

This module defines all request and response schemas for the API endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ==================== Enums ====================

class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    TXT = "txt"
    MD = "md"
    JSON = "json"
    GITHUB = "github"


class IndexSpaceType(str, Enum):
    """Vector space types for Endee."""
    COSINE = "cosine"
    L2 = "l2"
    IP = "ip"


class AgentType(str, Enum):
    """Agent types available in the system."""
    RETRIEVAL = "retrieval"
    ANALYSIS = "analysis"
    RECOMMENDATION = "recommendation"


# ==================== Document Models ====================

class DocumentUploadRequest(BaseModel):
    """Request to upload a document."""
    title: str = Field(..., description="Document title")
    content: Optional[str] = Field(None, description="Raw text content (for direct input)")
    source_type: DocumentType = Field(..., description="Source type")
    source_path: Optional[str] = Field(None, description="File path or GitHub URL")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class DocumentChunk(BaseModel):
    """Document chunk model."""
    chunk_id: str
    text: str
    source: str
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Document model."""
    doc_id: str
    title: str
    content: str
    source: str
    file_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunks: List[DocumentChunk] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    success: bool
    doc_id: str
    chunks_created: int
    message: str


# ==================== Index Models ====================

class IndexCreateRequest(BaseModel):
    """Request to create an index."""
    index_name: str = Field(..., description="Name of the index")
    dimension: int = Field(384, description="Vector dimension")
    space_type: IndexSpaceType = Field(IndexSpaceType.COSINE, description="Space type")
    m: int = Field(16, description="HNSW M parameter")
    ef_construct: int = Field(200, description="HNSW ef_construct parameter")
    precision: str = Field("int16", description="Quantization precision")


class IndexInfo(BaseModel):
    """Index information model."""
    name: str
    dimension: int
    total_elements: int
    space_type: str
    precision: str
    created_at: datetime


class IndexListResponse(BaseModel):
    """Response with list of indexes."""
    indexes: List[IndexInfo]


# ==================== Query Models ====================

class QueryRequest(BaseModel):
    """Request for RAG query."""
    query: str = Field(..., description="Query text")
    index_name: Optional[str] = Field(None, description="Index to search")
    top_k: int = Field(5, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    include_sources: bool = Field(True, description="Include sources in response")


class SourceDocument(BaseModel):
    """Source document in query response."""
    chunk_id: str
    text: str
    source: str
    score: float


class QueryResponse(BaseModel):
    """Response from RAG query."""
    success: bool
    answer: str
    query: str
    sources: List[SourceDocument]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ==================== Agent Models ====================

class AgentRequest(BaseModel):
    """Request for agent processing."""
    query: str = Field(..., description="Query text")
    agent_type: AgentType = Field(..., description="Type of agent to use")
    index_name: Optional[str] = Field(None, description="Index to search")
    top_k: int = Field(5, description="Number of documents")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")


class AgentResponse(BaseModel):
    """Response from agent."""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ==================== Search Models ====================

class SearchRequest(BaseModel):
    """Request for vector search."""
    query: str = Field(..., description="Search query")
    index_name: Optional[str] = Field(None, description="Index to search")
    top_k: int = Field(10, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters")


class SearchResult(BaseModel):
    """Search result model."""
    chunk_id: str
    text: str
    source: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response from search."""
    success: bool
    results: List[SearchResult]
    total: int
    query: str


# ==================== Health & Status ====================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    endee_connected: bool
    embedding_model_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.now)


class StatusResponse(BaseModel):
    """System status response."""
    version: str
    indexes: List[str]
    total_documents: int
    embedding_dimension: int
    llm_provider: str


# ==================== Error Models ====================

class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
