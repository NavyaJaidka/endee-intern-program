"""
API Routes for AI Research & Code Copilot.

This module defines all FastAPI endpoints for:
- Document ingestion (PDF, TXT, GitHub)
- Vector search
- RAG query
- AI Agents
- Index management
"""

import os
import json
from typing import List, Optional
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Body
from fastapi.responses import JSONResponse

from backend.core.logging import get_logger
from backend.core.config import settings
from backend.vectorstore import EndeeClient, get_endee_client
from backend.embedding import get_embedding_service
from backend.ingestion import DocumentProcessor
from backend.rag import get_rag_pipeline
from backend.agents import (
    get_agent_pipeline,
    RetrievalAgent,
    AnalysisAgent,
    RecommendationAgent
)
from backend.models import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    AgentRequest,
    AgentResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    HealthResponse,
    StatusResponse,
    ErrorResponse
)

logger = get_logger(__name__)

# Create router
router = APIRouter()


# ==================== Health & Status ====================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the health status of the system including:
    - Endee connection status
    - Embedding model status
    """
    # Check Endee connection
    endee_client = get_endee_client()
    endee_connected = endee_client.health_check()
    
    # Check embedding model (lazy check)
    embedding_loaded = False
    try:
        service = get_embedding_service()
        embedding_loaded = service is not None
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy" if endee_connected else "degraded",
        endee_connected=endee_connected,
        embedding_model_loaded=embedding_loaded
    )


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get system status.
    
    Returns information about:
    - Available indexes
    - Total documents
    - Configuration
    """
    endee_client = get_endee_client()
    
    # Get indexes
    try:
        indexes = endee_client.list_indexes()
        index_names = [idx.get("name", "") for idx in indexes]
    except Exception:
        index_names = []
    
    # Get embedding dimension
    embedding_dim = settings.endee_dimension
    
    return StatusResponse(
        version="1.0.0",
        indexes=index_names,
        total_documents=0,  # Would need to track this separately
        embedding_dimension=embedding_dim,
        llm_provider=settings.llm_provider
    )


# ==================== Index Management ====================

@router.post("/index/create")
async def create_index(
    index_name: str = Body(..., embed=True),
    dimension: int = Body(384),
    space_type: str = Body("cosine")
):
    """
    Create a new vector index in Endee.
    """
    try:
        endee_client = get_endee_client()
        
        # Check if index exists
        if endee_client.index_exists(index_name):
            raise HTTPException(status_code=409, detail="Index already exists")
        
        # Create index
        success = endee_client.create_index(
            index_name=index_name,
            dimension=dimension,
            space_type=space_type
        )
        
        if success:
            return {"success": True, "message": f"Index '{index_name}' created successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create index")
            
    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index/list")
async def list_indexes():
    """
    List all available indexes.
    """
    try:
        endee_client = get_endee_client()
        indexes = endee_client.list_indexes()
        return {"success": True, "indexes": indexes}
    except Exception as e:
        logger.error(f"Error listing indexes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/index/{index_name}")
async def delete_index(index_name: str):
    """
    Delete an index.
    """
    try:
        endee_client = get_endee_client()
        success = endee_client.delete_index(index_name)
        
        if success:
            return {"success": True, "message": f"Index '{index_name}' deleted"}
        else:
            raise HTTPException(status_code=404, detail="Index not found")
    except Exception as e:
        logger.error(f"Error deleting index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Document Ingestion ====================

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    index_name: Optional[str] = Form(None)
):
    """
    Upload and process a document (PDF, TXT, MD, JSON).
    
    The document will be:
    1. Processed to extract text
    2. Chunked into smaller pieces
    3. Embedded using sentence-transformers
    4. Stored in Endee vector database
    """
    index_name = index_name or settings.endee_index_name
    
    try:
        # Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        temp_file = temp_dir / file.filename
        
        # Write file
        content = await file.read()
        temp_file.write_bytes(content)
        
        # Process document
        processor = DocumentProcessor()
        
        if file.filename.endswith('.pdf'):
            doc = processor.process_pdf(str(temp_file))
        elif file.filename.endswith(('.txt', '.md')):
            doc = processor.process_txt(str(temp_file))
        elif file.filename.endswith('.json'):
            doc = processor.process_json(str(temp_file))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Generate embeddings and store in Endee
        embedding_service = get_embedding_service()
        endee_client = get_endee_client()
        
        # Ensure index exists
        endee_client.ensure_index_exists(index_name)
        
        # Process chunks
        vectors_to_insert = []
        
        for chunk in doc.chunks:
            # Generate embedding
            embedding = embedding_service.embed_text(chunk.text)
            
            vectors_to_insert.append({
                "id": chunk.chunk_id,
                "vector": embedding,
                "metadata": {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "source": doc.source,
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                    "text": chunk.text[:500]  # Store truncated text
                }
            })
        
        # Insert in batches
        batch_size = 100
        for i in range(0, len(vectors_to_insert), batch_size):
            batch = vectors_to_insert[i:i + batch_size]
            endee_client.insert_vectors(index_name, batch)
        
        # Cleanup
        temp_file.unlink()
        
        logger.info(f"Uploaded document '{doc.title}' with {len(doc.chunks)} chunks")
        
        return DocumentUploadResponse(
            success=True,
            doc_id=doc.doc_id,
            chunks_created=len(doc.chunks),
            message=f"Document uploaded successfully with {len(doc.chunks)} chunks"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/text", response_model=DocumentUploadResponse)
async def upload_text_document(
    request: DocumentUploadRequest,
    index_name: Optional[str] = None
):
    """
    Upload raw text content.
    """
    index_name = index_name or settings.endee_index_name
    
    try:
        if not request.content:
            raise HTTPException(status_code=400, detail="Content is required")
        
        # Process content
        processor = DocumentProcessor()
        doc = processor.process_content(
            content=request.content,
            title=request.title,
            source=request.source_path or "direct_input"
        )
        
        # Generate embeddings and store
        embedding_service = get_embedding_service()
        endee_client = get_endee_client()
        
        # Ensure index exists
        endee_client.ensure_index_exists(index_name)
        
        # Process chunks
        vectors_to_insert = []
        
        for chunk in doc.chunks:
            embedding = embedding_service.embed_text(chunk.text)
            
            vectors_to_insert.append({
                "id": chunk.chunk_id,
                "vector": embedding,
                "metadata": {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "source": doc.source,
                    "text": chunk.text[:500]
                }
            })
        
        # Insert
        endee_client.insert_vectors(index_name, vectors_to_insert)
        
        return DocumentUploadResponse(
            success=True,
            doc_id=doc.doc_id,
            chunks_created=len(doc.chunks),
            message=f"Text uploaded successfully with {len(doc.chunks)} chunks"
        )
        
    except Exception as e:
        logger.error(f"Error uploading text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/github", response_model=DocumentUploadResponse)
async def upload_github_repo(
    repo_url: str = Body(..., embed=True),
    branch: str = Body("main"),
    index_name: Optional[str] = None
):
    """
    Upload files from a GitHub repository.
    """
    index_name = index_name or settings.endee_index_name
    
    try:
        processor = DocumentProcessor()
        
        # Process repository
        docs = processor.process_github_repo(repo_url, branch)
        
        if not docs:
            raise HTTPException(status_code=404, detail="No files found in repository")
        
        # Generate embeddings and store
        embedding_service = get_embedding_service()
        endee_client = get_endee_client()
        
        # Ensure index exists
        endee_client.ensure_index_exists(index_name)
        
        total_chunks = 0
        
        for doc in docs:
            vectors_to_insert = []
            
            for chunk in doc.chunks:
                embedding = embedding_service.embed_text(chunk.text)
                
                vectors_to_insert.append({
                    "id": chunk.chunk_id,
                    "vector": embedding,
                    "metadata": {
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "source": doc.source,
                        "github_repo": doc.metadata.get("github_repo"),
                        "text": chunk.text[:500]
                    }
                })
            
            if vectors_to_insert:
                endee_client.insert_vectors(index_name, vectors_to_insert)
                total_chunks += len(vectors_to_insert)
        
        return DocumentUploadResponse(
            success=True,
            doc_id=f"github_{datetime.now().timestamp()}",
            chunks_created=total_chunks,
            message=f"Uploaded {len(docs)} files with {total_chunks} chunks"
        )
        
    except Exception as e:
        logger.error(f"Error uploading GitHub repo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Search ====================

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search for similar documents using vector similarity.
    """
    try:
        endee_client = get_endee_client()
        embedding_service = get_embedding_service()
        
        index_name = request.index_name or settings.endee_index_name
        
        # Generate query embedding
        query_embedding = embedding_service.embed_query(request.query)
        
        # Search
        results = endee_client.search(
            index_name=index_name,
            query_vector=query_embedding,
            k=request.top_k,
            filter_expr=request.filters
        )
        
        # Format results
        search_results = []
        
        for item in results:
            search_results.append(SearchResult(
                chunk_id=item.get("id", ""),
                text=item.get("text", item.get("content", "")),
                source=item.get("source", "unknown"),
                score=item.get("score", 0),
                metadata=item.get("metadata", {})
            ))
        
        return SearchResponse(
            success=True,
            results=search_results,
            total=len(search_results),
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== RAG Query ====================

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using RAG (Retrieval-Augmented Generation).
    
    This endpoint:
    1. Embeds the query
    2. Searches for similar documents in Endee
    3. Passes context to LLM
    4. Returns generated answer with sources
    """
    try:
        rag_pipeline = get_rag_pipeline()
        
        index_name = request.index_name or settings.endee_index_name
        
        # Query with RAG
        result = rag_pipeline.query_with_sources(
            query=request.query,
            index_name=index_name,
            top_k=request.top_k
        )
        
        # Format sources
        sources = [
            SourceDocument(
                chunk_id=src["chunk_id"],
                text=src["text"],
                source=src["source"],
                score=src["score"]
            )
            for src in result["sources"]
        ]
        
        return QueryResponse(
            success=True,
            answer=result["answer"],
            query=result["query"],
            sources=sources,
            metadata=result["metadata"]
        )
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== AI Agents ====================

@router.post("/agents/retrieval", response_model=AgentResponse)
async def retrieval_agent(request: AgentRequest):
    """
    Use the Retrieval Agent to find relevant documents.
    """
    try:
        agent = RetrievalAgent()
        
        index_name = request.index_name or settings.endee_index_name
        
        result = agent.search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            index_name=index_name
        )
        
        return AgentResponse(
            success=result.success,
            result=result.result,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"Retrieval agent error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/analysis", response_model=AgentResponse)
async def analysis_agent(
    query: str = Body(..., embed=True),
    top_k: int = Body(5)
):
    """
    Use the Analysis Agent to analyze documents and extract insights.
    """
    try:
        agent = AnalysisAgent()
        
        result = agent.analyze_with_rag(query=query, top_k=top_k)
        
        return AgentResponse(
            success=result.success,
            result=result.result,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"Analysis agent error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/recommendation", response_model=AgentResponse)
async def recommendation_agent(
    query: str = Body(..., embed=True)
):
    """
    Use the Recommendation Agent to generate actionable recommendations.
    """
    try:
        agent = RecommendationAgent()
        
        result = agent.recommend(query=query)
        
        return AgentResponse(
            success=result.success,
            result=result.result,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"Recommendation agent error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/pipeline")
async def full_agent_pipeline(
    query: str = Body(..., embed=True),
    top_k: int = Body(5)
):
    """
    Run the full agent pipeline (Retrieval -> Analysis -> Recommendation).
    """
    try:
        pipeline = get_agent_pipeline()
        
        result = pipeline.process(
            query=query,
            use_analysis=True,
            use_recommendations=True,
            top_k=top_k
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Agent pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Embeddings ====================

@router.post("/embeddings")
async def generate_embeddings(
    texts: List[str] = Body(...)
):
    """
    Generate embeddings for a list of texts.
    """
    try:
        service = get_embedding_service()
        
        embeddings = service.embed_texts(texts)
        
        return {
            "success": True,
            "embeddings": embeddings,
            "dimension": len(embeddings[0]) if embeddings else 0
        }
        
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
