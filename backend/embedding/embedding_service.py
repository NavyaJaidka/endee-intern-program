"""
Embedding Service for AI Research & Code Copilot.

This module provides text embedding functionality using sentence-transformers.
It supports batch processing, multiple models, and device selection (CPU/CUDA).

The embeddings are used for:
1. Document chunk indexing in Endee
2. Query embedding for similarity search
3. RAG pipeline context retrieval
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path

from backend.core.logging import get_logger
from backend.core.config import settings

logger = get_logger(__name__)

# Lazy import to avoid loading model at import time
_sentence_transformer = None
_openai_embeddings = None
_model_name = None
_provider = None
_device = None


def get_openai_embeddings(model_name: Optional[str] = None):
    """Get or initialize OpenAI embeddings."""
    global _openai_embeddings, _model_name, _provider
    
    model_name = model_name or settings.embedding_model
    
    if _openai_embeddings is None or _model_name != model_name or _provider != "openai":
        logger.info(f"Initializing OpenAI embeddings model: {model_name}")
        
        from langchain_openai import OpenAIEmbeddings
        
        _openai_embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=model_name
        )
        _model_name = model_name
        _provider = "openai"
        
        logger.info(f"OpenAI embeddings ready")
        
    return _openai_embeddings


def get_sentence_transformer(
    model_name: Optional[str] = None,
    device: Optional[str] = None
):
    """
    Get or initialize the sentence-transformer model.
    """
    global _sentence_transformer, _model_name, _device, _provider
    
    model_name = model_name or settings.embedding_model
    device = device or settings.embedding_device
    
    if _sentence_transformer is None or _model_name != model_name or _device != device or _provider != "local":
        logger.info(f"Loading sentence-transformer model: {model_name} on {device}")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            _sentence_transformer = SentenceTransformer(
                model_name,
                device=device
            )
            _model_name = model_name
            _device = device
            _provider = "local"
            
            logger.info(f"Model loaded successfully")
        except ImportError:
            logger.error("sentence-transformers not installed. Please install 'sentence-transformers' and 'torch' to use local embeddings.")
            raise ImportError(
                "Local embedding libraries (sentence-transformers/torch) are not installed. "
                "Please install them or switch settings.embedding_provider to 'openai'."
            )
    
    return _sentence_transformer


def get_embedding_dimension() -> int:
    """
    Get the embedding dimension for the configured model.
    
    Returns:
        Embedding dimension
    """
    model = get_sentence_transformer()
    return model.get_sentence_embedding_dimension()


class EmbeddingService:
    """
    Service for generating text embeddings.
    
    This class provides a high-level interface for generating embeddings
    from text, supporting batch processing, caching, and multiple models.
    
    Attributes:
        model_name: Name of the sentence-transformer model
        device: Device for embedding (cpu or cuda)
        batch_size: Batch size for processing
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        """
        Initialize the embedding service.
        
        Args:
            provider: Embedding provider (local or openai)
            model_name: Model name
            device: Device (cpu or cuda)
            batch_size: Batch size for processing
        """
        self.provider = provider or settings.embedding_provider
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.batch_size = batch_size or settings.embedding_batch_size
        
        # Will be loaded lazily
        self._model = None
        
        logger.info(
            f"EmbeddingService initialized with provider={self.provider}, "
            f"model={self.model_name}, device={self.device}"
        )
    
    @property
    def model(self):
        """Get the model (lazy loading)."""
        if self._model is None:
            if self.provider == "openai":
                self._model = get_openai_embeddings(self.model_name)
            else:
                self._model = get_sentence_transformer(self.model_name, self.device)
        return self._model
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        """
        if self.provider == "openai":
            return self.model.embed_query(text)
        else:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        """
        if not texts:
            return []
            
        if self.provider == "openai":
            return self.model.embed_documents(texts)
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=True
            )
            return [emb.tolist() for emb in embeddings]
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        This is an alias for embed_text with optional query-specific processing.
        
        Args:
            query: Search query
        
        Returns:
            Query embedding vector
        """
        return self.embed_text(query)
    
    def embed_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Embed documents with their metadata.
        
        Args:
            documents: List of document dictionaries
            text_field: Field name containing text content
        
        Returns:
            Documents with added embedding field
        """
        if not documents:
            return []
        
        # Extract texts
        texts = [doc.get(text_field, "") for doc in documents]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Add embeddings to documents
        result = []
        for doc, embedding in zip(documents, embeddings):
            doc_copy = doc.copy()
            doc_copy["embedding"] = embedding
            result.append(doc_copy)
        
        return result
    
    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Cosine similarity score
        """
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        return dot_product / (norm1 * norm2)
    
    def batch_embed_with_metadata(
        self,
        chunks: List[Dict[str, Any]],
        text_field: str = "text",
        id_field: str = "chunk_id"
    ) -> List[Dict[str, Any]]:
        """
        Embed text chunks with metadata for vector storage.
        
        Args:
            chunks: List of text chunks with metadata
            text_field: Field containing text
            id_field: Field containing chunk ID
        
        Returns:
            List of vector-ready documents
        """
        if not chunks:
            return []
        
        # Extract texts
        texts = [chunk.get(text_field, "") for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = self.embed_texts(texts)
        
        # Build vector documents
        vector_docs = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_doc = {
                "id": chunk.get(id_field, f"chunk_{i}"),
                "vector": embedding,
                "metadata": {
                    k: v for k, v in chunk.items()
                    if k not in [text_field, id_field, "embedding"]
                },
                "text": chunk.get(text_field, "")
            }
            vector_docs.append(vector_doc)
        
        return vector_docs
    
    def get_dimension(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            Embedding vector dimension
        """
        return self.model.get_sentence_embedding_dimension()


# Global service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Get the global embedding service instance.
    
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def embed_text(text: str) -> List[float]:
    """
    Convenience function to embed a single text.
    
    Args:
        text: Input text
    
    Returns:
        Embedding vector
    """
    service = get_embedding_service()
    return service.embed_text(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convenience function to embed multiple texts.
    
    Args:
        texts: List of input texts
    
    Returns:
        List of embedding vectors
    """
    service = get_embedding_service()
    return service.embed_texts(texts)


def embed_query(query: str) -> List[float]:
    """
    Convenience function to embed a search query.
    
    Args:
        query: Search query
    
    Returns:
        Query embedding
    """
    service = get_embedding_service()
    return service.embed_query(query)
