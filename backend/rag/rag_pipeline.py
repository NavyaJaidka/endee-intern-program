"""
RAG (Retrieval-Augmented Generation) Pipeline for AI Research & Code Copilot.

This module implements the complete RAG pipeline:
1. Query embedding
2. Vector similarity search in Endee
3. Context formatting
4. LLM response generation

The pipeline supports:
- Multiple LLM providers (OpenAI, Anthropic)
- Configurable retrieval parameters
- Source citation in responses
- Hybrid search (if applicable)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from backend.core.logging import get_logger
from backend.core.config import settings
from backend.vectorstore import EndeeClient, get_endee_client
from backend.embedding import EmbeddingService, get_embedding_service

logger = get_logger(__name__)


@dataclass
class RetrievedDocument:
    """
    Represents a retrieved document from the vector store.
    
    Attributes:
        chunk_id: Unique identifier
        text: Document text
        source: Source file/URL
        score: Similarity score
        metadata: Additional metadata
    """
    chunk_id: str
    text: str
    source: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """
    Represents a RAG pipeline response.
    
    Attributes:
        answer: Generated answer
        sources: List of source documents
        query: Original query
        metadata: Additional metadata
    """
    answer: str
    sources: List[RetrievedDocument]
    query: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMWrapper:
    """
    Wrapper for LLM providers (OpenAI, Anthropic).
    
    Provides a unified interface for different LLM APIs.
    Falls back to a mock response when no API key is available.
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize LLM wrapper.
        
        Args:
            provider: LLM provider (openai or anthropic)
        """
        self.provider = provider or settings.llm_provider
        self._client = None
        self._use_mock = False
        
        # Check if API key is available
        if self.provider == "openai" and not settings.openai_api_key:
            logger.warning("No OpenAI API key found. Using mock LLM responses.")
            self._use_mock = True
        elif self.provider == "anthropic" and not settings.anthropic_api_key:
            logger.warning("No Anthropic API key found. Using mock LLM responses.")
            self._use_mock = True
        
        logger.info(f"LLMWrapper initialized with provider: {self.provider}")
    
    @property
    def client(self):
        """Get or initialize the LLM client."""
        if self._use_mock:
            return None
            
        if self._client is None:
            if self.provider == "openai":
                self._client = self._get_openai_client()
            elif self.provider == "anthropic":
                self._client = self._get_anthropic_client()
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        
        return self._client
    
    def _get_openai_client(self):
        """Initialize OpenAI client."""
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens
        )
    
    def _get_anthropic_client(self):
        """Initialize Anthropic client."""
        from langchain_anthropic import ChatAnthropic
        
        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            temperature=settings.anthropic_temperature,
            max_tokens=settings.anthropic_max_tokens
        )
    
    def _generate_mock_response(self, query: str, context: str) -> str:
        """
        Generate a mock response when no API key is available.
        
        Args:
            query: User query
            context: Retrieved context
        
        Returns:
            Mock response text
        """
        if not context or context == "No relevant documents found.":
            return f"""I couldn't find any relevant documents to answer your query: "{query}"

Please try:
1. Uploading more documents
2. Rephrasing your query
3. Setting up an LLM API key (OPENAI_API_KEY or ANTHROPIC_API_KEY) for better results"""
        
        return f"""Based on the uploaded documents, here is the answer to your query: "{query}"

**Retrieved Context:**
{context}

---
*[Note: This is a simulated response because no LLM API key is configured. To enable full AI-powered responses with better summarization, please set either OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.]*"""
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
        
        Returns:
            Generated text
        """
        # Use mock if no API key
        if self._use_mock:
            return self._generate_mock_response(prompt, "")
        
        from langchain.schema import HumanMessage, SystemMessage
        
        messages = []
        
        if system_message:
            messages.append(SystemMessage(content=system_message))
        
        messages.append(HumanMessage(content=prompt))
        
        response = self.client.invoke(messages)
        
        return response.content
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate text with retrieval context.
        
        Args:
            query: User query
            context: Retrieved context
            system_message: System prompt
        
        Returns:
            Generated response
        """
        # Use mock if no API key - pass context for proper mock response
        if self._use_mock:
            return self._generate_mock_response(query, context)
        
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Instructions:
- Answer based ONLY on the provided context
- If the answer cannot be determined from the context, say so
- Be concise and specific
- Cite relevant sources when possible

Answer:"""
        
        default_system = "You are a helpful AI research assistant. Answer questions based on the provided context."
        
        return self.generate(prompt, system_message or default_system)


class RAGPipeline:
    """
    Complete RAG pipeline for question answering.
    
    This class implements:
    1. Query embedding using sentence-transformers
    2. Similarity search in Endee vector database
    3. Context formatting and truncation
    4. LLM response generation
    
    Attributes:
        embedding_service: Service for generating embeddings
        vector_client: Endee client for vector operations
        llm: LLM wrapper for text generation
    """
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_client: Optional[EndeeClient] = None,
        llm: Optional[LLMWrapper] = None,
        top_k: Optional[int] = None,
        max_context_length: Optional[int] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_service: Embedding service instance
            vector_client: Endee client instance
            llm: LLM wrapper instance
            top_k: Number of documents to retrieve
            max_context_length: Maximum context length in characters
        """
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_client = vector_client or get_endee_client()
        self.llm = llm or LLMWrapper()
        
        self.top_k = top_k or settings.top_k_documents
        self.max_context_length = max_context_length or settings.rag_max_context_length
        
        logger.info(
            f"RAGPipeline initialized with top_k={self.top_k}, "
            f"max_context_length={self.max_context_length}"
        )
    
    def retrieve(
        self,
        query: str,
        index_name: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents for a query.
        
        Steps:
        1. Embed the query
        2. Search Endee for similar vectors
        3. Format results as RetrievedDocument objects
        
        Args:
            query: Search query
            index_name: Index to search
            top_k: Number of results
            filters: Optional metadata filters
        
        Returns:
            List of retrieved documents
        """
        index_name = index_name or self.vector_client.index_name
        top_k = top_k or self.top_k
        
        # Generate query embedding
        logger.info(f"Embedding query: {query[:50]}...")
        query_embedding = self.embedding_service.embed_query(query)
        
        # Search vector store
        logger.info(f"Searching index {index_name} for top {top_k} results")
        
        try:
            results = self.vector_client.search(
                index_name=index_name,
                query_vector=query_embedding,
                k=top_k,
                filter_expr=filters
            )
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
        
        # Parse results
        documents = []
        
        # Endee returns results in different formats
        # Handle both JSON and MessagePack responses
        if isinstance(results, list):
            search_results = results
        elif isinstance(results, dict) and "results" in results:
            search_results = results["results"]
        else:
            logger.warning(f"Unexpected result format: {type(results)}")
            search_results = []
        
        for item in search_results:
            try:
                doc = RetrievedDocument(
                    chunk_id=item.get("id", item.get("chunk_id", "")),
                    text=item.get("text", item.get("content", "")),
                    source=item.get("source", item.get("metadata", {}).get("source", "unknown")),
                    score=item.get("score", item.get("distance", 1.0)),
                    metadata=item.get("metadata", {})
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to parse result: {e}")
                continue
        
        logger.info(f"Retrieved {len(documents)} documents")
        
        return documents
    
    def format_context(
        self,
        documents: List[RetrievedDocument]
    ) -> str:
        """
        Format retrieved documents into context string.
        
        Truncates context if it exceeds max_context_length.
        
        Args:
            documents: List of retrieved documents
        
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            # Estimate token length (roughly 4 chars per token)
            doc_length = len(doc.text)
            
            # Check if adding this document would exceed limit
            if current_length + doc_length > self.max_context_length:
                # Truncate remaining space
                remaining = self.max_context_length - current_length
                if remaining > 100:
                    truncated_text = doc.text[:remaining] + "..."
                    source_info = f"\n[Source {i+1}: {doc.source}]"
                    context_parts.append(truncated_text + source_info)
                break
            
            # Add document with source reference
            source_info = f"\n\n--- Document from {doc.source} ---\n"
            context_parts.append(source_info + doc.text)
            current_length += doc_length + len(source_info)
        
        return "\n".join(context_parts)
    
    def generate_answer(
        self,
        query: str,
        context: str,
        include_sources: bool = True
    ) -> str:
        """
        Generate answer using LLM with context.
        
        Args:
            query: User question
            context: Formatted context
            include_sources: Whether to include source citations
        
        Returns:
            Generated answer
        """
        logger.info("Generating answer with LLM")
        
        # Generate response
        response = self.llm.generate_with_context(
            query=query,
            context=context
        )
        
        return response
    
    def query(
        self,
        query: str,
        index_name: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_sources: bool = True
    ) -> RAGResponse:
        """
        Complete RAG pipeline query.
        
        Args:
            query: User question
            index_name: Index to search
            top_k: Number of documents to retrieve
            filters: Metadata filters
            include_sources: Include sources in response
        
        Returns:
            RAGResponse with answer and sources
        """
        logger.info(f"Processing RAG query: {query[:50]}...")
        
        # Step 1: Retrieve documents
        documents = self.retrieve(
            query=query,
            index_name=index_name,
            top_k=top_k,
            filters=filters
        )
        
        if not documents:
            return RAGResponse(
                answer="I couldn't find any relevant documents to answer your question. Please try uploading more documents or rephrasing your query.",
                sources=[],
                query=query,
                metadata={"status": "no_results"}
            )
        
        # Step 2: Format context
        context = self.format_context(documents)
        
        # Step 3: Generate answer
        answer = self.generate_answer(query, context, include_sources)
        
        # Step 4: Build response
        response = RAGResponse(
            answer=answer,
            sources=documents,
            query=query,
            metadata={
                "status": "success",
                "num_sources": len(documents),
                "top_score": documents[0].score if documents else 0.0
            }
        )
        
        logger.info("RAG query completed successfully")
        
        return response
    
    def query_with_sources(
        self,
        query: str,
        index_name: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query with explicit source information.
        
        Returns a dictionary with answer and sources for frontend display.
        
        Args:
            query: User question
            index_name: Index to search
            top_k: Number of documents
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        response = self.query(query, index_name, top_k)
        
        return {
            "answer": response.answer,
            "query": response.query,
            "sources": [
                {
                    "chunk_id": doc.chunk_id,
                    "text": doc.text[:500] + "..." if len(doc.text) > 500 else doc.text,
                    "source": doc.source,
                    "score": doc.score
                }
                for doc in response.sources
            ],
            "metadata": response.metadata
        }


# Global pipeline instance
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """
    Get the global RAG pipeline instance.
    
    Returns:
        RAGPipeline instance
    """
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def query_with_rag(
    query: str,
    index_name: Optional[str] = None,
    top_k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to query with RAG.
    
    Args:
        query: User question
        index_name: Index to search
        top_k: Number of documents
    
    Returns:
        RAG response dictionary
    """
    pipeline = get_rag_pipeline()
    return pipeline.query_with_sources(query, index_name, top_k)
