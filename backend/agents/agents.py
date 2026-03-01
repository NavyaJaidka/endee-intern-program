"""
AI Agents Module for AI Research & Code Copilot.

This module implements three specialized AI agents:

1. Retrieval Agent:
   - Finds relevant documents for a query
   - Uses vector similarity search
   - Filters by metadata
   - Returns ranked results

2. Analysis Agent:
   - Analyzes retrieved documents
   - Extracts key insights and patterns
   - Summarizes findings
   - Identifies relationships

3. Recommendation Agent:
   - Generates actionable recommendations
   - Provides code suggestions
   - Offers learning resources
   - Suggests next steps

Each agent is designed to work independently or together in a pipeline.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from backend.core.logging import get_logger
from backend.core.config import settings
from backend.rag import RAGPipeline, get_rag_pipeline

logger = get_logger(__name__)


class AgentType(Enum):
    """Types of AI agents."""
    RETRIEVAL = "retrieval"
    ANALYSIS = "analysis"
    RECOMMENDATION = "recommendation"


@dataclass
class AgentResponse:
    """
    Base response from an agent.
    
    Attributes:
        success: Whether the operation succeeded
        result: Result data
        error: Error message if failed
        metadata: Additional metadata
    """
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== Retrieval Agent ====================

class RetrievalAgent:
    """
    Retrieval Agent for finding relevant documents.
    
    This agent uses vector similarity search to find documents
    relevant to a user's query. It supports:
    - Keyword-based search
    - Semantic search via embeddings
    - Metadata filtering
    - Result ranking and scoring
    """
    
    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        default_top_k: Optional[int] = None
    ):
        """
        Initialize Retrieval Agent.
        
        Args:
            rag_pipeline: RAG pipeline instance
            default_top_k: Default number of results
        """
        self.rag_pipeline = rag_pipeline or get_rag_pipeline()
        self.default_top_k = default_top_k or settings.top_k_documents
        
        logger.info("RetrievalAgent initialized")
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None
    ) -> AgentResponse:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            index_name: Index to search
        
        Returns:
            AgentResponse with search results
        """
        try:
            top_k = top_k or self.default_top_k
            
            logger.info(f"RetrievalAgent: Searching for '{query[:50]}...'")
            
            # Perform search via RAG pipeline
            results = self.rag_pipeline.retrieve(
                query=query,
                top_k=top_k,
                filters=filters,
                index_name=index_name
            )
            
            if not results:
                return AgentResponse(
                    success=True,
                    result=[],
                    metadata={"query": query, "count": 0}
                )
            
            # Format results
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "chunk_id": doc.chunk_id,
                    "text": doc.text,
                    "source": doc.source,
                    "score": doc.score,
                    "metadata": doc.metadata
                })
            
            logger.info(f"RetrievalAgent: Found {len(formatted_results)} results")
            
            return AgentResponse(
                success=True,
                result=formatted_results,
                metadata={
                    "query": query,
                    "count": len(formatted_results),
                    "top_score": formatted_results[0]["score"] if formatted_results else 0
                }
            )
            
        except Exception as e:
            logger.error(f"RetrievalAgent error: {str(e)}")
            return AgentResponse(
                success=False,
                error=str(e),
                metadata={"query": query}
            )
    
    def search_by_keyword(
        self,
        keyword: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Search by keyword (simple text match).
        
        Args:
            keyword: Search keyword
            filters: Metadata filters
        
        Returns:
            AgentResponse with results
        """
        # Convert keyword to simple query
        return self.search(keyword, filters=filters)
    
    def get_similar_documents(
        self,
        document_id: str,
        top_k: Optional[int] = None,
        index_name: Optional[str] = None
    ) -> AgentResponse:
        """
        Find documents similar to a given document.
        
        Args:
            document_id: Document ID
            top_k: Number of similar documents
            index_name: Index to search
        
        Returns:
            AgentResponse with similar documents
        """
        try:
            # Get vector for the given document
            vector_client = self.rag_pipeline.vector_client
            
            vector_data = vector_client.get_vector(
                index_name=index_name or settings.endee_index_name,
                vector_id=document_id
            )
            
            if not vector_data:
                return AgentResponse(
                    success=False,
                    error=f"Document {document_id} not found"
                )
            
            # Search using the vector
            query_vector = vector_data.get("vector", [])
            
            results = vector_client.search(
                index_name=index_name or settings.endee_index_name,
                query_vector=query_vector,
                k=top_k or self.default_top_k
            )
            
            return AgentResponse(
                success=True,
                result=results,
                metadata={"document_id": document_id}
            )
            
        except Exception as e:
            logger.error(f"Similar documents error: {str(e)}")
            return AgentResponse(
                success=False,
                error=str(e)
            )


# ==================== Analysis Agent ====================

class AnalysisAgent:
    """
    Analysis Agent for extracting insights from documents.
    
    This agent analyzes retrieved documents to:
    - Summarize content
    - Extract key concepts
    - Identify relationships
    - Generate insights
    - Find patterns
    """
    
    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        """
        Initialize Analysis Agent.
        
        Args:
            rag_pipeline: RAG pipeline instance
        """
        self.rag_pipeline = rag_pipeline or get_rag_pipeline()
        
        logger.info("AnalysisAgent initialized")
    
    def analyze(
        self,
        documents: List[Dict[str, Any]],
        query: Optional[str] = None
    ) -> AgentResponse:
        """
        Analyze documents and extract insights.
        
        Args:
            documents: List of documents to analyze
            query: Optional query for context
        
        Returns:
            AgentResponse with analysis results
        """
        try:
            if not documents:
                return AgentResponse(
                    success=True,
                    result={"summary": "No documents to analyze."}
                )
            
            logger.info(f"AnalysisAgent: Analyzing {len(documents)} documents")
            
            # Extract key information
            analysis = {
                "document_count": len(documents),
                "sources": list(set(doc.get("source", "unknown") for doc in documents)),
                "key_themes": self._extract_themes(documents),
                "summary": self._generate_summary(documents),
                "insights": self._extract_insights(documents, query)
            }
            
            return AgentResponse(
                success=True,
                result=analysis,
                metadata={"analyzed_count": len(documents)}
            )
            
        except Exception as e:
            logger.error(f"AnalysisAgent error: {str(e)}")
            return AgentResponse(
                success=False,
                error=str(e)
            )
    
    def analyze_with_rag(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> AgentResponse:
        """
        Analyze documents retrieved via RAG.
        
        Args:
            query: Query to retrieve and analyze
            top_k: Number of documents
        
        Returns:
            AgentResponse with analysis
        """
        try:
            # Retrieve documents
            documents = self.rag_pipeline.retrieve(
                query=query,
                top_k=top_k or settings.top_k_documents
            )
            
            # Convert to dict format
            doc_dicts = []
            for doc in documents:
                doc_dicts.append({
                    "chunk_id": doc.chunk_id,
                    "text": doc.text,
                    "source": doc.source,
                    "score": doc.score
                })
            
            # Analyze
            return self.analyze(doc_dicts, query)
            
        except Exception as e:
            logger.error(f"AnalysisAgent RAG error: {str(e)}")
            return AgentResponse(
                success=False,
                error=str(e)
            )
    
    def _extract_themes(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract key themes from documents."""
        # Simple keyword-based theme extraction
        # In production, could use NLP for better extraction
        all_text = " ".join(doc.get("text", "") for doc in documents)
        
        # Simple theme extraction based on common terms
        common_terms = [
            "api", "database", "function", "class", "model", "data",
            "algorithm", "query", "search", "embedding", "vector",
            "machine learning", "neural", "training", "inference"
        ]
        
        themes = []
        for term in common_terms:
            if term.lower() in all_text.lower():
                themes.append(term)
        
        return themes[:5]  # Return top 5 themes
    
    def _generate_summary(self, documents: List[Dict[str, Any]]) -> str:
        """Generate a summary of documents."""
        if not documents:
            return "No documents to summarize."
        
        # Simple extractive summary - first few sentences from first doc
        first_text = documents[0].get("text", "")
        
        # Get first 2-3 sentences
        import re
        sentences = re.split(r'[.!?]+', first_text)
        summary = ". ".join(sentences[:2]).strip()
        
        if len(documents) > 1:
            summary += f" (and {len(documents) - 1} more documents)"
        
        return summary
    
    def _extract_insights(
        self,
        documents: List[Dict[str, Any]],
        query: Optional[str] = None
    ) -> List[str]:
        """Extract key insights from documents."""
        insights = []
        
        # Check document diversity
        sources = set(doc.get("source", "unknown") for doc in documents)
        if len(sources) > 1:
            insights.append(f"Information gathered from {len(sources)} different sources")
        
        # Check score distribution
        scores = [doc.get("score", 0) for doc in documents]
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score > 0.8:
                insights.append("High relevance documents retrieved")
            elif avg_score > 0.5:
                insights.append("Moderate relevance - consider refining query")
        
        # Query-specific insights
        if query:
            insights.append(f"Analysis focused on: {query}")
        
        return insights


# ==================== Recommendation Agent ====================

class RecommendationAgent:
    """
    Recommendation Agent for generating actionable suggestions.
    
    This agent provides:
    - Code suggestions
    - Learning resources
    - Next steps
    - Best practices
    - Related topics
    """
    
    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        """
        Initialize Recommendation Agent.
        
        Args:
            rag_pipeline: RAG pipeline instance
        """
        self.rag_pipeline = rag_pipeline or get_rag_pipeline()
        
        logger.info("RecommendationAgent initialized")
    
    def recommend(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]] = None,
        recommendation_type: str = "general"
    ) -> AgentResponse:
        """
        Generate recommendations based on query and context.
        
        Args:
            query: User query
            context: Optional context documents
            recommendation_type: Type of recommendations
        
        Returns:
            AgentResponse with recommendations
        """
        try:
            logger.info(f"RecommendationAgent: Generating recommendations for '{query[:50]}...'")
            
            # Generate recommendations
            recommendations = {
                "type": recommendation_type,
                "query": query,
                "suggestions": self._generate_suggestions(query, context),
                "resources": self._suggest_resources(query, context),
                "next_steps": self._suggest_next_steps(query, context)
            }
            
            return AgentResponse(
                success=True,
                result=recommendations,
                metadata={"query": query}
            )
            
        except Exception as e:
            logger.error(f"RecommendationAgent error: {str(e)}")
            return AgentResponse(
                success=False,
                error=str(e)
            )
    
    def recommend_with_analysis(
        self,
        analysis_result: AgentResponse
    ) -> AgentResponse:
        """
        Generate recommendations based on analysis results.
        
        Args:
            analysis_result: Result from Analysis Agent
        
        Returns:
            AgentResponse with recommendations
        """
        if not analysis_result.success:
            return AgentResponse(
                success=False,
                error=analysis_result.error
            )
        
        analysis = analysis_result.result
        query = analysis_result.metadata.get("query", "")
        
        # Extract themes for recommendations
        themes = analysis.get("key_themes", [])
        
        return self.recommend(
            query=query,
            recommendation_type="analysis_based"
        )
    
    def _generate_suggestions(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]]
    ) -> List[str]:
        """Generate query-specific suggestions."""
        suggestions = []
        query_lower = query.lower()
        
        # Code-related suggestions
        if "code" in query_lower or "implement" in query_lower:
            suggestions.append("Review the code examples in the retrieved documents")
            suggestions.append("Consider edge cases and error handling")
        
        # API-related
        if "api" in query_lower:
            suggestions.append("Check API rate limits and authentication requirements")
            suggestions.append("Review error handling patterns in documentation")
        
        # ML/AI related
        if "model" in query_lower or "training" in query_lower:
            suggestions.append("Consider data preprocessing requirements")
            suggestions.append("Evaluate model performance metrics")
        
        # Database related
        if "database" in query_lower or "query" in query_lower:
            suggestions.append("Review indexing strategies for performance")
            suggestions.append("Consider query optimization techniques")
        
        # Default suggestions
        if not suggestions:
            suggestions.append("Explore related documentation sections")
            suggestions.append("Test the implementation with sample data")
        
        return suggestions
    
    def _suggest_resources(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, str]]:
        """Suggest learning resources."""
        resources = []
        query_lower = query.lower()
        
        # General resources
        resources.append({
            "type": "documentation",
            "title": "Official Documentation",
            "description": "Refer to official docs for latest updates"
        })
        
        # Query-specific resources
        if "python" in query_lower:
            resources.append({
                "type": "tutorial",
                "title": "Python Best Practices",
                "description": "PEP 8 style guide and best practices"
            })
        
        if "api" in query_lower:
            resources.append({
                "type": "guide",
                "title": "API Design Guide",
                "description": "RESTful API design principles"
            })
        
        if "machine learning" in query_lower or "ml" in query_lower:
            resources.append({
                "type": "course",
                "title": "ML Fundamentals",
                "description": "Core machine learning concepts"
            })
        
        return resources
    
    def _suggest_next_steps(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]]
    ) -> List[str]:
        """Suggest next steps for the user."""
        next_steps = [
            "Review the retrieved documents for detailed information",
            "Implement a prototype based on the examples",
            "Test with your own data or use case"
        ]
        
        # Add query-specific next steps
        query_lower = query.lower()
        
        if "endee" in query_lower:
            next_steps.append("Set up Endee vector database using Docker")
            next_steps.append("Try the example API calls from documentation")
        
        if "rag" in query_lower:
            next_steps.append("Experiment with different chunk sizes")
            next_steps.append("Try different embedding models")
        
        if "agent" in query_lower:
            next_steps.append("Extend agent capabilities for your use case")
            next_steps.append("Add custom tools to the agent pipeline")
        
        return next_steps[:5]  # Return top 5


# ==================== Multi-Agent Pipeline ====================

class AgentPipeline:
    """
    Pipeline that coordinates all three agents.
    
    Provides a unified interface for:
    - Retrieval -> Analysis -> Recommendation
    - Or any subset of agents
    """
    
    def __init__(self):
        """Initialize the agent pipeline."""
        self.retrieval_agent = RetrievalAgent()
        self.analysis_agent = AnalysisAgent()
        self.recommendation_agent = RecommendationAgent()
        
        logger.info("AgentPipeline initialized")
    
    def process(
        self,
        query: str,
        use_analysis: bool = True,
        use_recommendations: bool = True,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process query through the full agent pipeline.
        
        Args:
            query: User query
            use_analysis: Whether to use Analysis Agent
            use_recommendations: Whether to use Recommendation Agent
            top_k: Number of documents to retrieve
        
        Returns:
            Dictionary with results from all agents
        """
        logger.info(f"AgentPipeline: Processing query '{query[:50]}...'")
        
        results = {
            "query": query,
            "retrieval": None,
            "analysis": None,
            "recommendations": None
        }
        
        # Step 1: Retrieval
        retrieval_result = self.retrieval_agent.search(query, top_k=top_k)
        results["retrieval"] = {
            "success": retrieval_result.success,
            "documents": retrieval_result.result,
            "metadata": retrieval_result.metadata
        }
        
        if not retrieval_result.success:
            results["error"] = retrieval_result.error
            return results
        
        # Step 2: Analysis (optional)
        if use_analysis and retrieval_result.result:
            analysis_result = self.analysis_agent.analyze(
                retrieval_result.result,
                query=query
            )
            results["analysis"] = {
                "success": analysis_result.success,
                "result": analysis_result.result,
                "metadata": analysis_result.metadata
            }
        
        # Step 3: Recommendations (optional)
        if use_recommendations:
            # Use analysis results if available, otherwise use retrieval
            context = None
            if use_analysis and results.get("analysis", {}).get("success"):
                context = results["analysis"].get("result", {}).get("key_themes")
            
            recommendation_result = self.recommendation_agent.recommend(
                query=query,
                context=context
            )
            results["recommendations"] = {
                "success": recommendation_result.success,
                "result": recommendation_result.result,
                "metadata": recommendation_result.metadata
            }
        
        logger.info("AgentPipeline: Processing complete")
        
        return results
    
    def query(self, query: str) -> AgentResponse:
        """
        Simple query interface.
        
        Args:
            query: User query
        
        Returns:
            AgentResponse with combined results
        """
        results = self.process(query)
        
        return AgentResponse(
            success=True,
            result=results,
            metadata={"query": query}
        )


# Global agent pipeline
_agent_pipeline: Optional[AgentPipeline] = None


def get_agent_pipeline() -> AgentPipeline:
    """
    Get the global agent pipeline instance.
    
    Returns:
        AgentPipeline instance
    """
    global _agent_pipeline
    if _agent_pipeline is None:
        _agent_pipeline = AgentPipeline()
    return _agent_pipeline


def run_agent_pipeline(query: str) -> Dict[str, Any]:
    """
    Convenience function to run the full agent pipeline.
    
    Args:
        query: User query
    
    Returns:
        Pipeline results
    """
    pipeline = get_agent_pipeline()
    return pipeline.process(query)
