"""
AI Agents module initialization.

This module provides three specialized AI agents:
1. Retrieval Agent - Finds relevant documents
2. Analysis Agent - Analyzes and extracts insights
3. Recommendation Agent - Generates actionable recommendations
"""

from backend.agents.agents import (
    AgentType,
    AgentResponse,
    RetrievalAgent,
    AnalysisAgent,
    RecommendationAgent,
    AgentPipeline,
    get_agent_pipeline,
    run_agent_pipeline
)

__all__ = [
    "AgentType",
    "AgentResponse",
    "RetrievalAgent",
    "AnalysisAgent",
    "RecommendationAgent",
    "AgentPipeline",
    "get_agent_pipeline",
    "run_agent_pipeline"
]
