"""
Vector Store module initialization.

This module provides the Endee vector database integration for the
AI Research & Code Copilot system.
"""

from backend.vectorstore.endee_client import (
    EndeeClient,
    get_endee_client,
    reset_endee_client
)

__all__ = [
    "EndeeClient",
    "get_endee_client",
    "reset_endee_client"
]
