"""
Core configuration module for the AI Research & Code Copilot.

This module handles all configuration settings using environment variables
and provides centralized access to configuration values throughout the application.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Uses pydantic-settings for automatic environment variable loading
    with validation and type coercion.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Endee Vector Database Configuration
    endee_base_url: str = Field(
        default="http://localhost:8080",
        description="Base URL for Endee vector database API"
    )
    endee_auth_token: Optional[str] = Field(
        default=None,
        description="Authentication token for Endee (optional)"
    )
    endee_index_name: str = Field(
        default="research_copilot",
        description="Default index name in Endee"
    )
    endee_dimension: int = Field(
        default=384,
        description="Embedding dimension for vector storage"
    )
    
    # LLM Configuration
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: openai or anthropic"
    )
    
    # OpenAI
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    openai_model: str = Field(
        default="gpt-4-turbo-preview",
        description="OpenAI model name"
    )
    openai_temperature: float = Field(
        default=0.7,
        description="OpenAI temperature parameter"
    )
    openai_max_tokens: int = Field(
        default=2000,
        description="Maximum tokens for OpenAI response"
    )
    
    # Anthropic
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    anthropic_model: str = Field(
        default="claude-3-opus-20240229",
        description="Anthropic model name"
    )
    anthropic_temperature: float = Field(
        default=0.7,
        description="Anthropic temperature parameter"
    )
    anthropic_max_tokens: int = Field(
        default=2000,
        description="Maximum tokens for Anthropic response"
    )
    
    # Embedding Model
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-transformers model name"
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for embeddings: cpu or cuda"
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )
    
    # Document Processing
    chunk_size: int = Field(
        default=1000,
        description="Size of text chunks for RAG"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between text chunks"
    )
    max_file_size_mb: int = Field(
        default=50,
        description="Maximum file size for upload in MB"
    )
    
    # GitHub
    github_token: Optional[str] = Field(
        default=None,
        description="GitHub token for repo access"
    )
    
    # RAG Pipeline
    top_k_documents: int = Field(
        default=5,
        description="Number of top documents to retrieve"
    )
    rag_max_context_length: int = Field(
        default=4000,
        description="Maximum context length for RAG"
    )
    
    # Server
    host: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    port: int = Field(
        default=8000,
        description="Server port"
    )
    debug: bool = Field(
        default=False,
        description="Debug mode"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: str = Field(
        default="logs/app.log",
        description="Log file path"
    )
    
    def get_llm_config(self) -> dict:
        """
        Get LLM configuration based on provider.
        
        Returns:
            Dictionary with LLM configuration
        """
        if self.llm_provider == "openai":
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "temperature": self.openai_temperature,
                "max_tokens": self.openai_max_tokens
            }
        elif self.llm_provider == "anthropic":
            return {
                "provider": "anthropic",
                "api_key": self.anthropic_api_key,
                "model": self.anthropic_model,
                "temperature": self.anthropic_temperature,
                "max_tokens": self.anthropic_max_tokens
            }
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        Settings instance
    """
    return settings
