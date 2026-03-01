"""
Endee Vector Database Client.

This module provides a Python client for interacting with the Endee vector database
via its HTTP API. It handles index creation, vector insertion, similarity search,
and metadata filtering.

Endee is a high-performance vector database that supports:
- HNSW indexing for fast approximate nearest neighbor search
- Dense and sparse vectors
- Metadata filtering (numeric, categorical, boolean)
- Multiple quantization levels (int8, int16, float32)

Documentation: https://github.com/endee-io/endee
"""

import json
import requests
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path

from backend.core.logging import get_logger
from backend.core.config import settings

logger = get_logger(__name__)


class EndeeClient:
    """
    Python client for Endee vector database.
    
    This client provides a high-level interface for interacting with Endee's
    HTTP API, handling authentication, request formatting, and response parsing.
    
    Attributes:
        base_url: Base URL for Endee API
        auth_token: Optional authentication token
        index_name: Current index name
        dimension: Vector dimension
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        """
        Initialize the Endee client.
        
        Args:
            base_url: Base URL for Endee API (default from settings)
            auth_token: Optional authentication token
            index_name: Default index name
            dimension: Vector dimension
        """
        self.base_url = base_url or settings.endee_base_url
        self.auth_token = auth_token or settings.endee_auth_token
        self.index_name = index_name or settings.endee_index_name
        self.dimension = dimension or settings.endee_dimension
        
        self._session = requests.Session()
        if self.auth_token:
            self._session.headers.update({"Authorization": self.auth_token})
        
        logger.info(f"EndeeClient initialized with base_url={self.base_url}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = self.auth_token
        return headers
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Endee API.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
        
        Returns:
            Response JSON
        
        Raises:
            requests.RequestException: On request failure
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self._session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=self._get_headers(),
                timeout=30
            )
            response.raise_for_status()
            
            # Handle empty responses
            if response.content:
                return response.json()
            return {"status": "success"}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {method} {url} - {str(e)}")
            raise
    
    # ==================== Index Management ====================
    
    def create_index(
        self,
        index_name: str,
        dimension: int,
        space_type: str = "cosine",
        m: int = 16,
        ef_construct: int = 200,
        precision: str = "int8"
    ) -> bool:
        """
        Create a new vector index.
        
        Args:
            index_name: Name of the index
            dimension: Vector dimension
            space_type: Distance metric (cosine, l2, ip)
            m: HNSW M parameter
            ef_construct: HNSW ef_construct parameter
            precision: Quantization level (int8, float32)
        
        Returns:
            True if successful
        """
        data = {
            "index_name": index_name,
            "dim": dimension,
            "space_type": space_type
        }
        
        logger.info(f"Creating index: {index_name} with dim={dimension}")
        result = self._make_request("POST", "/api/v1/index/create", data)
        
        # Endee returns "Index created successfully" on success
        return "success" in str(result).lower() or result.get("status") == "ok"
    
    def delete_index(self, index_name: str) -> bool:
        """
        Delete a vector index.
        
        Args:
            index_name: Name of the index to delete
        
        Returns:
            True if successful
        """
        logger.info(f"Deleting index: {index_name}")
        result = self._make_request(
            "DELETE",
            f"/api/v1/index/{index_name}/delete"
        )
        return "success" in str(result).lower()
    
    def list_indexes(self) -> List[Dict[str, Any]]:
        """
        List all available indexes.
        
        Returns:
            List of index information dictionaries
        """
        result = self._make_request("GET", "/api/v1/index/list")
        return result.get("indexes", [])
    
    def get_index_info(self, index_name: str) -> Dict[str, Any]:
        """
        Get information about an index.
        
        Args:
            index_name: Name of the index
        
        Returns:
            Index information dictionary
        """
        return self._make_request("GET", f"/api/v1/index/{index_name}/info")
    
    def index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists.
        
        Args:
            index_name: Name of the index
        
        Returns:
            True if index exists
        """
        try:
            self.get_index_info(index_name)
            return True
        except requests.exceptions.HTTPError:
            return False
    
    # ==================== Vector Operations ====================
    
    def insert_vectors(
        self,
        index_name: str,
        vectors: List[Dict[str, Any]]
    ) -> bool:
        """
        Insert vectors into an index.
        
        Args:
            index_name: Name of the index
            vectors: List of vector objects with:
                - id: Unique identifier
                - vector: Dense vector (list of floats)
                - sparse_indices: Optional sparse indices
                - sparse_values: Optional sparse values
                - meta: Optional metadata (JSON string)
                - filter: Optional filter expression
        
        Returns:
            True if successful
        """
        # Format vectors for Endee API
        formatted_vectors = []
        for vec in vectors:
            formatted = {}
            
            if "id" in vec:
                formatted["id"] = vec["id"]
            
            if "vector" in vec:
                formatted["vector"] = vec["vector"]
            
            if "sparse_indices" in vec and "sparse_values" in vec:
                formatted["sparse_indices"] = vec["sparse_indices"]
                formatted["sparse_values"] = vec["sparse_values"]
            
            if "metadata" in vec:
                formatted["meta"] = json.dumps(vec["metadata"])
            
            if "filter" in vec:
                formatted["filter"] = vec["filter"]
            
            if "norm" in vec:
                formatted["norm"] = vec["norm"]
            
            formatted_vectors.append(formatted)
        
        logger.debug(f"Inserting {len(vectors)} vectors into {index_name}")
        
        result = self._make_request(
            "POST",
            f"/api/v1/index/{index_name}/vector/insert",
            formatted_vectors
        )
        
        return "success" in str(result).lower() or result.get("status") == 200
    
    def insert_vector(
        self,
        index_name: str,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict] = None,
        filter_expr: Optional[Dict] = None
    ) -> bool:
        """
        Insert a single vector.
        
        Args:
            index_name: Name of the index
            vector_id: Unique identifier
            vector: Dense vector
            metadata: Optional metadata dictionary
            filter_expr: Optional filter expression
        
        Returns:
            True if successful
        """
        vec = {
            "id": vector_id,
            "vector": vector
        }
        
        if metadata:
            vec["meta"] = json.dumps(metadata)
        
        if filter_expr:
            vec["filter"] = json.dumps(filter_expr)
        
        return self.insert_vectors(index_name, [vec])
    
    def delete_vector(self, index_name: str, vector_id: str) -> bool:
        """
        Delete a vector by ID.
        
        Args:
            index_name: Name of the index
            vector_id: Vector ID to delete
        
        Returns:
            True if successful
        """
        result = self._make_request(
            "DELETE",
            f"/api/v1/index/{index_name}/vector/{vector_id}/delete"
        )
        return "success" in str(result).lower()
    
    def get_vector(self, index_name: str, vector_id: str) -> Optional[Dict]:
        """
        Get a vector by ID.
        
        Args:
            index_name: Name of the index
            vector_id: Vector ID
        
        Returns:
            Vector data or None if not found
        """
        result = self._make_request(
            "POST",
            f"/api/v1/index/{index_name}/vector/get",
            {"id": vector_id}
        )
        return result
    
    # ==================== Search Operations ====================
    
    def search(
        self,
        index_name: str,
        query_vector: List[float],
        k: int = 5,
        filter_expr: Optional[Dict] = None,
        include_vectors: bool = False,
        ef: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search.
        
        Args:
            index_name: Name of the index
            query_vector: Query vector
            k: Number of results to return
            filter_expr: Optional filter expression
            include_vectors: Include vectors in results
            ef: Search ef parameter (higher = more accurate but slower)
        
        Returns:
            List of search results with id, score, and metadata
        """
        data = {
            "vector": query_vector,
            "k": k,
            "include_vectors": include_vectors
        }
        
        if filter_expr:
            data["filter"] = json.dumps([filter_expr])
        
        if ef:
            data["ef"] = ef
        
        logger.debug(f"Searching index {index_name} with k={k}")
        
        # Endee returns MessagePack format
        try:
            response = self._session.post(
                f"{self.base_url}/api/v1/index/{index_name}/search",
                json=data,
                headers=self._get_headers(),
                timeout=30
            )
            response.raise_for_status()
            
            # Try to parse response as JSON first
            try:
                return response.json()
            except json.JSONDecodeError:
                # Try MessagePack format
                try:
                    import msgpack
                    results = msgpack.unpackb(response.content, raw=False)
                    # Format results to match expected structure
                    formatted_results = []
                    if isinstance(results, list):
                        for item in results:
                            formatted_results.append({
                                "id": item.get("id", ""),
                                "chunk_id": item.get("id", ""),
                                "text": item.get("text", item.get("content", "")),
                                "source": item.get("source", "unknown"),
                                "score": 1.0 - item.get("distance", 0.0),
                                "metadata": item.get("metadata", {})
                            })
                    return formatted_results
                except Exception as e:
                    logger.warning(f"Failed to parse response: {str(e)}, returning raw")
                    return [{"raw_response": response.content.decode('utf-8', errors='ignore')}]
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Search failed: {str(e)}")
            raise
    
    def search_by_id(
        self,
        index_name: str,
        vector_id: str,
        k: int = 5,
        include_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search using an existing vector by ID.
        
        Args:
            index_name: Name of the index
            vector_id: ID of the vector to search with
            k: Number of results
            include_vectors: Include vectors in results
        
        Returns:
            List of similar vectors
        """
        # First get the vector
        vector_data = self.get_vector(index_name, vector_id)
        
        if not vector_data:
            raise ValueError(f"Vector {vector_id} not found")
        
        # Extract vector from response
        query_vector = vector_data.get("vector", [])
        
        return self.search(
            index_name=index_name,
            query_vector=query_vector,
            k=k,
            include_vectors=include_vectors
        )
    
    # ==================== Metadata Filtering ====================
    
    def search_with_metadata_filter(
        self,
        index_name: str,
        query_vector: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search with metadata filtering.
        
        Supports:
        - Numeric filters: {"field": {"$gte": 10, "$lte": 20}}
        - Category filters: {"field": {"$in": ["value1", "value2"]}}
        - Boolean filters: {"field": true}
        
        Args:
            index_name: Name of the index
            query_vector: Query vector
            k: Number of results
            filters: Filter dictionary
            include_vectors: Include vectors in results
        
        Returns:
            Filtered search results
        """
        filter_expr = None
        if filters:
            # Convert filters to Endee format
            filter_expr = {}
            for field, value in filters.items():
                if isinstance(value, dict):
                    filter_expr[field] = value
                elif isinstance(value, list):
                    filter_expr[field] = {"$in": value}
                else:
                    filter_expr[field] = value
        
        return self.search(
            index_name=index_name,
            query_vector=query_vector,
            k=k,
            filter_expr=filter_expr,
            include_vectors=include_vectors
        )
    
    # ==================== Utility Methods ====================
    
    def ensure_index_exists(
        self,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Ensure index exists, create if it doesn't.
        
        Args:
            index_name: Index name (uses default if not provided)
            dimension: Vector dimension
            **kwargs: Additional index creation parameters
        
        Returns:
            Index name
        """
        index_name = index_name or self.index_name
        dimension = dimension or self.dimension
        
        if not self.index_exists(index_name):
            logger.info(f"Creating index {index_name}")
            self.create_index(index_name, dimension, **kwargs)
        
        return index_name
    
    def health_check(self) -> bool:
        """
        Check if Endee server is healthy.
        
        Returns:
            True if server is healthy
        """
        try:
            response = self._session.get(
                f"{self.base_url}/api/v1/health",
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def close(self):
        """Close the HTTP session."""
        self._session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Global client instance
_endee_client: Optional[EndeeClient] = None


def get_endee_client() -> EndeeClient:
    """
    Get the global Endee client instance.
    
    Returns:
        EndeeClient instance
    """
    global _endee_client
    if _endee_client is None:
        _endee_client = EndeeClient()
    return _endee_client


def reset_endee_client():
    """Reset the global Endee client."""
    global _endee_client
    if _endee_client:
        _endee_client.close()
    _endee_client = None
