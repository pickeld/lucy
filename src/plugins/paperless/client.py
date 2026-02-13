"""Paperless-NGX REST API client."""

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class PaperlessClient:
    """Client for Paperless-NGX REST API.
    
    API Documentation: https://docs.paperless-ngx.com/api/
    """
    
    def __init__(self, base_url: str, api_token: str):
        """Initialize client.
        
        Args:
            base_url: Paperless-NGX server URL (e.g., http://paperless:8000)
            api_token: API authentication token
        """
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Token {api_token}"}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def test_connection(self) -> bool:
        """Test API connectivity.
        
        Returns:
            True if connection successful
        """
        try:
            resp = self.session.get(f"{self.base_url}/api/", timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Paperless connection test failed: {e}")
            return False
    
    def get_documents(
        self,
        page: int = 1,
        page_size: int = 50,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch document list with pagination.
        
        Args:
            page: Page number (1-indexed)
            page_size: Results per page
            tags: Optional list of tag names to filter by
            
        Returns:
            API response with 'results', 'count', 'next', 'previous' keys
        """
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        
        if tags:
            # Build tags__name__in filter
            params["tags__name__in"] = ",".join(tags)
        
        resp = self.session.get(
            f"{self.base_url}/api/documents/",
            params=params,
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    
    def get_document_content(self, doc_id: int) -> str:
        """Fetch full text content of a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document text content
        """
        resp = self.session.get(
            f"{self.base_url}/api/documents/{doc_id}/",
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("content", "")
    
    def get_document_metadata(self, doc_id: int) -> Dict[str, Any]:
        """Fetch document metadata.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document metadata dict
        """
        resp = self.session.get(
            f"{self.base_url}/api/documents/{doc_id}/",
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
