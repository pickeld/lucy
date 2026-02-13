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
        """Test API connectivity with authentication validation.
        
        Uses /api/documents/?page_size=1 which requires a valid token,
        unlike /api/ which returns 200 even without authentication.
        
        Returns:
            True if connection and authentication successful
        """
        try:
            resp = self.session.get(
                f"{self.base_url}/api/documents/",
                params={"page_size": 1},
                timeout=5,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Paperless connection test failed: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Documents
    # -------------------------------------------------------------------------
    
    def get_documents(
        self,
        page: int = 1,
        page_size: int = 50,
        tags: Optional[List[int]] = None,
        exclude_tags: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Fetch document list with pagination.
        
        Args:
            page: Page number (1-indexed)
            page_size: Results per page
            tags: Optional list of tag **IDs** to filter by (include —
                documents must have ALL listed tags)
            exclude_tags: Optional list of tag IDs to exclude
            
        Returns:
            API response with 'results', 'count', 'next', 'previous' keys
        """
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        
        if tags:
            # tags__id__all = documents must have ALL of these tag IDs
            params["tags__id__all"] = ",".join(str(t) for t in tags)
        
        if exclude_tags:
            # Exclude documents that have any of these tag IDs
            params["tags__id__none"] = ",".join(str(t) for t in exclude_tags)
        
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
    
    def add_tag_to_document(self, doc_id: int, tag_id: int) -> bool:
        """Add a tag to a document.
        
        Fetches the document's current tags, appends the new tag ID,
        and PATCHes the document.
        
        Args:
            doc_id: Document ID
            tag_id: Tag ID to add
            
        Returns:
            True if successful
        """
        try:
            # Get current document data
            resp = self.session.get(
                f"{self.base_url}/api/documents/{doc_id}/",
                timeout=10,
            )
            resp.raise_for_status()
            doc = resp.json()
            
            # Get current tag IDs and add the new one
            current_tags = doc.get("tags", [])
            if tag_id in current_tags:
                return True  # Already tagged
            
            current_tags.append(tag_id)
            
            # PATCH the document with updated tags
            patch_resp = self.session.patch(
                f"{self.base_url}/api/documents/{doc_id}/",
                json={"tags": current_tags},
                timeout=10,
            )
            patch_resp.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to add tag {tag_id} to document {doc_id}: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Tags
    # -------------------------------------------------------------------------
    
    def get_tags(self) -> List[Dict[str, Any]]:
        """Fetch all tags.
        
        Returns:
            List of tag dicts with 'id', 'name', etc.
        """
        try:
            resp = self.session.get(
                f"{self.base_url}/api/tags/",
                params={"page_size": 1000},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as e:
            logger.error(f"Failed to fetch tags: {e}")
            return []
    
    def get_tag_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find a tag by its exact name.
        
        Args:
            name: Tag name to search for
            
        Returns:
            Tag dict if found, None otherwise
        """
        try:
            resp = self.session.get(
                f"{self.base_url}/api/tags/",
                params={"name__iexact": name},
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Failed to search for tag '{name}': {e}")
            return None
    
    def create_tag(self, name: str, color: str = "#a6cee3") -> Optional[Dict[str, Any]]:
        """Create a new tag.
        
        Args:
            name: Tag name
            color: Tag color in hex format (default: light blue)
            
        Returns:
            Created tag dict, or None on failure
        """
        try:
            resp = self.session.post(
                f"{self.base_url}/api/tags/",
                json={"name": name, "color": color},
                timeout=10,
            )
            resp.raise_for_status()
            tag = resp.json()
            logger.info(f"Created Paperless tag: '{name}' (id={tag.get('id')})")
            return tag
        except Exception as e:
            logger.error(f"Failed to create tag '{name}': {e}")
            return None
    
    def get_or_create_tag(self, name: str, color: str = "#a6cee3") -> Optional[int]:
        """Get a tag ID by name, creating it if it doesn't exist.
        
        Args:
            name: Tag name
            color: Tag color for creation (default: light blue)
            
        Returns:
            Tag ID, or None on failure
        """
        existing = self.get_tag_by_name(name)
        if existing:
            return existing["id"]
        
        created = self.create_tag(name, color)
        if created:
            return created["id"]
        
        return None
