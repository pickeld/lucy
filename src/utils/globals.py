"""Global utility functions for WhatsApp-GPT application.

Provides HTTP request helpers with retry logic for external API calls.
"""

from typing import Any, Dict, Optional, Union

import requests
from requests.models import Response
from retry import retry

from config import settings
from utils.exceptions import WAHAAPIError
from utils.logger import logger


@retry(
    exceptions=(requests.RequestException, requests.Timeout),
    tries=3,
    delay=1,
    backoff=2,
    max_delay=10,
    logger=None  # We'll log manually for better control
)
def send_request(
    method: str,
    endpoint: str,
    payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30
) -> Union[Dict[str, Any], Response]:
    """Send HTTP request to WAHA API with automatic retry on failure.
    
    Args:
        method: HTTP method (GET, POST, PUT, DELETE)
        endpoint: API endpoint path (e.g., '/api/sendText')
        payload: Optional JSON body for POST/PUT requests
        params: Optional query parameters
        timeout: Request timeout in seconds (default: 30)
        
    Returns:
        JSON response as dictionary, or Response object for non-JSON responses
        
    Raises:
        WAHAAPIError: When the API request fails after all retries
    """
    payload = payload or {}
    params = params or {}

    url = f"{settings.waha_base_url}{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": settings.waha_api_key,
    }

    response: Optional[Response] = None
    
    try:
        method_upper = method.upper()
        
        if method_upper == "POST":
            response = requests.post(
                url, json=payload, headers=headers, params=params, timeout=timeout
            )
        elif method_upper == "PUT":
            response = requests.put(
                url, json=payload, headers=headers, params=params, timeout=timeout
            )
        elif method_upper == "DELETE":
            response = requests.delete(
                url, headers=headers, params=params, timeout=timeout
            )
        else:  # GET
            response = requests.get(
                url, headers=headers, params=params, timeout=timeout
            )

        response.raise_for_status()
        
        # Try to parse as JSON, return response object if not JSON
        try:
            return response.json()
        except ValueError:
            return response
            
    except requests.Timeout as e:
        error_msg = f"Request timeout after {timeout}s"
        logger.error(f"WAHA API timeout: {error_msg} | endpoint={endpoint}")
        raise WAHAAPIError(error_msg) from e
        
    except requests.RequestException as e:
        status_code = response.status_code if response is not None else None
        response_text = response.text if response is not None else None
        
        error_msg = f"HTTP {method_upper} {endpoint} failed"
        logger.error(
            f"WAHA API error: {error_msg} | "
            f"status={status_code} | "
            f"error={e} | "
            f"response={response_text[:200] if response_text else 'N/A'}"
        )
        
        raise WAHAAPIError(
            error_msg,
            status_code=status_code,
            response_body=response_text
        ) from e
