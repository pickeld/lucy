from typing import Union, Dict
from config import config
from utiles.logger import logger
import requests
from requests.models import Response


def send_request(
    method: str,
    endpoint: str,
    payload: Union[Dict, None] = None,
    params: Union[Dict, None] = None,
) -> Dict:
    payload = payload or {}
    params = params or {}

    url = f"{config.waha_base_url}{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": config.waha_api_key,
        "stream": "true",
    }

    response = None
    try:
        if method.upper() == "POST":
            response = requests.post(url, json=payload, headers=headers, params=params)
        elif method.upper() == "PUT":
            response = requests.put(url, json=payload, headers=headers, params=params)
        else:
            response = requests.get(url, headers=headers, params=params)

        response.raise_for_status()
        return response.json()
    except requests.RequestException as req_err:
        if response is not None:
            logger.error(f"HTTP error occurred: {req_err} - Response: {response.text}")
        else:
            logger.error(f"HTTP error occurred: {req_err} - No response received")
        return {}
