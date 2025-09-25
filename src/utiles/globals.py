from typing import Union, Dict
from config import config
from utiles.logger import logger
import requests
from requests.models import Response


def send_request(method: str, endpoint: str, payload: Union[Dict, None] = None, params: Union[Dict, None] = None) -> Dict:
    payload = payload or {}
    params = params or {}

    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": config.waha_api_key,
        "stream": "true"
    }

    url = f"{config.waha_base_url}{endpoint}"

    try:
        if method.upper() == "POST":
            response = requests.post(
                url, json=payload, headers=headers, params=params)
        elif method.upper() == "PUT":
            response = requests.put(
                url, json=payload, headers=headers, params=params)
        else:
            response = requests.get(url, headers=headers, params=params)

        response.raise_for_status()
        try:
            return response.json()
        except Exception as json_err:
            return response

    except requests.HTTPError as http_err:
        # Try to inspect JSON response for known errors
        response_json = response.json()
        logger.error(
            f"HTTP error occurred: {http_err} - Response: {response_json}")
