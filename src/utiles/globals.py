from typing import Union, Dict
from config import config
from utiles.logger import logger
import requests
from requests.models import Response


def send_request(method: str, endpoint: str, payload: Union[Dict, None] = None, params: Union[Dict, None] = None):
    payload = payload or {}
    params = params or {}
    response = Response()
    headers = {"Content-Type": "application/json",
            "X-Api-Key": config.waha_api_key,
            "stream": "true"}
    url = f"{config.waha_base_url}{endpoint}"
    if method.upper() == "POST":
        response = requests.post(
            url, json=payload, headers=headers, params=params)
    elif method.upper() == "GET":
        response = requests.get(url, headers=headers, params=params)
    elif method.upper() == "PUT":
        response = requests.put(
            url, json=payload, headers=headers, params=params)
    return response
