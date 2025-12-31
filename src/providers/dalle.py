from config import config
from openai import OpenAI
from utils.logger import logger


class Dalle:
    def __init__(self) -> None:
        self.model = config.get("DALLE_MODEL", "dall-e-3")
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.context = ""
        self.prompt = ""
        
    def request(self) -> str | None:
        logger.info(f"Sending prompt to OpenAI DALL-E with context: {self.context} and prompt: {self.prompt}")
        response = self.client.images.generate(
            model=self.model,
            prompt=f"some earlier context: {self.context}, my request: {self.prompt}"
        )
        if response.data and len(response.data) > 0:
            return response.data[0].url
        return None