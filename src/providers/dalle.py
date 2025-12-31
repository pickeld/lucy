from config import config
from openai import OpenAI
from utils.logger import Logger

logger = Logger(__name__)

class Dalle:
    def __init__(self):
        self.model = config.dalle_model
        self.client = OpenAI(api_key=config.openai_api_key)
        self.context = ""
        self.prompt = ""
        
    def request(self):
        logger.info(f"Sending prompt to OpenAI DALL-E with context: {self.context} and prompt: {self.prompt}")
        response = self.client.images.generate(
            model=self.model,
            prompt=f"some erlier context: {self.context}, my request: {self.prompt}"
        )
        image_url = response.data[0].url
        return image_url