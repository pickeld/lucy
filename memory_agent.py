from letta_client import AgentState, EmbeddingConfig, Letta, MessageCreate, MessageCreateContent, TextContent, ImageContent, CreateBlock
from letta_client.types.agent_state import AgentState
from typing import Union, Optional
from letta_client import Base64Image, ImageContent, TextContent
from utiles.logger import Logger
from letta_client.core.api_error import ApiError

"""
Letta Memory Agent
Handles memory and interactions with the Letta API.
"""
SUPPORTED_MEDIA_TYPES = {"image/jpeg", "image/png"}
logger = Logger()


class MemoryAgent:
    def __init__(self, recipient: str):
        self.llm_model_name = "letta-free"
        self.model = None
        self.recipient = recipient
        self.chat_id = recipient.replace("@", "_").replace(".", "_")
        self.client = Letta(base_url="http://localhost:8283")
        self.agent: Optional[AgentState] = None
        self._initialize_agent()

    def _initialize_agent(self) -> None:
        """Initialize the agent with proper error handling."""
        try:
            if not self.model:
                self.get_models()
            self.agent = self.get_agent()
            logger.debug(
                f"Initialized MemoryAgent for {self.chat_id} with agent ID {self.agent.name}")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise

    def remember(self, text: str, role: str) -> bool:
        """
        Remember a piece of text with associated role.

        Args:
            text: The text to remember
            role: The role associated with the text

        Returns:
            bool: True if successful, False otherwise
        """
        if not text:
            logger.debug("Empty text provided to remember, skipping")
            return False

        if not self.agent:
            logger.error("Agent not properly initialized")
            return False

        # Truncate long messages
        if len(text) > 1000:
            text = text[:1000] + "..."

        try:
            self.client.agents.passages.create(
                agent_id=self.agent.id,
                text=f"[{role}]: {text}"
            )
            logger.debug(
                f"Remembered text for {self.agent.name}: [{role}]: {text}")
            return True
        except ApiError as e:
            logger.error(f"Failed to create passage: {str(e)}")
            # Try to reinitialize agent if it seems to be the issue
            if e.status_code in [404, 500]:
                try:
                    self._initialize_agent()
                    # Retry once after reinitialization
                    self.client.agents.passages.create(
                        agent_id=self.agent.id,
                        text=f"[{role}]: {text}"
                    )
                    logger.debug(
                        f"Successfully remembered text after agent reinitialization")
                    return True
                except Exception as retry_error:
                    logger.error(
                        f"Failed to remember text even after reinitialization: {str(retry_error)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in remember: {str(e)}")
            return False

    def get_models(self):
        """Get available models and set the one we want to use."""
        try:
            models = self.client.models.list()
            for model in models:
                if model.model == self.llm_model_name:
                    self.model = model
                    return model
            raise ValueError(
                f"Model {self.llm_model_name} not found in available models.")
        except Exception as e:
            logger.error(f"Failed to get models: {str(e)}")
            raise

    def get_agent(self) -> AgentState:
        """Get existing agent or create a new one."""
        try:
            agents = self.client.agents.list(name=self.chat_id)
            return agents[0] if agents else self.set_agent()
        except Exception as e:
            logger.error(f"Failed to get agent: {str(e)}")
            raise

    def set_agent(self) -> AgentState:
        """Create a new agent with the specified configuration."""
        try:
            return self.client.agents.create(
                name=self.chat_id,
                llm_config=self.model if self.model else self.get_models(),
                embedding_config=EmbeddingConfig(
                    embedding_endpoint_type="openai",
                    embedding_model="text-embedding-3-small",
                    embedding_dim=1024,
                ),
                memory_blocks=[
                    CreateBlock(value="", label="human"),
                    CreateBlock(value="", label="persona")
                ],
                enable_sleeptime=True,
                tags=["whatsapp", self.chat_id]
            )
        except Exception as e:
            logger.error(f"Failed to create agent: {str(e)}")
            raise

    def get_recent_text_context(self, max_chars=3500, max_messages=20) -> str:
        """Get recent message context up to specified limits."""
        if not self.agent:
            logger.error("Agent not properly initialized")
            return ""

        try:
            messages = self.client.agents.messages.list(
                agent_id=self.agent.id, limit=max_messages)
            buffer = []
            total_chars = 0

            for msg in reversed(messages):  # Oldest to newest
                role = getattr(msg, "message_type", "")

                content = getattr(msg, "content", None)
                if role == "user_message" and "This is an automated system message" in content:
                    continue

                if role == "reasoning_message":
                    content = getattr(msg, "reasoning", None)
                if content is None:
                    continue

                labeled_text = f"{role.replace('_message', '').capitalize()}: {content}"
                if total_chars + len(labeled_text) > max_chars:
                    break

                buffer.append(labeled_text)
                total_chars += len(labeled_text)
            return "\n".join(buffer)
        except Exception as e:
            logger.error(f"Error getting text context: {str(e)}")
            return ""

    def send_message(self, whatsapp_msg) -> Optional[str]:
        """Send a message through the agent and get the response."""
        if not self.agent:
            logger.error("Agent not properly initialized")
            return None

        try:
            content: list[Union[TextContent, ImageContent]] = []

            if whatsapp_msg.has_media and whatsapp_msg.media.type in SUPPORTED_MEDIA_TYPES:
                content.append(ImageContent(
                    source=Base64Image(
                        type="base64",
                        media_type=whatsapp_msg.media.type,
                        data=whatsapp_msg.media.base64
                    )
                ))
            if whatsapp_msg.quoted:
                if whatsapp_msg.quoted.type == "chat":
                    content.append(TextContent(
                        text=f"{whatsapp_msg.message}\n\n(Quoted): {whatsapp_msg.quoted.body}"))
                if whatsapp_msg.quoted.type == "image" and whatsapp_msg.quoted.mimetype in SUPPORTED_MEDIA_TYPES:
                    content.append(ImageContent(
                        source=Base64Image(
                            type="base64",
                            media_type=whatsapp_msg.quoted.mimetype,
                            data=whatsapp_msg.quoted.base64_data)
                    ))
            else:
                content.append(TextContent(text=f"{whatsapp_msg.message}"))

            response = self.client.agents.passages.create(
                agent_id=self.agent.id,
                messages=[MessageCreate(role="user", content=content)]
            )

            assistant_reply = next(
                m for m in response.messages if m.message_type == "assistant_message")
            return assistant_reply.content
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            return None
