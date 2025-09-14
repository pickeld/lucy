from letta_client import AgentState, EmbeddingConfig, Letta, MessageCreate, MessageCreateContent, TextContent, ImageContent, CreateBlock
from letta_client.types.agent_state import AgentState
from typing import Union
from letta_client import Base64Image, ImageContent, TextContent
from utiles.logger import Logger


SUPPORTED_MEDIA_TYPES = {"image/jpeg", "image/png"}
logger = Logger()


class MemoryAgent:
    def __init__(self, recipient: str):
        self.llm_model_name = "letta-free"
        self.model = None
        self.recipient = recipient
        self.chat_id = recipient.replace("@", "_").replace(".", "_")
        self.client = Letta(base_url="http://localhost:8283")
        self.agent: AgentState = self.get_agent()
        logger.debug(
            f"Initialized MemoryAgent for {self.chat_id} with agent ID {self.agent.name}")

    def remember(self, text: str, role: str):
        pass
        if not text:
            return

        if len(text) > 1000:
            text = text[:1000] + "..."

        self.client.agents.passages.create(
            agent_id=self.agent.id,
            text=f"[{role}]: {text}"
        )

        logger.debug(
            f"Remembered text for {self.agent.name}: [{role}]: {text}")

    def get_models(self):
        models = self.client.models.list()
        for model in models:
            logger.debug(f"Available model: {model.model}")
            if model.model == self.llm_model_name:
                self.model = model
                return model
        raise ValueError(
            f"Model {self.llm_model_name} not found in available models.")

    def get_agent(self) -> AgentState:
        agents = self.client.agents.list(name=self.chat_id)
        return agents[0] if agents else self.set_agent()

    def set_agent(self) -> AgentState:
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
            enable_sleeptime=True
        )

    def get_recent_text_context(self, max_chars=3500, max_messages=20) -> str:
        messages = self.client.agents.messages.list(
            agent_id=self.agent.id, limit=max_messages)
        buffer = []
        total_chars = 0

        for msg in reversed(messages):  # Oldest to newest
            role = getattr(msg, "message_type", "")

            content = getattr(msg, "content", None)
            if role == "user_message" and "This is an automated system message" in content:
                pass
            # content = getattr(msg, "content", getattr(msg, "reasoning", None))
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

    def send_message(self, whatsapp_msg):
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

        response = self.client.agents.messages.create(
            agent_id=self.agent.id,
            messages=[MessageCreate(role="user", content=content)]
        )

        assistant_reply = next(
            m for m in response.messages if m.message_type == "assistant_message")
        # print(f"Assistant Reply: {assistant_reply.content}")
        return assistant_reply.content
