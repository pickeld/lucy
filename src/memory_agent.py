from letta_client import AgentState, EmbeddingConfig, Letta, MessageCreate, MessageCreateContent, TextContent, ImageContent, CreateBlock
from letta_client.types.agent_state import AgentState
from typing import List, Union, Optional
from letta_client import Base64Image, ImageContent, TextContent
from templates import PERSONA_TEMPLATE_GROUP, IDENTITY_POLICY
from utiles.logger import logger
from letta_client.core.api_error import ApiError
from whatsapp import WhatsappMSG
from config import config


LLM_MODEL_NAME = "letta-free"


class MemoryManager:
    def __init__(self):
        self.agents = {}

    def get_agent(self, msg: WhatsappMSG) -> 'MemoryAgent':
        if msg.is_group:
            chat_id = msg.group.id
            chat_name = msg.group.name
        else:
            if msg.contact.is_me:
                chat_id = msg.to.replace("@c.us", "")
                chat_name = msg.contact.name
            else:
                chat_id = msg.contact.number
                chat_name = msg.contact.name

        if chat_id not in self.agents:
            self.agents[chat_id] = MemoryAgent(chat_id, chat_name)
        return self.agents[chat_id]


class MemoryAgent:
    def __init__(self, chat_id: str, chat_name: str):
        self.model = None
        self.chat_name = chat_name
        self.chat_id = chat_id.replace("@", "_").replace(".", "_")
        self.is_group = True if chat_id.endswith("@g.us") else False
        self.client = Letta(
            base_url=f"http://{config.LETTA_HOST}:{config.LETTA_PORT}")
        self.agent: AgentState = None
        self.tools = self.list_tools()
        self.get_set_agent()

    def remember(self, timestamp, sender, msg) -> bool:
        try:
            self.client.agents.passages.create(
                agent_id=self.agent.id,
                text=f"[{timestamp}] {sender} :: {msg}"
            )
            return True

        except Exception as e:
            logger.error(f"Unexpected error in remember: {str(e)}")
            raise

    def get_models(self):
        """Get available models and set the one we want to use."""
        try:
            models = self.client.models.list()
            for model in models:
                if model.model == LLM_MODEL_NAME:
                    self.model = model
                    return model
            raise ValueError(
                f"Model {LLM_MODEL_NAME} not found in available models.")
        except Exception as e:
            logger.error(f"Failed to get models: {str(e)}")
            raise

    def list_tools(self) -> List[str]:
        """List available tools."""
        try:
            tools = self.client.tools.list()
            return [tool.id for tool in tools]
        except Exception as e:
            logger.error(f"Failed to list tools: {str(e)}")
            return []

    def get_set_agent(self) -> AgentState:
        """Get existing agent or create a new one."""
        try:
            agents = self.client.agents.list(name=self.chat_id)
            self.agent = agents[0]
        except IndexError:
            logger.debug(
                f"No existing agent for {self.chat_id}, creating new one.")
            try:
                agent = self.client.agents.create(
                    name=self.chat_id,
                    llm_config=self.model if self.model else self.get_models(),
                    embedding_config=EmbeddingConfig(
                        embedding_endpoint_type="openai",
                        embedding_model="text-embedding-3-small",
                        embedding_dim=1536,
                    ),
                    memory_blocks=[
                        CreateBlock(
                            label="persona",
                            value=PERSONA_TEMPLATE_GROUP.format(
                                CHAT_TYPE="group" if self.is_group else "contact",
                                CHAT_NAME=self.chat_name)
                        ),
                        CreateBlock(
                            label="identity_policy",
                            value=IDENTITY_POLICY.format(
                                JID="test"
                            )
                        ),
                    ],
                    enable_sleeptime=True,
                    tags=["whatsapp", self.chat_id, self.chat_name],
                    include_base_tools=True,
                    # include_default_source=True,
                    timezone="Asia/Jerusalem",
                    include_base_tool_rules=True,
                    tool_ids=self.tools
                )
                self.agent = agent
            except Exception as e:
                logger.error(f"Failed to create agent: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Failed to get agent: {str(e)}")
            raise

    # def get_recent_text_context(self, max_chars=3500, max_messages=20) -> str:
    #     """Get recent message context up to specified limits."""
    #     if not self.agent:
    #         logger.error("Agent not properly initialized")
    #         return ""

    #     try:
    #         messages = self.client.agents.messages.list(
    #             agent_id=self.agent.id, limit=max_messages)
    #         buffer = []
    #         total_chars = 0

    #         for msg in reversed(messages):  # Oldest to newest
    #             role = getattr(msg, "message_type", "")

    #             content = getattr(msg, "content", None)
    #             if role == "user_message" and "This is an automated system message" in content:
    #                 continue

    #             if role == "reasoning_message":
    #                 content = getattr(msg, "reasoning", None)
    #             if content is None:
    #                 continue

    #             labeled_text = f"{role.replace('_message', '').capitalize()}: {content}"
    #             if total_chars + len(labeled_text) > max_chars:
    #                 break

    #             buffer.append(labeled_text)
    #             total_chars += len(labeled_text)
    #         return "\n".join(buffer)
    #     except Exception as e:
    #         logger.error(f"Error getting text context: {str(e)}")
    #         return ""

    # def send_message(self, whatsapp_msg) -> Optional[str]:
    #     """Send a message through the agent and get the response."""
    #     if not self.agent:
    #         logger.error("Agent not properly initialized")
    #         return None

    #     try:
    #         content: list[Union[TextContent, ImageContent]] = []

    #         if whatsapp_msg.has_media and whatsapp_msg.media.type in SUPPORTED_MEDIA_TYPES:
    #             content.append(ImageContent(
    #                 source=Base64Image(
    #                     type="base64",
    #                     media_type=whatsapp_msg.media.type,
    #                     data=whatsapp_msg.media.base64
    #                 )
    #             ))
    #         if whatsapp_msg.quoted:
    #             if whatsapp_msg.quoted.type == "chat":
    #                 content.append(TextContent(
    #                     text=f"{whatsapp_msg.message}\n\n(Quoted): {whatsapp_msg.quoted.body}"))
    #             if whatsapp_msg.quoted.type == "image" and whatsapp_msg.quoted.mimetype in SUPPORTED_MEDIA_TYPES:
    #                 content.append(ImageContent(
    #                     source=Base64Image(
    #                         type="base64",
    #                         media_type=whatsapp_msg.quoted.mimetype,
    #                         data=whatsapp_msg.quoted.base64_data)
    #                 ))
    #         else:
    #             content.append(TextContent(text=f"{whatsapp_msg.message}"))

    #         response = self.client.agents.passages.create(
    #             agent_id=self.agent.id,
    #             messages=[MessageCreate(role="user", content=content)]
    #         )

    #         assistant_reply = next(
    #             m for m in response.messages if m.message_type == "assistant_message")
    #         return assistant_reply.content
    #     except Exception as e:
    #         logger.error(f"Error sending message: {str(e)}")
    #         return None
