from letta_client import AgentState, EmbeddingConfig, Letta, MessageCreate, MessageCreateContent, TextContent, ImageContent, CreateBlock
from letta_client.types.agent_state import AgentState
from typing import List, Union, Optional
from letta_client import Base64Image, ImageContent, TextContent
from templates import PERSONA_TEMPLATE_GROUP, IDENTITY_POLICY_GLOBAL_TMPL
from utiles.logger import logger
from letta_client.core.api_error import ApiError
from whatsapp import WhatsappMSG
from config import config


LLM_MODEL_NAME = "letta-free"


class MemoryManager:
    def __init__(self):
        self.client = Letta(
            base_url=f"http://{config.LETTA_HOST}:{config.LETTA_PORT}")
        self.agents = {}
        self.global_agent = MemoryAgent("global", "Global Agent")

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
        self.client = Letta(
            base_url=f"http://{config.LETTA_HOST}:{config.LETTA_PORT}")

        self.model = None
        self.chat_name = chat_name
        self.chat_id = chat_id.replace("@", "_").replace(".", "_")
        self.is_group = True if chat_id.endswith("@g.us") else False

        self.tools = self.list_tools()
        self.get_set_agent()

    def remember(self, timestamp, sender, msg, *args, **kwargs) -> bool:
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
                            value=IDENTITY_POLICY_GLOBAL_TMPL.substitute()
                        )

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
