from letta_client import Letta


client = Letta(base_url="http://localhost:8283")
agents = client.agents.list()
for agent in agents:
    delete_agent = client.agents.delete(agent.id)
