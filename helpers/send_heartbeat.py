from letta_client import Letta

def send_heartbeat(client, agent_id: str):
    client.agents.messages.create(
        agent_id=agent_id,
        messages=[{"role": "user", "content": "heartbeat"}]
)



if __name__ == "__main__":
    client = Letta(base_url="http://localhost:8283")
    agents = client.agents.list()
    for i in range(5):
        for agent in agents:
            if not agent.name.endswith("sleeptime"):
                print(f"Agent: {agent.name} ({agent.id})")
                send_heartbeat(client, agent.id)