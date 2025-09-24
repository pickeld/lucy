# delete all letta agents:


from letta_client import Letta, AgentState
from letta_client.core.api_error import ApiError


def delete_all_sources(client):
    try:
        sources = client.sources.list()
        print(f"Found {len(sources)} sources.")

        for src in sources:
            try:
                print(f"Deleting source: {src.name} ({src.id})")
                client.sources.delete(source_id=src.id)
            except ApiError as e:
                print(f"Could not delete {src.name}: {e}")

    except Exception as e:
        print(f"Failed to list or delete sources: {e}")


def delete_all_agents(client):
    client = Letta(base_url="http://localhost:8283")
    all_agents = client.agents.list()
    for agent in all_agents:
        print(f"Deleting agent: {agent.name} ({agent.id})")
        client.agents.delete(agent.id)
    print("All agents deleted.")


def list_agent_tools(client, agent_id: str = "agent-e338c3ba-3511-42e5-a6a9-63accea39f10"):
    tools = client.agents.tools.list(agent_id=agent_id)
    print(f"Tools for agent {agent_id}:")
    for tool in tools:
        print(f"- {tool.name} (ID: {tool.id})")


def delete_all_identities(client):
    identities = client.identities.list()
    for identity in identities:
        print(f"Deleting identity: {identity.name} ({identity.id})")
        client.identities.delete(identity.id)
    print("All identities deleted.")


def force_sleep_agents(client):
    all_agents = client.agents.list()
    for agent in all_agents:
        print(f"Forcing agent to sleep: {agent.name} ({agent.id})")
        client.agents.sleep.run(agent_id=agent.id)
    print("All agents forced to sleep.")


if __name__ == "__main__":
    client = Letta(base_url="http://localhost:8283")
    # delete_all_agents(client=client)
    # delete_all_sources(client=client)
    # delete_all_identities(cliendt=client)
    # list_agent_tools(client=client)
    # force_sleep_agents(client=client)
    client.agents.messages.create(
        agent_id="agent-cbd569d3-a985-4511-90cb-843741f5eb24",
        messages=[{"role": "user", "content": "Say OK"}],
        max_steps=5
    )

    a = client.agents.retrieve(
        agent_id="agent-cbd569d3-a985-4511-90cb-843741f5eb24")
    print(a.last_run_completion, a.last_run_duration_ms)
