"""
AgentGrid client.

Wraps MCPToolClient — all LLM interaction happens via tool calls:
    env.reset()
    obs = env.call_tool("get_observation", agent_id="A")
    env.call_tool("broadcast", agent_id="A", message="I have spare energy.")
    result = env.call_tool("get_step_result", agent_id="A")
"""

try:
    from openenv.core.mcp_client import MCPToolClient
except ImportError:
    from openenv.core.mcp_client import MCPToolClient


class AgentGridClient(MCPToolClient):
    pass
