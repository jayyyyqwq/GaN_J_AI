"""
FastAPI application for AgentGrid Environment.

Usage:
    uvicorn agentgrid_env.server.app:app --reload --host 0.0.0.0 --port 8000
"""
import os

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .agentgrid_environment import AgentGridEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from agentgrid_env.server.agentgrid_environment import AgentGridEnvironment

hardware_url = os.getenv("HARDWARE_BRIDGE_URL", None)
episode_steps = int(os.getenv("EPISODE_STEPS", "50"))
max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "4"))


def _env_factory() -> AgentGridEnvironment:
    return AgentGridEnvironment(
        hardware_url=hardware_url,
        episode_steps=episode_steps,
    )


app = create_app(
    _env_factory,
    CallToolAction,
    CallToolObservation,
    env_name="agentgrid_env",
    max_concurrent_envs=max_concurrent,
)


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
