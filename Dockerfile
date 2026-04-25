FROM python:3.11-slim

RUN useradd -m -u 1000 appuser

WORKDIR /app
COPY . .

# pyproject.toml carries openenv-core, fastmcp, fastapi, uvicorn, httpx, pydantic
# requirements.txt is intentionally NOT used here — it has gradio which conflicts
RUN pip install --no-cache-dir -e .

USER appuser
EXPOSE 7860

CMD ["uvicorn", "agentgrid_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
