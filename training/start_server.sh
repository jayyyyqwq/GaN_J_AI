#!/bin/bash
export HOME=/tmp
export PATH="/tmp/.local/bin:$PATH"

# Auto-setup on every boot (skips if already done)
if [ ! -d /workspace/AgentGrid_V1 ]; then
    echo "=== Cloning repo ==="
    mkdir -p /workspace
    git clone https://github.com/jayyyyqwq/GaN_J_AI.git /workspace/AgentGrid_V1

    echo "=== Installing deps ==="
    cd /workspace/AgentGrid_V1 && pip install -q -e .
    pip install -q unsloth trl
fi

echo "=== Starting JupyterLab at /workspace ==="
jupyter lab \
  --ip=0.0.0.0 \
  --port=7860 \
  --no-browser \
  --allow-root \
  --notebook-dir=/workspace \
  --IdentityProvider.token='' \
  --ServerApp.password='' \
  --ServerApp.disable_check_xsrf=True \
  --ServerApp.tornado_settings="{'headers': {'Content-Security-Policy': 'frame-ancestors *'}}" \
  --IdentityProvider.cookie_options="{'SameSite': 'None', 'Secure': True}"
