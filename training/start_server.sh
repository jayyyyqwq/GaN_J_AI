#!/bin/bash
export HOME=/tmp
export PATH="/tmp/.local/bin:$PATH"

WORK_DIR="$HOME/workspace"

if [ ! -d "$WORK_DIR/AgentGrid_V1" ]; then
    echo "=== Creating workspace ==="
    mkdir -p "$WORK_DIR"

    echo "=== Cloning repo ==="
    git clone https://github.com/jayyyyqwq/GaN_J_AI.git "$WORK_DIR/AgentGrid_V1"

    echo "=== Installing deps ==="
    cd "$WORK_DIR/AgentGrid_V1" && pip install -q -e .
    pip install -q unsloth trl "pyopenssl>=24.0.0" plotly matplotlib pandas

    # Force torch to cu118 after unsloth (driver 525 supports CUDA 12.0 max, cu118 needs >= 520)
    pip install -q --force-reinstall \
        "torch==2.4.1+cu118" \
        "torchvision==0.19.1+cu118" \
        --index-url https://download.pytorch.org/whl/cu118
fi

echo "=== Starting JupyterLab at $WORK_DIR ==="
jupyter lab \
  --ip=0.0.0.0 \
  --port=7860 \
  --no-browser \
  --allow-root \
  --notebook-dir="$WORK_DIR" \
  --IdentityProvider.token='' \
  --ServerApp.password='' \
  --ServerApp.disable_check_xsrf=True \
  --ServerApp.tornado_settings="{'headers': {'Content-Security-Policy': 'frame-ancestors *'}}" \
  --IdentityProvider.cookie_options="{'SameSite': 'None', 'Secure': True}"
