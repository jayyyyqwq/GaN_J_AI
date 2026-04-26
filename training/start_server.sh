#!/bin/bash
export HOME=/tmp
export PATH="/tmp/.local/bin:$PATH"
export USER=user
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache

WORK_DIR="$HOME/workspace"

if [ ! -d "$WORK_DIR/AgentGrid_V1" ]; then
    echo "=== Creating workspace ==="
    mkdir -p "$WORK_DIR"

    echo "=== Cloning repo ==="
    git clone https://github.com/jayyyyqwq/GaN_J_AI.git "$WORK_DIR/AgentGrid_V1"

    echo "=== Installing torch (cu118, pinned before unsloth) ==="
    pip install -q "torch==2.5.1+cu118" "torchvision==0.20.1+cu118" \
        --index-url https://download.pytorch.org/whl/cu118

    echo "=== Installing deps ==="
    cd "$WORK_DIR/AgentGrid_V1" && pip install -q -e .
    # unsloth 2026.x pulls torch 2.11 (CUDA 13) — incompatible with HF A100 driver 525 (CUDA 12.0)
    # 2024.11.5 is last version that works with torch 2.5.x + cu118
    pip install -q "unsloth==2024.11.5" trl plotly matplotlib pandas
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
