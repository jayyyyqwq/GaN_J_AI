#!/usr/bin/env python3
"""
Patch 02_grpo_selfplay.ipynb: rewrite paths to /tmp/workspace/,
and replace Cell 2 (pip install) and Cell 3 (clone+install) with no-op
stubs since start_server.sh now handles those on every HF Space boot.

Usage:
    python patch_notebook_paths.py
"""
import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent / "02_grpo_selfplay.ipynb"

# HF Space container runs as non-root, so /workspace is not writable.
# HOME=/tmp is always writable. We use $HOME/workspace = /tmp/workspace.
WORK_BASE = "/tmp/workspace"

def main():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    code_cell_idx = 0  # Counter for code cells only (skip markdown)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        code_cell_idx += 1
        source = cell["source"]

        # --- Cell 2 (code cell #2): pip install ---
        if code_cell_idx == 2:
            cell["source"] = [
                "# Already installed by start_server.sh on HF Space boot.\n",
                "# If running outside HF Space, uncomment:\n",
                "# !pip install -q unsloth trl\n",
                "print('Dependencies pre-installed by start_server.sh')\n",
            ]
            print(f"  Patched code cell #{code_cell_idx} (pip install -> skip stub)")
            continue

        # --- Cell 3 (code cell #3): clone + cd + install ---
        if code_cell_idx == 3:
            cell["source"] = [
                "# Already cloned by start_server.sh on HF Space boot.\n",
                "# If running outside HF Space, uncomment:\n",
                f"# !git clone https://github.com/jayyyyqwq/GaN_J_AI.git {WORK_BASE}/AgentGrid_V1\n",
                f"# %cd {WORK_BASE}/AgentGrid_V1\n",
                "# !pip install -q -e .\n",
                "import os\n",
                f"os.chdir('{WORK_BASE}/AgentGrid_V1')\n",
                "print('Working dir:', os.getcwd())\n",
            ]
            print(f"  Patched code cell #{code_cell_idx} (clone -> chdir stub)")
            continue

        # --- All other code cells: replace old paths with new base ---
        if isinstance(source, list):
            new_source = []
            changed = False
            for line in source:
                new_line = line
                # Replace any remaining /content/ or /workspace/ with the correct base
                for old in ["/content/", "/workspace/"]:
                    if old in new_line:
                        new_line = new_line.replace(old, f"{WORK_BASE}/")
                        changed = True
                new_source.append(new_line)
            if changed:
                cell["source"] = new_source
                print(f"  Patched code cell #{code_cell_idx}: paths -> {WORK_BASE}/")
        elif isinstance(source, str):
            new_source = source
            for old in ["/content/", "/workspace/"]:
                if old in new_source:
                    new_source = new_source.replace(old, f"{WORK_BASE}/")
            if new_source != source:
                cell["source"] = new_source
                print(f"  Patched code cell #{code_cell_idx}: paths -> {WORK_BASE}/")

    # Write back
    NOTEBOOK.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nNotebook saved: {NOTEBOOK}")

    # Verify no old path references remain
    raw = NOTEBOOK.read_text(encoding="utf-8")
    for old_prefix in ["/content/", '"/workspace/']:
        remaining = raw.count(old_prefix)
        if remaining == 0:
            print(f"  OK: zero '{old_prefix}' references")
        else:
            print(f"  WARNING: {remaining} '{old_prefix}' references still found")

    new_count = raw.count(WORK_BASE)
    print(f"  {WORK_BASE} occurrences: {new_count}")

if __name__ == "__main__":
    main()
