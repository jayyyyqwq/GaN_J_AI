#!/usr/bin/env python3
"""
Patch 02_grpo_selfplay.ipynb: rewrite /content/ paths to /workspace/,
and replace Cell 2 (pip install) and Cell 3 (clone+install) with no-op
stubs since start_server.sh now handles those on every HF Space boot.

Usage:
    python patch_notebook_paths.py
"""
import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent / "02_grpo_selfplay.ipynb"

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
            # Replace with skip stub
            cell["source"] = [
                "# Already installed by start_server.sh on HF Space boot.\n",
                "# If running outside HF Space, uncomment:\n",
                "# !pip install -q unsloth trl\n",
                "print('Dependencies pre-installed by start_server.sh')\n",
            ]
            print(f"  Patched code cell #{code_cell_idx} (pip install → skip stub)")
            continue

        # --- Cell 3 (code cell #3): clone + cd + install ---
        if code_cell_idx == 3:
            cell["source"] = [
                "# Already cloned by start_server.sh on HF Space boot.\n",
                "# If running outside HF Space, uncomment:\n",
                "# !git clone https://github.com/jayyyyqwq/GaN_J_AI.git /workspace/AgentGrid_V1\n",
                "# %cd /workspace/AgentGrid_V1\n",
                "# !pip install -q -e .\n",
                "import os\n",
                "os.chdir('/workspace/AgentGrid_V1')\n",
                "print('Working dir:', os.getcwd())\n",
            ]
            print(f"  Patched code cell #{code_cell_idx} (clone → chdir stub)")
            continue

        # --- All other code cells: replace /content/ with /workspace/ ---
        if isinstance(source, list):
            new_source = []
            changed = False
            for line in source:
                if "/content/" in line:
                    new_line = line.replace("/content/", "/workspace/")
                    new_source.append(new_line)
                    changed = True
                else:
                    new_source.append(line)
            if changed:
                cell["source"] = new_source
                print(f"  Patched code cell #{code_cell_idx}: /content/ → /workspace/")
        elif isinstance(source, str):
            if "/content/" in source:
                cell["source"] = source.replace("/content/", "/workspace/")
                print(f"  Patched code cell #{code_cell_idx}: /content/ → /workspace/")

    # Write back
    NOTEBOOK.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n✓ Notebook saved: {NOTEBOOK}")

    # Verify no /content/ references remain
    raw = NOTEBOOK.read_text(encoding="utf-8")
    remaining = raw.count("/content/")
    if remaining == 0:
        print("✓ Verification passed: zero /content/ references in notebook.")
    else:
        print(f"⚠ {remaining} /content/ references still found — manual review needed.")

if __name__ == "__main__":
    main()
