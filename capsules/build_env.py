#!/usr/bin/env python3
"""Writes each capsule's .codeocean/environment.json and environment/postInstall from environments.json.

    ./build_env.py              every capsule
    ./build_env.py ingest ...   only these
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def environment_json(common: dict, core: list[dict], capsule: dict) -> dict:
    return {
        "version": 1,
        "base_image": capsule.get("base_image", common["base_image"]),
        "post_install": True,
        "options": common["options"],
        "installers": {
            "pip3": {"packages": core + capsule["pip3"], "options": {}, "pre_install_options": {}},
            "vscode": {"packages": common["vscode"]["packages"], "version": common["vscode"]["version"]},
        },
    }


def post_install(common: dict, capsule: dict) -> str:
    blocks = [capsule.get("post_install", []), common["post_install"]]
    body = "\n\n".join("\n".join(lines) for lines in blocks if lines)
    return f"#!/usr/bin/env bash\nset -euo pipefail\n\n{body}\n"


def build(name: str, spec: dict) -> Path:
    capsule = spec["capsules"][name]
    root = HERE / name
    (root / ".codeocean").mkdir(parents=True, exist_ok=True)
    (root / "environment").mkdir(parents=True, exist_ok=True)
    env = environment_json(spec["common"], spec["core"], capsule)
    (root / ".codeocean" / "environment.json").write_text(json.dumps(env, indent="\t") + "\n")
    script = root / "environment" / "postInstall"
    script.write_text(post_install(spec["common"], capsule))
    script.chmod(0o755)
    return root


def main(names: list[str]) -> None:
    spec = json.loads((HERE / "environments.json").read_text())
    for name in names or list(spec["capsules"]):
        capsule = spec["capsules"][name]
        if capsule.get("base_image") == "":
            print(f"skipped {name}: set its base_image first ({capsule.get('base_image_note', '')})")
            continue
        print(f"wrote {build(name, spec).relative_to(HERE)}")


if __name__ == "__main__":
    main(sys.argv[1:])
