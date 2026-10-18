from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def frontend_root() -> Path:
    return project_root() / "frontend"


def default_workspace_path() -> Path:
    configured = os.getenv("TS_HARNESS_WORKSPACE")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".harnessingts" / "workspaces" / "default").resolve()


def control_mode_from_env() -> str:
    raw = os.getenv("TS_HARNESS_CONTROL_MODE", "auto").strip().lower()
    if raw not in {"auto", "manual"}:
        raise ValueError(
            f"Invalid TS_HARNESS_CONTROL_MODE={raw!r}; expected 'auto' or 'manual'. "
            "Use auto for automatic node transitions or manual for UI approval."
        )
    return raw


# The package-locked DSH runtime's required files, relative to backend/dsh_runtime.
REQUIRED_DSH_RUNTIME_FILES = (
    "session-jsonrpc.mjs",
    "tool-allowlist.mjs",
    "node_modules/@deepseek-ai/dsh-sdk-jsonrpc-server/lib/index.js",
    "node_modules/@deepseek-ai/dsh-sdk-jsonrpc-demo/lib/bin.js",
    "node_modules/@deepseek-ai/dsh-mcp-client/lib/index.js",
    "node_modules/@deepseek-ai/dsh-tool-subagent/lib/index.js",
    "node_modules/@deepseek-ai/dsh-web/lib/index.js",
    "node_modules/@deepseek-ai/dsh-tool-web/lib/index.js",
)
