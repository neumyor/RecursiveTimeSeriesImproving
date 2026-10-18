from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable

from harnessing_ts.paths import REQUIRED_DSH_RUNTIME_FILES, frontend_root, project_root
from harnessing_ts.runtime_base import DEFAULT_PYTHON_VERSION, prepare_runtime_base, report


def setup_server(
    *,
    python_version: str = DEFAULT_PYTHON_VERSION,
    skip_frontend: bool = False,
    skip_runtime_base: bool = False,
    reporter: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Prepare all repository-local assets needed before the first server start."""
    root = project_root()
    result: dict[str, Any] = {
        "state": "running",
        "projectRoot": str(root),
        "dshRuntime": {"state": "pending"},
        "frontend": {"state": "skipped" if skip_frontend else "pending"},
        "runtimeBase": {"state": "skipped" if skip_runtime_base else "pending"},
    }
    report(reporter, "[1/4] Backend environment is synchronized by `uv run`.")

    if not skip_frontend:
        frontend_status = _prepare_frontend(frontend_root(), reporter)
        result["frontend"] = frontend_status
        if frontend_status["state"] != "ready":
            result.update({"state": "failed", "message": frontend_status["message"]})
            return result
    else:
        report(reporter, "[2/4] Frontend setup skipped by request.")

    report(reporter, "[3/4] Installing and validating the DeepSeek Harness runtime extensions.")
    dsh_runtime_status = _prepare_dsh_runtime(root / "backend" / "dsh_runtime", reporter)
    result["dshRuntime"] = dsh_runtime_status
    if dsh_runtime_status["state"] != "ready":
        result.update({"state": "failed", "message": dsh_runtime_status["message"]})
        return result

    if not skip_runtime_base:
        report(reporter, "[4/4] Preparing the shared machine runtime base.")
        runtime_status = prepare_runtime_base(python_version=python_version, reporter=reporter)
        result["runtimeBase"] = runtime_status
        if runtime_status.get("state") != "ready":
            result.update({"state": "failed", "message": runtime_status.get("message", "runtime base setup failed")})
            return result
    else:
        report(reporter, "[4/4] Runtime-base setup skipped by request.")

    result.update({
        "state": "ready",
        "message": "Server prerequisites are ready. Start ts-harness-server with the desired workspace and variant.",
    })
    report(reporter, "Setup complete. The server will initialize TS_HARNESS_WORKSPACE automatically on first start.")
    return result


def _prepare_frontend(root: Path, reporter: Callable[[str], None] | None) -> dict[str, Any]:
    bun = shutil.which("bun")
    if not bun:
        return {
            "state": "failed",
            "message": "bun executable not found in PATH; install Bun before running setup-server",
            "root": str(root),
        }
    timeout = int(os.getenv("TS_HARNESS_FRONTEND_SETUP_TIMEOUT", "900"))
    commands = ([bun, "install", "--frozen-lockfile"], [bun, "run", "build"])
    report(reporter, "[2/4] Installing frontend dependencies and building frontend/dist.")
    for command in commands:
        try:
            completed = subprocess.run(command, cwd=root, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "state": "failed",
                "message": f"frontend setup could not run {' '.join(command)}: {exc}",
                "root": str(root),
            }
        if completed.returncode != 0:
            return {
                "state": "failed",
                "message": f"frontend setup command failed: {' '.join(command)}",
                "command": " ".join(command),
                "returncode": completed.returncode,
                "root": str(root),
            }
    return {
        "state": "ready",
        "message": "frontend dependencies installed and production bundle built",
        "root": str(root),
        "dist": str(root / "dist"),
        "commands": [" ".join(command) for command in commands],
    }


def _prepare_dsh_runtime(root: Path, reporter: Callable[[str], None] | None) -> dict[str, Any]:
    npm = shutil.which("npm")
    if not npm:
        return {
            "state": "failed",
            "message": "npm executable not found in PATH; install Node.js 20+ before running setup-server",
            "root": str(root),
        }
    if not (root / "package.json").is_file() or not (root / "package-lock.json").is_file():
        return {
            "state": "failed",
            "message": "DeepSeek Harness runtime package.json or package-lock.json is missing",
            "root": str(root),
        }
    timeout = int(os.getenv("TS_HARNESS_DSH_RUNTIME_SETUP_TIMEOUT", "900"))
    try:
        completed = subprocess.run([npm, "ci"], cwd=root, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "state": "failed",
            "message": f"DeepSeek Harness runtime setup could not run npm ci: {exc}",
            "root": str(root),
        }
    if completed.returncode != 0:
        return {
            "state": "failed",
            "message": "DeepSeek Harness runtime setup command failed: npm ci",
            "command": "npm ci",
            "returncode": completed.returncode,
            "root": str(root),
        }
    required = [root / rel for rel in REQUIRED_DSH_RUNTIME_FILES]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        return {
            "state": "failed",
            "message": f"DeepSeek Harness runtime is incomplete after npm ci: {', '.join(missing)}",
            "root": str(root),
        }
    report(reporter, "DeepSeek Harness runtime extensions are ready.")
    return {
        "state": "ready",
        "message": "DeepSeek Harness Node runtime dependencies installed and validated",
        "root": str(root),
        "command": "npm ci",
    }
