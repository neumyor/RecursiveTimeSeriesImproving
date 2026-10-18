from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from harnessing_ts.paths import project_root

LlmProtocol = Literal["deepseek", "openai-compat"]
LlmAuthMode = Literal["sdk-default", "manual"]


@dataclass(frozen=True)
class LlmConfig:
    # sdk-default means that the DeepSeek Harness runtime reads DEEPSEEK_* from
    # its environment.  manual passes the workspace settings explicitly.
    authMode: LlmAuthMode = "sdk-default"
    model: str = ""
    apiKey: str | None = None
    baseUrl: str | None = None
    protocol: LlmProtocol | None = None
    # Cap the runtime's per-request max_tokens: the DSH default is 256k, which
    # gateways that validate against the model's own output limit reject with
    # a 400 that the runtime swallows as an empty turn.
    maxTokens: int | None = None


@dataclass(frozen=True)
class SdkInvocationConfig:
    model: str | None = None
    provider: str = "deepseek-official"
    max_tokens: int | None = None
    env: dict[str, str] | None = None


def read_effective_llm_config(workspace_path: Path) -> LlmConfig:
    file_config = _read_llm_config_file(workspace_path)
    env_config = _read_llm_env_config()
    return LlmConfig(
        authMode=env_config.authMode if os.getenv("TS_HARNESS_LLM_AUTH_MODE") else file_config.authMode,
        model=env_config.model or file_config.model,
        apiKey=env_config.apiKey or file_config.apiKey,
        baseUrl=env_config.baseUrl or file_config.baseUrl,
        protocol=env_config.protocol or file_config.protocol,
        maxTokens=env_config.maxTokens or file_config.maxTokens,
    )


def build_sdk_invocation_config(cfg: LlmConfig) -> SdkInvocationConfig:
    model = cfg.model.strip() or None
    if cfg.authMode == "manual" and not (cfg.apiKey and cfg.apiKey.strip()):
        raise RuntimeError("Manual DeepSeek Harness configuration requires apiKey.")
    if cfg.authMode != "manual":
        return SdkInvocationConfig(model=model)
    env: dict[str, str] = {}
    if cfg.apiKey:
        env["DEEPSEEK_API_KEY"] = cfg.apiKey.strip()
    if cfg.baseUrl and cfg.baseUrl.strip():
        env["DEEPSEEK_BASE_URL"] = cfg.baseUrl.strip()
    return SdkInvocationConfig(model=model, max_tokens=cfg.maxTokens, env=env)


def mask_sdk_invocation_config(cfg: SdkInvocationConfig) -> dict[str, Any]:
    out: dict[str, Any] = {"provider": cfg.provider}
    if cfg.model:
        out["model"] = cfg.model
    if cfg.max_tokens:
        out["maxTokens"] = cfg.max_tokens
    if cfg.env:
        out["env"] = {
            "DEEPSEEK_BASE_URL": cfg.env.get("DEEPSEEK_BASE_URL"),
            "DEEPSEEK_API_KEY": _mask_secret(cfg.env.get("DEEPSEEK_API_KEY")),
        }
    return out


def mask_llm_config(cfg: LlmConfig) -> dict[str, Any]:
    return {
        "authMode": cfg.authMode,
        "model": cfg.model,
        **({"apiKey": _mask_secret(cfg.apiKey)} if cfg.apiKey else {}),
        **({"baseUrl": cfg.baseUrl} if cfg.baseUrl else {}),
        **({"protocol": cfg.protocol} if cfg.protocol else {}),
        **({"maxTokens": cfg.maxTokens} if cfg.maxTokens else {}),
    }


def _read_llm_config_file(workspace_path: Path) -> LlmConfig:
    for path in (
        workspace_path / "config.llm.json",
        Path.cwd() / "config.llm.json",
        project_root() / "config.llm.json",
    ):
        if path.exists():
            return sanitize_llm_config(json.loads(path.read_text(encoding="utf-8")))
    return LlmConfig()


def _read_llm_env_config() -> LlmConfig:
    env_max_tokens = os.getenv("TS_HARNESS_LLM_MAX_TOKENS")
    return sanitize_llm_config({
        "authMode": os.getenv("TS_HARNESS_LLM_AUTH_MODE"),
        "protocol": os.getenv("TS_HARNESS_LLM_PROTOCOL"),
        "model": os.getenv("TS_HARNESS_LLM_MODEL"),
        "apiKey": os.getenv("TS_HARNESS_LLM_API_KEY"),
        "baseUrl": os.getenv("TS_HARNESS_LLM_BASE_URL"),
        "maxTokens": int(env_max_tokens) if env_max_tokens and env_max_tokens.isdigit() else None,
    })


def sanitize_llm_config(raw: dict[str, Any], *, default_auth_mode: str = "sdk-default") -> LlmConfig:
    """One sanitizer for every LLM config source: file reads, env reads and API writes."""
    auth = raw.get("authMode") if raw.get("authMode") in {"manual", "sdk-default"} else default_auth_mode
    protocol = raw.get("protocol") if raw.get("protocol") in {"deepseek", "openai-compat"} else None
    max_tokens = raw.get("maxTokens")
    if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1):
        max_tokens = None
    return LlmConfig(
        authMode=auth,
        model=raw.get("model") if isinstance(raw.get("model"), str) else "",
        apiKey=raw.get("apiKey") if isinstance(raw.get("apiKey"), str) else None,
        baseUrl=raw.get("baseUrl") if isinstance(raw.get("baseUrl"), str) else None,
        protocol=protocol,
        maxTokens=max_tokens,
    )


def _mask_secret(secret: str | None) -> str | None:
    if not secret:
        return None
    return "****" if len(secret) <= 8 else f"****{secret[-4:]}"
