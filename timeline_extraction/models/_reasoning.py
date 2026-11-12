"""Explicit provider-native reasoning options, with known capability constraints.

Unknown/new hosted models are sent the native option and let the provider's
validation decide support; model selection is not restricted to an allowlist.
Sources: https://developers.openai.com/api/docs/guides/reasoning,
https://console.groq.com/docs/api-reference,
https://docs.together.ai/docs/inference/chat/reasoning,
https://ai.google.dev/gemini-api/docs/thinking.
"""

import re

EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


def reasoning_options(provider: str, model: str, effort: str | None):
    if effort in {None, "default"}:
        return {}
    if effort not in EFFORTS:
        raise ValueError(f"Unsupported reasoning effort: {effort!r}")
    name = model.lower().removeprefix("models/")
    if provider in {"antigravity", "codex", "claude"}:
        supported = {
            "antigravity": {"low", "medium", "high", "xhigh", "max"},
            "codex": {"minimal", "low", "medium", "high", "xhigh", "max"},
            "claude": {"low", "medium", "high", "max"},
        }[provider]
        if effort not in supported:
            raise ValueError(f"{provider} CLI does not support reasoning effort {effort}; supported: {', '.join(sorted(supported))}")
        return {"effort": effort}
    supported = None
    if provider == "huggingface":
        raise ValueError("Hugging Face local models have no generic reasoning_effort interface; use default")
    if provider == "openai":
        if re.match(r"^(?:gpt-[1-4](?:\D|$)|chatgpt-4|davinci|babbage)", name):
            raise ValueError(f"{model} does not support reasoning effort")
        if re.match(r"^o\d", name):
            supported = {"low", "medium", "high"}
        elif name.startswith(("gpt-6.1-sol", "gpt-6-astra")):
            supported = {"low", "medium", "high", "xhigh", "max"}
        if supported is not None and effort not in supported:
            raise ValueError(f"{model} does not support reasoning effort {effort}; supported: {', '.join(sorted(supported))}")
        return {"reasoning": {"effort": effort}}
    if provider == "google":
        if re.match(r"^gemini-(?:1|2\.0|decision)(?:\D|$)", name):
            raise ValueError(f"{model} does not support thinking levels")
        # GenerateContent's old 2.5 budget is not equivalent to effort levels.
        if name.startswith("gemini-2.5"):
            raise ValueError("Gemini 2.5 GenerateContent uses token thinking budgets, not effort levels; use default or a Gemini 3 model")
        supported = {"minimal", "low", "medium", "high"}
        if name.startswith("gemini-3.8-flash") or name.startswith("gemini-3.1-pro"):
            supported = {"low", "medium", "high"}
        elif name.startswith("gemini-3-pro"):
            supported = {"low", "high"}
        elif name.startswith("gemini-3.1-flash-lite-image"):
            supported = {"minimal", "high"}
        if effort not in supported:
            raise ValueError(f"{model} does not support thinking level {effort}; supported: {', '.join(sorted(supported))}")
        return {"thinking_level": effort.upper()}
    if provider == "groq":
        if name.startswith("openai/gpt-oss"):
            supported = {"low", "medium", "high"}
        elif name.startswith("qwen/qwen3.8"):
            supported = {"none", "low", "medium", "high"}
        elif name.startswith("qwen/qwen3-32b") or name.startswith("qwen/qwen3.6"):
            supported = {"none"}
        elif name.startswith(("llama", "meta-llama/", "gemma", "whisper")):
            raise ValueError(f"{model} does not support reasoning_effort")
    elif provider == "together":
        if name.startswith("openai/gpt-oss"):
            supported = {"low", "medium", "high"}
        elif name.startswith("deepseek-ai/deepseek-v4-pro"):
            supported = {"high", "max"}
        elif name.startswith(("meta-llama/", "google/gemma", "mistralai/")):
            raise ValueError(f"{model} does not support reasoning_effort")
    else:
        raise ValueError(f"Unsupported reasoning provider: {provider}")
    if supported is not None and effort not in supported:
        raise ValueError(f"{model} does not support reasoning effort {effort}; supported: {', '.join(sorted(supported))}")
    return {"reasoning_effort": effort}
