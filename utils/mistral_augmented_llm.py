"""
Mistral-compatible LLM wrapper for MCP Agent's OpenAI augmented LLM.
"""

import os
from typing import Any

from mcp_agent.config import OpenAISettings
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM


class MistralAugmentedLLM(OpenAIAugmentedLLM):
    """
    Route MCP Agent's OpenAI-compatible LLM path to Mistral configuration.

    MCP's OpenAIAugmentedLLM reads `context.config.openai` directly in several places.
    This class remaps that section from `context.config.mistral` (or env fallbacks)
    before parent initialization so model/base_url/api_key are consistent.
    """

    def _remap_openai_config_from_mistral(self, context: Any) -> None:
        if not context or not getattr(context, "config", None):
            return

        existing_openai = getattr(context.config, "openai", None)
        mistral_cfg = getattr(context.config, "mistral", {}) or {}

        api_key = (
            mistral_cfg.get("api_key")
            or os.environ.get("MISTRAL_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        base_url = (
            mistral_cfg.get("base_url")
            or os.environ.get("MISTRAL_BASE_URL")
            or "https://api.mistral.ai/v1"
        )
        default_model = mistral_cfg.get("default_model", "mistral-small-latest")

        reasoning_effort = getattr(existing_openai, "reasoning_effort", "medium")
        # Mistral chat endpoint rejects OpenAI's optional `user` field (422 extra_forbidden).
        user = None
        default_headers = getattr(existing_openai, "default_headers", None)
        context.config.openai = OpenAISettings(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            reasoning_effort=reasoning_effort,
            user=user,
            default_headers=default_headers,
        )

    def __init__(self, *args: Any, **kwargs: Any):
        context = kwargs.get("context")
        if context is None:
            for arg in args:
                if hasattr(arg, "config"):
                    context = arg
                    break

        # Try early remap if context is directly available.
        self._remap_openai_config_from_mistral(context)

        super().__init__(*args, **kwargs)

        # Context is usually attached by parent init; enforce remap again.
        post_context = getattr(self, "context", None)
        self._remap_openai_config_from_mistral(post_context)

        # Ensure request defaults align with mistral config after remap.
        if (
            post_context
            and getattr(post_context, "config", None)
            and getattr(post_context.config, "openai", None)
            and getattr(self, "default_request_params", None)
        ):
            self.default_request_params.model = post_context.config.openai.default_model
