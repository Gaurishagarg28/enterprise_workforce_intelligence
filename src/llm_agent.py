from __future__ import annotations

import os
from typing import Any


class WorkforceLLMAgent:
    """LLM reasoning/explanation layer. Business-critical scores stay deterministic."""

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.6")
        self._client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def explain_decision(self, intelligence: dict[str, Any]) -> str:
        if not self.enabled:
            return (
                "LLM explanation is disabled. The deterministic decision layer returned: "
                f"{intelligence['recommendation']['decision']}."
            )

        prompt = (
            "You are an HR workforce-intelligence explanation agent. Explain the supplied "
            "decision using only the supplied structured facts. Do not invent employee facts, "
            "do not make employment decisions, and do not expose sensitive attributes. "
            "Return concise reasoning, key evidence, and suggested human-review questions.\n\n"
            f"Structured intelligence:\n{intelligence}"
        )
        response = self._client.responses.create(model=self.model, input=prompt)
        return response.output_text
