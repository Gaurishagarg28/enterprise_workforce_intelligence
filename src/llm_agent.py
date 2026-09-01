from __future__ import annotations

import os
from typing import Any

import requests


class WorkforceLLMAgent:
    """Provider-agnostic explanation layer with resilient local fallback."""

    def __init__(self):
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    @property
    def provider(self) -> str:
        if self.openrouter_key:
            return "OpenRouter"
        if self.openai_key:
            return "OpenAI"
        return "Local fallback"

    @property
    def enabled(self) -> bool:
        return bool(self.openrouter_key or self.openai_key)

    def explain_decision(self, intelligence: dict[str, Any]) -> str:
        prompt = self._build_prompt(intelligence)
        if self.openrouter_key:
            try:
                return self._openrouter(prompt)
            except Exception as exc:
                return self._fallback_from_error(intelligence, "OpenRouter", exc)
        if self.openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key)
                response = client.responses.create(model=self.openai_model, input=prompt)
                return response.output_text
            except Exception as exc:
                return self._fallback_from_error(intelligence, "OpenAI", exc)
        return self._fallback_from_error(intelligence, "LLM", None)

    def _openrouter(self, prompt: str) -> str:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Gaurishagarg28/enterprise_workforce_intelligence",
                "X-Title": "Enterprise Workforce Intelligence",
            },
            json={
                "model": self.openrouter_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _build_prompt(intelligence: dict[str, Any]) -> str:
        return (
            "You are an HR workforce-intelligence explanation agent. Use ONLY the supplied structured facts. "
            "Do not invent employee facts, do not use sensitive demographic attributes, and do not make an autonomous employment decision. "
            "Produce exactly three concise sections: Evidence, Recommended human action, Review questions.\n\n"
            f"Structured intelligence:\n{intelligence}"
        )

    @staticmethod
    def _fallback_from_error(intelligence: dict[str, Any], provider: str, exc: Exception | None) -> str:
        recommendation = intelligence["recommendation"]["decision"]
        probability = float(intelligence["attrition"]["probability"])
        risk = intelligence["attrition"]["risk_level"]
        gap = intelligence["skills"].get("skill_gap", [])
        readiness = float(intelligence["skills"].get("readiness", 0))
        gap_text = ", ".join(gap) if gap else "no capability gap identified"
        if exc is None:
            status = f"{provider} is not configured"
        else:
            message = str(exc).lower()
            if "429" in message or "quota" in message or "credit_balance" in message:
                status = f"{provider} quota is unavailable"
            elif "401" in message or "authentication" in message:
                status = f"{provider} authentication failed"
            else:
                status = f"{provider} explanation service is unavailable"
        return (
            f"### Evidence\n{status}. The deterministic workforce engine remains authoritative. "
            f"Predicted attrition risk is {probability * 100:.1f}% ({risk}); reskill readiness is "
            f"{readiness:.1f}%; capability gap: {gap_text}.\n\n"
            f"### Recommended human action\nReview the **{recommendation}** recommendation with HR and the employee's manager before any employment action.\n\n"
            "### Review questions\nConfirm that the source data is current, verify the capability gap, and consider employee goals plus available internal development opportunities."
        )
