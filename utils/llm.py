from __future__ import annotations

import json
import os
from typing import Any

from config import GEMINI_MODEL, LLM_TIMEOUT_SECONDS


class LLMClient:
    def __init__(self, model_name: str = GEMINI_MODEL):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str) -> str:
        if not self.available:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError("google-generativeai is not installed.") from exc
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                request_options={"timeout": LLM_TIMEOUT_SECONDS},
            )
            text = getattr(response, "text", "") or ""
        except Exception as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc
        if not text.strip():
            raise RuntimeError("Gemini returned an empty response.")
        return text.strip()

    def generate_json(self, prompt: str) -> dict[str, Any]:
        text = self.generate(prompt)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise RuntimeError("Gemini returned malformed JSON.")
