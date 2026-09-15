"""Small dependency-free client for the DeepSeek OpenAI-compatible API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class DeepSeekStructuredClient:
    """Call DeepSeek Chat Completions and request a JSON response."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: int = 180,
        max_tokens: int = 4096,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is required for the DeepSeek provider"
            )
        self.base_url = (
            base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self._opener = opener or urlopen

    def complete(self, prompt: str, schema_path: Path) -> str:
        if not schema_path.exists():
            raise RuntimeError("structured output schema is missing: %s" % schema_path)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": prompt}],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                raw_response = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[-1200:]
            raise RuntimeError(
                "DeepSeek API request failed (%s): %s" % (exc.code, detail)
            ) from exc
        except URLError as exc:
            raise RuntimeError("DeepSeek API connection failed: %s" % exc.reason) from exc

        try:
            result = json.loads(raw_response)
            content = result["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("DeepSeek API returned an invalid response") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("DeepSeek API returned empty message content")
        return content
