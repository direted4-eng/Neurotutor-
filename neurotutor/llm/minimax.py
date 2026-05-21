from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Iterable

import httpx

from ..config import SETTINGS


class MiniMaxClient:
    """Thin MiniMax client. Text + vision chat, embeddings.

    Vision images are loaded one at a time and never kept in memory —
    we read → base64 → send → drop.
    """

    def __init__(self, *, timeout: float = 60.0) -> None:
        if not SETTINGS.minimax_api_key:
            raise RuntimeError("MINIMAX_API_KEY is not set")
        self._client = httpx.Client(
            base_url=SETTINGS.minimax_base_url,
            headers={
                "Authorization": f"Bearer {SETTINGS.minimax_api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": model or SETTINGS.text_model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        r = self._client.post("/text/chatcompletion_v2", json=payload)
        r.raise_for_status()
        return r.json()

    def vision_chat(
        self,
        messages: list[dict],
        image_path: Path,
        *,
        temperature: float = 0.1,
    ) -> dict:
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        # inject image into the last user message
        msgs = list(messages)
        last = msgs[-1]
        last_content = last.get("content")
        if isinstance(last_content, str):
            last["content"] = [
                {"type": "text", "text": last_content},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        else:
            last_content.append(
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )
        del b64  # free immediately
        return self.chat(msgs, model=SETTINGS.vision_model, temperature=temperature)

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        r = self._client.post(
            "/embeddings",
            json={
                "model": SETTINGS.embed_model,
                "texts": list(texts),
                "type": "db",
            },
        )
        r.raise_for_status()
        data = r.json()
        return data.get("vectors") or data.get("data") or []


def extract_tool_calls(response: dict) -> list[dict]:
    """Normalize tool calls from a chat completion response."""
    try:
        msg = response["choices"][0]["message"]
    except (KeyError, IndexError):
        return []
    calls = msg.get("tool_calls") or []
    out = []
    for c in calls:
        fn = c.get("function", {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        out.append({"id": c.get("id"), "name": fn.get("name"), "arguments": args})
    return out


def extract_text(response: dict) -> str:
    try:
        return response["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError):
        return ""
