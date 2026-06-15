from __future__ import annotations

import base64
import json
import logging
import mimetypes
import random
import time
from pathlib import Path
from typing import Any, Iterable

import httpx

from ..config import SETTINGS

log = logging.getLogger(__name__)

# Transient statuses worth retrying. 529 = MiniMax "server overloaded" (the
# common one); the rest are standard rate-limit / gateway hiccups. Hermes
# survives on the very same endpoint because it retries — so do we.
_RETRY_STATUS = {429, 500, 502, 503, 504, 529}
_MAX_ATTEMPTS = 4


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
        self._params: dict[str, str] = (
            {"GroupId": SETTINGS.minimax_group_id}
            if SETTINGS.minimax_group_id else {}
        )

    def close(self) -> None:
        self._client.close()

    def _post(self, path: str, payload: dict) -> httpx.Response:
        """POST with exponential backoff on transient overload (529) / 5xx /
        timeouts. Raises the last error if every attempt fails."""
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                r = self._client.post(
                    path, json=payload, params=self._params or None
                )
                if r.status_code in _RETRY_STATUS:
                    raise httpx.HTTPStatusError(
                        f"transient {r.status_code}", request=r.request, response=r
                    )
                r.raise_for_status()
                return r
            except (httpx.HTTPStatusError, httpx.TransportError, httpx.TimeoutException) as exc:
                # Non-retryable HTTP errors (e.g. 4xx other than 429) fail fast.
                if isinstance(exc, httpx.HTTPStatusError) and \
                        exc.response.status_code not in _RETRY_STATUS:
                    raise
                last_exc = exc
                if attempt == _MAX_ATTEMPTS:
                    break
                # 1s, 2s, 4s + jitter — gives MiniMax time to drain its queue.
                delay = (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                log.warning(
                    "MiniMax %s attempt %d/%d failed (%s); retrying in %.1fs",
                    path, attempt, _MAX_ATTEMPTS, exc.__class__.__name__, delay,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

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
        return self._post("/text/chatcompletion_v2", payload).json()

    def vision_chat(
        self,
        messages: list[dict],
        image_path: Path,
        *,
        temperature: float = 0.1,
    ) -> dict:
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        data_url = f"data:{mime};base64,{b64}"

        # build fresh outer list AND fresh last-message dict so the caller's
        # objects are never mutated
        msgs = list(messages[:-1])
        last = dict(messages[-1])
        last_content = last.get("content")
        if isinstance(last_content, str):
            last["content"] = [
                {"type": "text", "text": last_content},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        else:
            last["content"] = list(last_content or []) + [
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        msgs.append(last)
        del b64, data_url  # free immediately
        return self.chat(msgs, model=SETTINGS.vision_model, temperature=temperature)

    def embed(
        self, texts: Iterable[str], *, type_: str = "db"
    ) -> list[list[float]]:
        """Embed texts.

        MiniMax embo-01 is asymmetric: pass type_='db' when indexing
        passages, type_='query' when embedding user queries. Mixing types
        silently degrades cosine similarity.
        """
        data = self._post(
            "/embeddings",
            {
                "model": SETTINGS.embed_model,
                "texts": list(texts),
                "type": type_,
            },
        ).json()
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
