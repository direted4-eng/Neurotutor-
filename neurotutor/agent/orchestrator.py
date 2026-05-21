"""Hermes-style tool-calling loop.

Stateless agent: one input message → up to MAX_STEPS tool-call rounds →
one final assistant message. State lives in SQLite, never in the agent.
"""
from __future__ import annotations

import json
import logging

from ..llm.minimax import MiniMaxClient, extract_text, extract_tool_calls
from .persona import DEFAULT_PERSONA, PERSONAS, compose
from .roles import ROLES, Role
from .tools import TOOLS, schemas_for

log = logging.getLogger(__name__)

MAX_STEPS = 6


def run_turn(
    role_name: str,
    user_message: str,
    *,
    history: list[dict] | None = None,
    persona: str = DEFAULT_PERSONA,
) -> dict:
    role: Role = ROLES[role_name]
    pers = PERSONAS.get(persona, PERSONAS[DEFAULT_PERSONA])
    tools = schemas_for(role.tools)

    messages: list[dict] = [{"role": "system",
                             "content": compose(role.system, pers)}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    client = MiniMaxClient()
    trace: list[dict] = []
    try:
        for _ in range(MAX_STEPS):
            resp = client.chat(messages, tools=tools)
            calls = extract_tool_calls(resp)
            if not calls:
                text = extract_text(resp)
                if pers.signature and not text.rstrip().endswith(pers.signature):
                    text = f"{text.rstrip()}\n\n{pers.signature}"
                return {"role": role.name, "persona": pers.code,
                        "reply": text, "trace": trace,
                        "messages": messages + [
                            {"role": "assistant", "content": text}]}

            # echo assistant's tool-call message
            messages.append(resp["choices"][0]["message"])

            for call in calls:
                fn = TOOLS.get(call["name"])
                if not fn:
                    result = {"error": f"unknown tool {call['name']}"}
                else:
                    try:
                        result = fn(**(call["arguments"] or {}))
                    except Exception as e:  # surface to LLM, don't crash session
                        log.exception("tool %s failed", call["name"])
                        result = {"error": f"{type(e).__name__}: {e}"}
                trace.append({"tool": call["name"],
                              "args": call["arguments"], "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })
        return {"role": role.name,
                "reply": "[max tool-call steps reached]",
                "trace": trace, "messages": messages}
    finally:
        client.close()
