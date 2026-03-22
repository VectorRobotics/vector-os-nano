"""ChatManager — async AI chat with conversation memory.

Manages multi-turn conversation with Claude Haiku via OpenRouter.
Detects robot commands vs general chat and routes accordingly.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)

_CHAT_SYSTEM_PROMPT = """\
You are Vector OS Nano's AI assistant. You control a robot arm (SO-101) through natural language.

Your capabilities:
- Execute robot commands: pick, place, home, scan, detect, open, close
- Answer questions about the robot, objects on the table, and system status
- Chat naturally in Chinese and English

When the user wants the robot to do something, respond with a brief acknowledgment, then the system will execute the command.

When the user asks a question or chats, respond naturally and helpfully.

Keep responses concise (1-3 sentences). Use the same language as the user.

Current robot state:
{state_info}

Objects on table:
{objects_info}
"""

# Commands that should be routed to Agent.execute()
_COMMAND_KEYWORDS = [
    "pick", "grab", "grasp", "抓", "拿",
    "place", "put", "放",
    "home", "回",
    "scan", "扫",
    "detect", "find", "look", "检测", "找", "看",
    "open", "打开", "张",
    "close", "关", "合",
]


def _is_robot_command(text: str) -> bool:
    """Heuristic: does this message look like a robot command?"""
    lower = text.lower().strip()
    return any(kw in lower for kw in _COMMAND_KEYWORDS)


class ChatManager:
    """Manages AI conversation with multi-turn memory.

    Args:
        api_key: OpenRouter API key.
        model: LLM model identifier.
        api_base: API base URL.
        max_history: max conversation turns to keep.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-haiku-4-5",
        api_base: str = "https://openrouter.ai/api/v1",
        max_history: int = 30,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._endpoint = f"{api_base.rstrip('/')}/chat/completions"
        self._max_history = max_history
        self._history: list[dict[str, str]] = []
        self._http = httpx.AsyncClient(timeout=30.0)

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    def add_system_message(self, content: str) -> None:
        """Add a system/execution result to history."""
        self._history.append({"role": "assistant", "content": content})
        self._trim_history()

    async def chat(
        self,
        user_message: str,
        state_info: str = "",
        objects_info: str = "",
    ) -> str:
        """Send a message and get AI response.

        Returns the AI's text response. Updates conversation history.
        """
        self._history.append({"role": "user", "content": user_message})
        self._trim_history()

        system = _CHAT_SYSTEM_PROMPT.format(
            state_info=state_info or "Unknown",
            objects_info=objects_info or "Unknown",
        )

        messages = [{"role": "system", "content": system}] + self._history

        try:
            resp = await self._http.post(
                self._endpoint,
                json={
                    "model": self._model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 1024,
                },
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            logger.warning("Chat LLM error: %s", exc)
            text = f"LLM error: {exc}"

        self._history.append({"role": "assistant", "content": text})
        self._trim_history()
        return text

    def is_command(self, text: str) -> bool:
        """Check if user message is a robot command."""
        return _is_robot_command(text)

    def _trim_history(self) -> None:
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    async def close(self) -> None:
        await self._http.aclose()
