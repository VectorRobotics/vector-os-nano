"""ClaudeProvider — LLM provider for Anthropic Claude via OpenRouter or direct API.

Uses httpx (synchronous) for HTTP calls. Supports:
- OpenRouter API (default): https://openrouter.ai/api/v1
- Direct Anthropic API: https://api.anthropic.com/v1

No ROS2 imports. No hardcoded secrets.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from vector_os_nano.core.types import TaskPlan, TaskStep
from vector_os_nano.llm.prompts import build_planning_prompt

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "anthropic/claude-haiku-4-5"
_DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
_REQUEST_TIMEOUT = 30.0  # seconds


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` code fences from LLM output.

    LLMs frequently wrap JSON in markdown code blocks. This strips them so
    json.loads() can parse the content.
    """
    # Match ```json ... ``` or ``` ... ``` (greedy — take the last ```)
    pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_plan_response(goal: str, raw_text: str) -> TaskPlan:
    """Parse an LLM text response into a TaskPlan.

    Handles:
    - Valid JSON with "steps" list
    - Clarification responses with "requires_clarification": true
    - Markdown-wrapped JSON (```json ... ```)
    - Malformed or empty responses (returns empty TaskPlan)

    Args:
        goal: the original planning goal (preserved in TaskPlan.goal).
        raw_text: raw text from the LLM response.

    Returns:
        TaskPlan — never raises.
    """
    if not raw_text or not raw_text.strip():
        return TaskPlan(goal=goal)

    cleaned = _strip_markdown_fences(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning("Failed to parse LLM response as JSON: %.200s", raw_text)
        return TaskPlan(goal=goal)

    if not isinstance(data, dict):
        log.warning("LLM response is not a JSON object: %s", type(data))
        return TaskPlan(goal=goal)

    # Extract message (new: AI's conversational text)
    ai_message = data.get("message")

    # Clarification response
    if data.get("requires_clarification"):
        return TaskPlan(
            goal=goal,
            steps=[],
            requires_clarification=True,
            clarification_question=data.get("clarification_question") or ai_message,
            message=ai_message,
        )

    # Normal plan response
    raw_steps = data.get("steps", [])
    if not isinstance(raw_steps, list):
        log.warning("LLM 'steps' field is not a list: %s", type(raw_steps))
        return TaskPlan(goal=goal, message=ai_message)

    steps: list[TaskStep] = []
    for raw in raw_steps:
        if not isinstance(raw, dict):
            log.warning("Skipping non-dict step: %s", raw)
            continue
        try:
            step = TaskStep.from_dict(raw)
            steps.append(step)
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("Skipping malformed step %s: %s", raw, exc)

    return TaskPlan(goal=goal, steps=steps, message=ai_message)


def parse_action_response(raw_text: str) -> dict:
    """Parse an LLM response for the agent loop into a single action dict.

    Returns one of:
        {"action": "skill_name", "params": {...}, "reasoning": "..."}
        {"done": true, "summary": "..."}

    Triple fallback:
        1. JSON parse (with markdown fence stripping)
        2. Regex extract action name
        3. Default to {"action": "scan"} (safe, non-destructive)

    Never raises.
    """
    if not raw_text or not raw_text.strip():
        log.warning("Empty LLM response for agent loop, defaulting to scan")
        return {"action": "scan", "params": {}, "reasoning": "empty response fallback"}

    cleaned = _strip_markdown_fences(raw_text)

    # Attempt 1: Full JSON parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            if data.get("done"):
                return data
            if "action" in data:
                data.setdefault("params", {})
                data.setdefault("reasoning", "")
                return data
    except json.JSONDecodeError:
        pass

    # Attempt 2: Regex extract action name
    m = re.search(r'"action"\s*:\s*"(\w+)"', raw_text)
    if m:
        action = m.group(1)
        log.warning("JSON parse failed, regex extracted action=%r", action)
        # Try to extract params too
        params: dict = {}
        pm = re.search(r'"params"\s*:\s*(\{[^}]*\})', raw_text)
        if pm:
            try:
                params = json.loads(pm.group(1))
            except json.JSONDecodeError:
                pass
        return {"action": action, "params": params, "reasoning": "regex fallback"}

    # Attempt 3: Safe default
    log.warning("Could not parse agent loop response: %.200s — defaulting to scan", raw_text)
    return {"action": "scan", "params": {}, "reasoning": "parse failure fallback"}


# ---------------------------------------------------------------------------
# ClaudeProvider
# ---------------------------------------------------------------------------

class ClaudeProvider:
    """LLM provider using Anthropic Claude via OpenRouter or direct API.

    Supports the LLMProvider protocol:
    - plan(): build system prompt, send to LLM, parse TaskPlan
    - query(): simple chat completion for free-form questions

    Args:
        api_key: API key for OpenRouter or Anthropic.
        model: model identifier (OpenRouter format by default).
        api_base: API base URL. Defaults to OpenRouter.
        max_history: maximum conversation turns to keep (default 20).
        temperature: sampling temperature (default 0.0 for determinism).
        max_tokens: max tokens in the completion (default 2048).
        timeout: HTTP request timeout in seconds (default 30.0).
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        api_base: str | None = None,
        max_history: int = 20,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = _REQUEST_TIMEOUT,
    ) -> None:
        self._api_key: str = api_key
        self.model: str = model
        self.api_base: str = api_base or _DEFAULT_API_BASE
        self.max_history: int = max_history
        self._temperature: float = temperature
        self._max_tokens: int = max_tokens
        self._endpoint: str = f"{self.api_base.rstrip('/')}/chat/completions"
        self._http: httpx.Client = httpx.Client(timeout=timeout)

    def __repr__(self) -> str:
        return f"ClaudeProvider(model={self.model!r}, api_base={self.api_base!r})"

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def plan(
        self,
        goal: str,
        world_state: dict[str, Any],
        skill_schemas: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
        model_override: str | None = None,
    ) -> TaskPlan:
        """Decompose a natural-language goal into a TaskPlan.

        Builds the planning system prompt from skill schemas and world state,
        sends the goal to the LLM, and parses the JSON response.

        Returns an empty TaskPlan (no steps) on any network or parse error.
        """
        system_prompt = build_planning_prompt(skill_schemas, world_state)

        messages: list[dict[str, Any]] = []

        # Inject history (trimmed to max_history)
        if history:
            trimmed = history[-self.max_history:]
            messages.extend(trimmed)

        messages.append({"role": "user", "content": goal})

        raw_response = self._chat_completion(system_prompt, messages, model_override)
        return parse_plan_response(goal, raw_response)

    def query(
        self,
        prompt: str,
        image: Any = None,
        model_override: str | None = None,
    ) -> str:
        """Send a free-form prompt and return the LLM's text response."""
        system_prompt = (
            "You are a helpful assistant for a robot system. "
            "Answer concisely and accurately."
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        return self._chat_completion(system_prompt, messages, model_override)

    def classify(self, user_message: str, model_override: str | None = None) -> str:
        """Classify user intent: chat | task | direct | query.

        Returns one of: "chat", "task", "direct", "query".
        Falls back to "chat" on any error.
        """
        from vector_os_nano.llm.prompts import build_classify_prompt
        system = build_classify_prompt(user_message)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        result = self._chat_completion(system, messages, model_override).strip().lower()
        if result in ("chat", "task", "direct", "query"):
            return result
        # Extract first valid word
        for word in result.split():
            if word in ("chat", "task", "direct", "query"):
                return word
        return "chat"

    def chat(
        self,
        user_message: str,
        system_prompt: str,
        history: list[dict[str, Any]] | None = None,
        model_override: str | None = None,
        image: Any = None,
    ) -> str:
        """Free-form chat with conversation history.

        Args:
            user_message: User text.
            system_prompt: System prompt.
            history: Prior conversation turns.
            model_override: Override model.
            image: Optional numpy array (H, W, 3) RGB image to include
                   in the user message via base64 encoding.
        """
        messages: list[dict[str, Any]] = []
        if history:
            messages.extend(history[-self.max_history:])

        if image is not None:
            # Build multimodal message with image
            content: list[dict[str, Any]] = [
                {"type": "text", "text": user_message},
            ]
            try:
                b64 = self._encode_image(image)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            except Exception as exc:
                log.warning("Failed to encode image: %s", exc)
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_message})

        return self._chat_completion(system_prompt, messages, model_override)

    @staticmethod
    def _encode_image(image: Any) -> str:
        """Encode a numpy RGB image to base64 JPEG string."""
        import base64
        import io
        try:
            from PIL import Image
            img = Image.fromarray(image)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except ImportError:
            pass
        # Fallback: cv2
        import cv2
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        import base64
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def summarize(
        self,
        original_request: str,
        execution_trace: str,
        model_override: str | None = None,
    ) -> str:
        """Summarize execution results for the user."""
        from vector_os_nano.llm.prompts import build_summarize_prompt
        system = build_summarize_prompt(original_request, execution_trace)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Summarize the execution results."}
        ]
        return self._chat_completion(system, messages, model_override)

    def decide_next_action(
        self,
        goal: str,
        observation: dict,
        skill_schemas: list[dict],
        history: list[dict],
        model_override: str | None = None,
    ) -> dict:
        """Decide the next action for the agent loop.

        Returns a dict with either {"action": ..., "params": ...} or {"done": true, "summary": ...}.
        """
        from vector_os_nano.llm.prompts import build_agent_loop_prompt
        system_prompt = build_agent_loop_prompt(
            goal=goal,
            observation=observation,
            skill_schemas=skill_schemas,
            history=history,
        )
        messages: list[dict] = [{"role": "user", "content": goal}]
        raw = self._chat_completion(system_prompt, messages, model_override)
        return parse_action_response(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat_completion(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        model_override: str | None = None,
    ) -> str:
        """POST to the chat completions endpoint and return the assistant text.

        Returns an error description string on any failure.
        """
        model = model_override or self.model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = self._http.post(self._endpoint, json=payload, headers=headers)
            resp.raise_for_status()
        except httpx.TimeoutException:
            log.warning("LLM request timed out")
            return "LLM error: request timed out"
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:300]
            log.warning("LLM HTTP error %s: %s", exc.response.status_code, body)
            return f"LLM error: HTTP {exc.response.status_code}"
        except httpx.RequestError as exc:
            log.warning("LLM network error: %s", exc)
            return f"LLM error: network error — {exc}"

        return self._extract_text(resp.json())

    def _extract_text(self, data: dict[str, Any]) -> str:
        """Extract assistant content text from an OpenAI-format response."""
        try:
            content = data["choices"][0]["message"].get("content") or ""
            return content
        except (KeyError, IndexError, TypeError) as exc:
            log.warning("Unexpected LLM response shape: %s", exc)
            return "LLM error: unexpected response format"
