"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Any | None:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def _json_mode(plain_llm: Any, prompt: Any, schema: type[T], agent_name: str) -> T | None:
    """Ask for the schema as a trailing JSON object and validate it.

    Why this path exists (measured 2026-07-29 against the configured endpoint):
    LangChain's ``with_structured_output`` binds a *forced* tool choice
    (``tool_choice={"type":"tool","name":...}``).  Some Anthropic-compatible
    relays reject exactly that and nothing else::

        tool_choice={"type":"tool",...}  -> 400 InvalidParameter
        tool_choice={"type":"auto"}      -> works, returns a tool_use block
        no tools, "end with strict JSON" -> works

    So the provider *does* support tool use -- only the forced variant fails.
    Previously that 400 dropped straight through to free text, the model wrote
    prose with no ``**Rating**:`` header, and the rating landed in the database
    as a bogus "Hold".  Asking for trailing JSON needs no tool support at all,
    which makes it the most portable middle rung.

    Returns a validated schema instance, or ``None`` if the model did not emit
    parseable JSON (caller then falls through to free text).
    """
    try:
        fields = ", ".join(schema.model_fields)
    except Exception:  # pragma: no cover - non-pydantic schema
        return None

    instruction = (
        "\n\n---\n"
        "After your analysis, output a final line containing ONLY a JSON object "
        f"with exactly these keys: {fields}. No markdown fence, no commentary "
        "after it. The JSON must be the last thing in your reply."
    )
    try:
        text = plain_llm.invoke(_append_instruction(prompt, instruction)).content
    except Exception as exc:
        logger.warning("%s: json-mode call failed (%s)", agent_name, exc)
        return None

    # Scan from the end: the schema JSON is the last object in the reply, and
    # the prose above it may itself contain braces.
    for match in reversed(list(re.finditer(r"\{.*?\}", text, re.DOTALL))):
        try:
            return schema.model_validate_json(match.group(0))
        except Exception:
            continue
    logger.warning("%s: json-mode reply had no valid %s object",
                   agent_name, schema.__name__)
    return None


def _append_instruction(prompt: Any, instruction: str) -> Any:
    """Append text to whatever shape of prompt the caller passed."""
    if isinstance(prompt, str):
        return prompt + instruction
    if isinstance(prompt, list) and prompt:
        out = list(prompt)
        last = dict(out[-1])
        if isinstance(last.get("content"), str):
            last["content"] = last["content"] + instruction
            out[-1] = last
            return out
    return prompt


def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
    schema: type[T] | None = None,
    outcome: dict | None = None,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.
    """
    def _note(mode: str) -> None:
        if outcome is not None:
            outcome["mode"] = mode

    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            if result is None:
                # A thinking model can answer in plain text instead of calling
                # the tool, leaving the parser with nothing to return. Treat it
                # as a structured miss and fall back, with a clear reason.
                raise ValueError("structured output returned no parsed result")
            _note("structured")
            return render(result)
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); trying json mode",
                agent_name, exc,
            )

    # Middle rung: trailing-JSON. Keeps the typed shape (and therefore the
    # rendered `**Rating**:` header) even when forced tool choice is rejected.
    if schema is not None:
        result = _json_mode(plain_llm, prompt, schema, agent_name)
        if result is not None:
            _note("json")
            return render(result)

    # Last resort: free prose. The pipeline never blocks, but callers that pass
    # ``outcome`` can see this happened -- silently degrading here is exactly
    # how bogus "Hold" ratings reached the database before.
    logger.warning("%s: falling back to free-text generation", agent_name)
    _note("freetext")
    response = plain_llm.invoke(prompt)
    return response.content
