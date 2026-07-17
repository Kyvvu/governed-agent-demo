# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kyvvu B.V.

"""LLM factory for the governed Invoice / Expense assistant demo.

Modern LangChain agents (``langchain.agents.create_agent``) are tool-calling
agents: the model returns structured ``tool_calls`` and the agent runtime
executes the matching tools. :func:`build_llm` returns one of two backends:

    * **Mock (default, used for recording).** :class:`ScriptedToolCallingChatModel`
      — a deterministic, zero-latency chat model that emits fixed ``tool_calls``
      (and a final answer) per demo scenario. It routes on the user's question
      and tracks progress by counting the tool results already in the message
      history, so it is robust to retries and needs no prompt-string parsing.
    * **OpenAI (optional).** If ``OPENAI_API_KEY`` is set and the mock is not
      forced, returns a :class:`~langchain_openai.ChatOpenAI` (which supports
      tool calling natively). Tool choices are not guaranteed to match the
      scripted scenarios; the recording always uses the mock.

Selection:
    * ``KV_DEMO_MOCK_LLM=1`` (default when unset) forces the mock.
    * Otherwise, if ``OPENAI_API_KEY`` is present, OpenAI is used.
    * With neither, the mock is used so the demo always runs offline.

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable


def _mock_forced() -> bool:
    """Return True when the mock LLM is explicitly requested via env.

    Returns:
        True if ``KV_DEMO_MOCK_LLM`` is a truthy value ("1", "true", "yes").
    """
    return os.getenv("KV_DEMO_MOCK_LLM", "").strip().lower() in {"1", "true", "yes"}


#: A single planned tool call: (tool_name, arguments).
ToolStep = Tuple[str, Dict[str, Any]]

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_INVOICE_RE = re.compile(r"\b(INV[-\s]?\d+|[A-Z]{2,}-\d+)\b", re.IGNORECASE)
_NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5,
    "couple": 2, "several": 3, "multiple": 3, "few": 3,
}
_SEND_WORDS = ("send", "forward", "email", "e-mail", "mail", "notify", "remind")
_HANDBOOK_WORDS = ("policy", "handbook", "reimburs", "allowance", "guideline")


def _extract_email(text: str) -> Optional[str]:
    """Return the first email address in ``text``, or None."""
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else None


def _extract_invoice_id(text: str) -> str:
    """Return the first invoice-like id in ``text`` (default ``INV-1042``)."""
    m = _INVOICE_RE.search(text)
    if not m:
        return "INV-1042"
    return re.sub(r"\s+", "-", m.group(0).upper())


def _has_send_intent(q: str) -> bool:
    """Return True if the (lowercased) request asks to send/forward a message."""
    return any(w in q for w in _SEND_WORDS)


def _has_handbook_intent(q: str) -> bool:
    """Return True if the (lowercased) request asks about the expense handbook."""
    return "expense" in q or any(w in q for w in _HANDBOOK_WORDS)


def _requested_send_count(q: str) -> int:
    """Infer how many emails the (lowercased) request wants (>= 1).

    Recognizes an explicit digit, a number word ("two", "several"), or a plural
    ("reminders", "emails"). Capped at 5 to keep an unpoliced run bounded — with
    the single-send policy assigned the second send is blocked regardless.
    """
    m = re.search(r"\b(\d+)\b", q)
    if m and int(m.group(1)) >= 2:
        return min(int(m.group(1)), 5)
    for word, n in _NUMBER_WORDS.items():
        if word in q:
            return n
    if "reminders" in q or "emails" in q or "messages" in q:
        return 2
    return 1


class ScriptedToolCallingChatModel(BaseChatModel):
    """Deterministic tool-calling chat model that scripts the demo scenarios.

    The model never calls a network. On each invocation it reads the first
    human message (the user's request) to select a scenario, and counts the
    :class:`~langchain_core.messages.ToolMessage` entries already in the
    history to know how many tools have run. It then returns either an
    :class:`~langchain_core.messages.AIMessage` carrying ``tool_calls`` (to run
    the next tool) or a plain ``AIMessage`` (to finish the task).

    Because progress is derived from the message history rather than an internal
    counter, the same run is reproducible even if the agent retries a step.

    Intent routing (keyword/regex over the user's free-text request — see
    :func:`_plan`), in priority order:

        * invoice + send intent → get_invoice, then send_email
          (e.g. "look up invoice INV-9 and forward it to a@b.com"). With the
          policies assigned the financial-path rule blocks the send.
        * invoice only          → get_invoice, then answer.
        * expense/handbook      → search_finance_handbook, then answer.
        * send intent           → one or more send_email calls (count inferred
          from "two"/"2"/"reminders"); the single-send policy blocks the 2nd.
        * greeting / help / else → answer directly, no tool.

    This lets ``main.py`` accept a real typed request rather than fixed scenario
    keys, while staying fully deterministic.
    """

    @property
    def _llm_type(self) -> str:
        """Return the model type identifier.

        Returns:
            The string ``"scripted-tool-calling-mock"``.
        """
        return "scripted-tool-calling-mock"

    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Accept tool binding from the agent runtime.

        The scripted model emits tool calls by name from its own scenario
        logic, so it does not need the bound tool schemas. It simply returns
        itself so ``create_agent`` can treat it as a tool-calling model.

        Args:
            tools: Tools the agent would bind (ignored by the mock).
            **kwargs: Additional binding options (ignored).

        Returns:
            This model instance, unchanged.
        """
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Produce the next scripted AI message for the current scenario.

        Args:
            messages: The running message history (system + human + any
                assistant/tool messages so far).
            stop: Stop sequences (unused).
            run_manager: Optional callback manager (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A :class:`ChatResult` wrapping a single scripted AI message that
            either requests a tool call or gives the final answer.
        """
        message = self._script(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @staticmethod
    def _question(messages: Sequence[BaseMessage]) -> str:
        """Return the user's raw request (first human message).

        Args:
            messages: The message history.

        Returns:
            The first human message content, or "" if none. Case is preserved
            so recipients / invoice ids can be extracted; callers lowercase for
            keyword routing.
        """
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content
                return content if isinstance(content, str) else str(content)
        return ""

    @staticmethod
    def _tools_run(messages: Sequence[BaseMessage]) -> int:
        """Return how many tools have already completed in this task.

        Args:
            messages: The message history.

        Returns:
            The number of :class:`ToolMessage` entries in the history.
        """
        return sum(1 for msg in messages if isinstance(msg, ToolMessage))

    @staticmethod
    def _call(name: str, args: Dict[str, Any], index: int) -> AIMessage:
        """Build an AIMessage that requests a single tool call.

        Args:
            name: Tool name to invoke.
            args: Tool arguments.
            index: Zero-based step index, used to build a unique call id.

        Returns:
            An :class:`AIMessage` carrying exactly one ``tool_call``.
        """
        return AIMessage(
            content="",
            tool_calls=[{
                "name": name,
                "args": args,
                "id": f"call_{index + 1}",
                "type": "tool_call",
            }],
        )

    def _script(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Return the next tool call or final answer for the request.

        Builds the intended tool-call plan from the user's request, then emits
        the step at the current progress index (or the final answer once every
        planned step has run).

        Args:
            messages: The message history.

        Returns:
            An :class:`AIMessage` — either a tool-call request or a final
            answer.
        """
        request = self._question(messages)
        steps, final_answer = self._plan(request)
        done = self._tools_run(messages)
        if done < len(steps):
            name, args = steps[done]
            return self._call(name, args, done)
        return AIMessage(content=final_answer)

    @staticmethod
    def _plan(request: str) -> Tuple[List[ToolStep], str]:
        """Turn a free-text request into a tool-call plan and a final answer.

        Deterministic keyword/regex routing over ``request`` (see the class
        docstring for priority order). The returned steps are executed one per
        model turn; the final answer is only reached if no policy blocks a step.

        Args:
            request: The user's raw request text.

        Returns:
            ``(steps, final_answer)`` where ``steps`` is an ordered list of
            ``(tool_name, args)`` and ``final_answer`` is the closing message.
        """
        q = request.lower()
        recipient = _extract_email(request)

        # Invoice + send intent → read then forward (financial-path block).
        if "invoice" in q and _has_send_intent(q):
            invoice_id = _extract_invoice_id(request)
            return (
                [
                    ("get_invoice", {"invoice_id": invoice_id}),
                    ("send_email", {
                        "to": recipient or "external@example.com",
                        "subject": f"Invoice {invoice_id}",
                        "body": "Payment details attached.",
                    }),
                ],
                f"Forwarded the details of invoice {invoice_id}.",
            )

        # Invoice lookup only.
        if "invoice" in q:
            invoice_id = _extract_invoice_id(request)
            return (
                [("get_invoice", {"invoice_id": invoice_id})],
                f"Invoice {invoice_id}: Northwind Logistics, EUR 48,250.00, "
                "approved for payment.",
            )

        # Expense handbook lookup.
        if _has_handbook_intent(q):
            return (
                [("search_finance_handbook", {"query": request})],
                "Per the handbook, economy airfare and meals within the daily "
                "caps are reimbursable; submit receipts within 30 days.",
            )

        # Send one or more emails (count inferred from the request).
        if _has_send_intent(q):
            to = recipient or "finance@acme.com"
            n = _requested_send_count(q)
            steps: List[ToolStep] = [
                ("send_email", {
                    "to": to,
                    "subject": f"Reminder {i + 1}" if n > 1 else "Message",
                    "body": "Following up on the pending approval.",
                })
                for i in range(n)
            ]
            return steps, f"Sent {n} email(s) to {to}."

        # Greeting / capability question / anything else → answer directly.
        return (
            [],
            "I can look up the expense handbook, fetch invoice details, and "
            "send emails on your behalf. What would you like to do?",
        )


def build_llm() -> BaseChatModel:
    """Build the demo LLM: deterministic mock by default, OpenAI optional.

    Selection order:
        1. If ``KV_DEMO_MOCK_LLM`` is truthy → mock.
        2. Else if ``OPENAI_API_KEY`` is set → ``ChatOpenAI``.
        3. Else → mock (so the demo always runs offline).

    Returns:
        A LangChain chat model ready to drive ``create_agent``.

    Raises:
        ImportError: If OpenAI mode is selected but ``langchain_openai`` is
            not installed.
    """
    openai_key = os.getenv("OPENAI_API_KEY")

    if _mock_forced() or not openai_key:
        return ScriptedToolCallingChatModel()

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(api_key=openai_key, temperature=0.0, model="gpt-4o-mini")
