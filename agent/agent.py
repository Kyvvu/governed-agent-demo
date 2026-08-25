# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kyvvu B.V.

"""Governed Invoice / Expense assistant — orchestration and Kyvvu wiring.

:class:`InvoiceAgent` is a modern LangChain tool-calling agent
(``langchain.agents.create_agent``, LangGraph under the hood) for a finance
team. It can look up the expense handbook, look up invoices (which carry
sensitive financial data), and send email. Kyvvu governs it in-process via
:class:`~kyvvu.integrations.langgraph.KyvvuLangGraphHandler`: policy blocks
raise :class:`~kyvvu.exceptions.KyvvuBlockedError`, which this module catches
at the run boundary and renders as a clean verdict.

The agent registers idempotently (cached on disk per ``agent_key``), so running
the process repeatedly re-registers without error and re-fetches policies on
each launch.

Example::

    agent = InvoiceAgent()
    verdict = agent.run("Look up invoice INV-1042")
    print(verdict.decision)  # "allowed"

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from kyvvu import Kyvvu
from kyvvu.exceptions import KyvvuBlockedError
from kyvvu.integrations.langgraph import KyvvuLangGraphHandler
from kyvvu.schemas import Environment, RiskClassification
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from llm import build_llm
from tools import build_tools

MODULE_VERSION = "2.0.0"

#: agent_key the demo policies must be attached to.
AGENT_KEY = "governed-demo-agent"

#: Path to the demo behavior template shipped alongside this module.
# Behavior template (YAML content). The `.btmpl` extension keeps it out of the
# platform's manifest scanner (which only picks up .yaml/.yml) so it isn't
# mistaken for an invalid policy manifest; the SDK loads it by explicit path.
TEMPLATE_PATH = Path(__file__).resolve().parent / "kyvvu-langgraph-demo.btmpl"

#: System prompt for the tool-calling agent.
_SYSTEM_PROMPT = (
    "You are an internal finance assistant. Use the available tools to look up "
    "the expense handbook, look up invoices, and send email on the user's "
    "behalf. Call a tool when it is needed, then give a short final answer."
)


def _route_trace_to_platform(api_url: str) -> None:
    """Point the behavioral trace at the local platform.

    ``KV_LOG_LOCATION`` controls where the **behavioral trace** goes — the audit
    of the steps the agent takes (task.start, step.model, step.resource, …).
    Default it to the platform API so the trace shows up in the dashboard Logs
    view instead of dumping to stdout. Overridable: an operator who sets
    ``KV_LOG_LOCATION`` (to ``stdout``, a file, or another URL) keeps it.

    This is distinct from :func:`_quiet_demo_logging` (Python diagnostic logs).

    Args:
        api_url: The resolved Kyvvu API base URL to send the trace to.
    """
    os.environ.setdefault("KV_LOG_LOCATION", api_url)


def _quiet_demo_logging() -> None:
    """Quiet Python diagnostic logging for a clean demo terminal.

    This is standard Python logging verbosity (INFO lines like "task … ended"
    and "batch accepted") — DISTINCT from the behavioral trace, which still
    streams to the platform via :func:`_route_trace_to_platform`. The demo
    defaults the ``kyvvu`` / ``kyvvu_engine`` loggers (and LangChain's callback
    logger, which WARNs whenever our handler raises to block a step) to WARNING.

    Override with ``KV_DEMO_LOG_LEVEL`` (e.g. ``DEBUG``). This intentionally does
    NOT read ``KV_LOG_LEVEL``, so the demo terminal stays quiet even when the
    repo ``.env`` sets ``KV_LOG_LEVEL=DEBUG`` globally. Call it after the
    :class:`~kyvvu.Kyvvu` client is built, since constructing it applies
    ``KV_LOG_LEVEL`` to those loggers.
    """
    level = os.environ.get("KV_DEMO_LOG_LEVEL", "WARNING").upper()
    for name in ("kyvvu", "kyvvu_engine"):
        logging.getLogger(name).setLevel(level)
    logging.getLogger("langchain_core.callbacks.manager").setLevel(logging.ERROR)


@dataclass
class Verdict:
    """Outcome of a single agent run, rendered for the demo screen.

    Attributes:
        decision: ``"allowed"`` if the task completed, ``"blocked"`` if a
            policy blocked a step.
        answer: The agent's final answer (only set when allowed).
        enforcement_point: Policy enforcement point of the block (e.g.
            ``"step_execution"``).
        step_type: Step type that was blocked (e.g. ``"step.resource"``).
        verb: Verb of the blocked step (e.g. ``"POST"``).
        tool_name: Name of the blocked step/tool, when known.
        policy_name: Name of the blocking policy.
        risk_score: Aggregate risk score reported by the engine.
    """

    decision: str
    answer: str | None = None
    enforcement_point: str | None = None
    step_type: str | None = None
    verb: str | None = None
    tool_name: str | None = None
    policy_name: str | None = None
    risk_score: float | None = None

    def render(self) -> str:
        """Render a human-readable, multi-line verdict for the terminal.

        Returns:
            A formatted string suitable for pausing on during a recording.
        """
        lines: list[str] = []
        if self.decision == "allowed":
            lines.append("VERDICT: ALLOWED")
            if self.answer:
                lines.append(f"  answer      : {self.answer}")
        else:
            lines.append("VERDICT: BLOCKED")
            lines.append(f"  enforcement : {self.enforcement_point}")
            lines.append(f"  step        : ({self.step_type}, {self.verb})")
            lines.append(f"  tool        : {self.tool_name}")
            lines.append(f"  policy      : {self.policy_name}")
            if self.risk_score is not None:
                lines.append(f"  risk_score  : {self.risk_score:.2f}")
        return "\n".join(lines)


class InvoiceAgent:
    """Kyvvu-governed tool-calling agent for invoice / expense tasks.

    Registers with Kyvvu on construction (idempotent across restarts), loads
    the demo behavior template so ``get_invoice`` is classified as financial
    data, builds a modern ``create_agent`` over the three demo tools, and wires
    the Kyvvu LangGraph callback handler so policy blocks raise and are
    rendered as a clean verdict.

    Attributes:
        agent_key: Kyvvu agent key (``"governed-demo-agent"``).
        handler: The Kyvvu LangGraph callback handler.
        executor: The compiled LangGraph agent from ``create_agent``.
    """

    def __init__(
        self,
        kyvvu_api_key: str | None = None,
        kyvvu_api_url: str | None = None,
        environment: str = Environment.DEVELOPMENT,
        risk_classification: str = RiskClassification.HIGH,
    ) -> None:
        """Initialize and register the governed invoice agent.

        Args:
            kyvvu_api_key: Kyvvu API key. Falls back to ``KV_DEMO_API_KEY``
                then ``KV_API_KEY``.
            kyvvu_api_url: Kyvvu API base URL. Falls back to
                ``KV_DEMO_API_URL`` then ``KV_API_URL`` then
                ``http://localhost:8000``.
            environment: Deployment environment string.
            risk_classification: EU AI Act risk tier. High by default, so the
                manifest's critical policies always apply.

        Raises:
            ValueError: If no Kyvvu API key can be resolved.
        """
        api_key = kyvvu_api_key or os.getenv("KV_DEMO_API_KEY") or os.getenv("KV_API_KEY")
        if not api_key:
            raise ValueError(
                "Kyvvu API key required. Set KV_DEMO_API_KEY (or KV_API_KEY) "
                "or pass kyvvu_api_key."
            )
        # Defaults to the hosted platform; set KV_DEMO_API_URL (or KV_API_URL)
        # to http://localhost:8000 when running against a self-hosted stack.
        api_url = (
            kyvvu_api_url
            or os.getenv("KV_DEMO_API_URL")
            or os.getenv("KV_API_URL")
            or "https://platform.kyvvu.com"
        )

        # The handler resolves the template at construction time, so point
        # KV_TEMPLATE_LOCATION at the demo template first. This is what attaches
        # data.classification: financial to get_invoice.
        os.environ["KV_TEMPLATE_LOCATION"] = str(TEMPLATE_PATH)

        # Behavioral trace -> local platform (dashboard Logs view). Must be set
        # before the client, which reads KV_LOG_LOCATION when it builds exporters.
        _route_trace_to_platform(api_url)

        self.agent_key: str = AGENT_KEY

        # Register the agent (idempotent — cached on disk per agent_key).
        kv = Kyvvu(
            api_key=api_key,
            api_url=api_url,
            agent_key=self.agent_key,
            environment=environment,
            risk_classification=risk_classification,
        )
        # Quiet Kyvvu's Python diagnostic logging (INFO chatter) now that the
        # client has applied KV_LOG_LEVEL — the trace above still goes to the
        # platform. Distinct concern from the trace sink.
        _quiet_demo_logging()
        kv.register_agent(
            name="Invoice / Expense Assistant",
            purpose=(
                "Finance assistant that reads the expense handbook and invoices "
                "and sends email, governed by Kyvvu."
            ),
            declared_tools=["search_finance_handbook", "get_invoice", "send_email"],
            metadata={
                "framework": "langgraph",
                "agent_type": "tool_calling",
                "demo": "governed-video-demo",
            },
        )

        # Raise-on-block is the handler default (raise_error=True), so a policy
        # block propagates KyvvuBlockedError out of the tool invocation.
        self.handler = KyvvuLangGraphHandler(kv)

        # Modern tool-calling agent (LangGraph under the hood).
        self.executor = create_agent(
            build_llm(),
            build_tools(),
            system_prompt=_SYSTEM_PROMPT,
        )

    def run(self, request: str) -> Verdict:
        """Run one governed task and return a rendered verdict.

        Each call is a fresh task (start-and-kill semantics). If a policy blocks
        a step, the resulting :class:`KyvvuBlockedError` is caught and converted
        into a ``blocked`` :class:`Verdict`.

        Args:
            request: The natural-language user request.

        Returns:
            A :class:`Verdict` describing the outcome (allowed or blocked).
        """
        try:
            result = self.executor.invoke(
                {"messages": [HumanMessage(content=request)]},
                config={"callbacks": [self.handler]},
            )
            answer = self._final_answer(result)
            return Verdict(decision="allowed", answer=answer)
        except KyvvuBlockedError as exc:
            return self._verdict_from_block(exc)

    @staticmethod
    def _final_answer(result: dict) -> str | None:
        """Extract the agent's final answer text from the graph result.

        Args:
            result: The dict returned by ``create_agent(...).invoke``.

        Returns:
            The last message's text content, or None if unavailable.
        """
        messages = result.get("messages") if isinstance(result, dict) else None
        if not messages:
            return None
        content = getattr(messages[-1], "content", None)
        if isinstance(content, str):
            return content.strip() or None
        return str(content) if content else None

    @staticmethod
    def _verdict_from_block(exc: KyvvuBlockedError) -> Verdict:
        """Build a ``blocked`` verdict from a :class:`KyvvuBlockedError`.

        The engine's :class:`~kyvvu_engine.schemas.EvalResult` reliably carries
        ``risk_score`` and the per-policy outcomes (name, rule_type, params); it
        does not carry the intended behavior, so the blocked ``(step_type,
        verb)`` is derived from the violated policy's parameters where
        available. The blocked tool name comes from ``exc.step_name``.

        Args:
            exc: The block exception raised by the handler.

        Returns:
            A ``blocked`` :class:`Verdict`.
        """
        result = getattr(exc, "result", None)
        violated = [
            p for p in getattr(result, "policies", []) if getattr(p, "violated", False)
        ]
        risk_score = getattr(result, "risk_score", None)

        policy = violated[0] if violated else None
        policy_name = policy.name if policy else "policy violation"

        # Derive the blocked (step_type, verb) from the violated policy's
        # parameters. tainted_path_block uses target_step_types/target_verb;
        # execution_max_steps uses step_type/verb.
        step_type = verb = None
        details = getattr(policy, "violation_details", None) or {}
        rule_params = details.get("params") if isinstance(details, dict) else None
        source = rule_params if isinstance(rule_params, dict) else {}
        if source:
            targets = source.get("target_step_types") or source.get("step_type")
            step_type = targets[0] if isinstance(targets, list) else targets
            verb = source.get("target_verb") or source.get("verb")

        return Verdict(
            decision="blocked",
            enforcement_point="step_execution",
            step_type=str(step_type) if step_type else "step.resource",
            verb=str(verb) if verb else "POST",
            tool_name=exc.step_name or None,
            policy_name=policy_name,
            risk_score=risk_score,
        )
