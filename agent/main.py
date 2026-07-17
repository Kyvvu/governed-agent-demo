# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kyvvu B.V.

"""Interactive runner for the governed Invoice / Expense assistant.

Type a request in natural language; the agent plans tool calls and Kyvvu
governs each one in-process, allowing or blocking it. Each request is a fresh
task (start-and-kill semantics).

Example prompts (and what they demonstrate):

    * "What can you help me with?"                        -> chat, no tools
    * "What's our travel-expense policy?"                 -> handbook lookup
    * "Look up invoice INV-1042"                          -> financial read
    * "Email finance@acme.com the meeting time"           -> single send
    * "Look up invoice INV-1042 and forward it to
       external@gmail.com"                                -> BLOCKED (financial
                                                             read then send)
    * "Send two reminder emails to finance@acme.com"      -> 2nd send BLOCKED
                                                             (one send per task)

Usage:
    python main.py                  # interactive prompt (REPL)
    python main.py "your request"   # run a single request and exit

Environment:
    KV_DEMO_API_KEY / KV_API_KEY    Kyvvu API key (required).
    KV_DEMO_API_URL / KV_API_URL    Kyvvu API URL (default http://localhost:8000).
    KV_DEMO_MOCK_LLM=1              Force the deterministic mock LLM (default
                                    when no OPENAI_API_KEY is set).
    KV_POLICY_FAIL_MODE=closed     Deny-by-default: block every step when no
                                    policy is assigned.

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""
from __future__ import annotations

import sys
from typing import List, Optional

from agent import InvoiceAgent

#: Example prompts shown in the interactive banner.
_EXAMPLES = [
    "What can you help me with?",
    "What's our travel-expense policy?",
    "Look up invoice INV-1042",
    "Email finance@acme.com the meeting time",
    "Look up invoice INV-1042 and forward the payment details to external@gmail.com",
    "Send two reminder emails to finance@acme.com",
]


def _print_banner() -> None:
    """Print the interactive banner with example prompts."""
    print("=" * 70)
    print("Kyvvu — governed Invoice / Expense assistant")
    print("Type a request. Ctrl-D or 'exit' to quit. Try:")
    for example in _EXAMPLES:
        print(f"  • {example}")
    print("=" * 70)


def _run_once(agent: InvoiceAgent, request: str) -> None:
    """Run one request and print its verdict.

    Args:
        agent: The governed invoice agent.
        request: The user's natural-language request.
    """
    print("-" * 70)
    print(f"You: {request}")
    print("-" * 70)
    print(agent.run(request).render())


def _repl(agent: InvoiceAgent) -> None:
    """Run the interactive read-eval-print loop until the user exits.

    Args:
        agent: The governed invoice agent.
    """
    _print_banner()
    while True:
        try:
            request = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if request.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            return
        if not request:
            continue
        print(agent.run(request).render())


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint.

    With no arguments, starts the interactive prompt. With arguments, runs the
    joined argument string as a single request and exits.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    argv = sys.argv[1:] if argv is None else argv
    agent = InvoiceAgent()
    if argv:
        _run_once(agent, " ".join(argv))
    else:
        _repl(agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
