# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kyvvu B.V.

"""Tool surface for the governed Invoice / Expense assistant demo.

Three modern LangChain structured tools (``@tool``), named so the built-in
Kyvvu LangChain template assigns the correct verb automatically:

    * ``search_finance_handbook`` — ``^search`` → ``step.resource`` GET
      (benign read of the expense handbook).
    * ``get_invoice`` — ``^get`` → ``step.resource`` GET. The demo template
      additionally tags this call ``data.classification: financial``, making
      it the taint source for the ``tainted_path_block`` policy.
    * ``send_email`` — ``^send`` → ``step.resource`` POST (external send;
      the taint target and the count-limit target).

Every body is a pure, side-effect-free stub: it returns a fixed canned
string and never touches the network, a database, or a real mail server.
This keeps the demo fully reproducible and safe to run anywhere.

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""
from __future__ import annotations

from langchain_core.tools import BaseTool, tool


@tool
def search_finance_handbook(query: str) -> str:
    """Look up company expense and travel policy in the finance handbook.

    Benign read tool. Emits ``step.resource`` GET via the ``^search``
    pattern in the template. Never tainted — this is the "normal allowed
    tool" in the demo.

    Args:
        query: Free-text question about company expense policy.

    Returns:
        A fixed handbook excerpt describing the travel-expense policy.
    """
    return (
        "Finance Handbook — Travel & Expenses:\n"
        "- Economy airfare is reimbursable; upgrades are not.\n"
        "- Meals are capped at EUR 50/day domestic, EUR 75/day international.\n"
        "- Submit receipts within 30 days via the Expenses portal.\n"
        "- Any single expense over EUR 1,000 requires manager pre-approval."
    )


@tool
def get_invoice(invoice_id: str) -> str:
    """Look up an invoice by ID, returning its sensitive payment details.

    Emits ``step.resource`` GET via the ``^get`` pattern. The demo template
    tags this call ``data.classification: financial``, so reading an invoice
    "taints" the task: any later external send is then blocked by the
    ``tainted_path_block`` policy.

    Args:
        invoice_id: The invoice identifier to look up (e.g. ``"INV-1042"``).

    Returns:
        A fixed invoice record including amount, vendor, and an IBAN /
        payment detail — the sort of financial data that must not be
        exfiltrated.
    """
    return (
        f"Invoice {invoice_id or 'INV-1042'}:\n"
        "- Vendor: Northwind Logistics B.V.\n"
        "- Amount: EUR 48,250.00\n"
        "- Payment IBAN: NL91ABNA0417164300\n"
        "- Reference: PO-2026-0098\n"
        "- Status: approved for payment"
    )


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (stub: returns a confirmation without sending anything).

    Emits ``step.resource`` POST via the ``^send`` pattern. This is the
    external send that the two security policies guard: it is blocked after
    a financial invoice read (path), and only one such send is allowed per
    task (count).

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body text.

    Returns:
        A fixed confirmation string. No email is actually dispatched.
    """
    return f"Email queued to {to or 'unknown'} with subject '{subject or '(none)'}'."


def build_tools() -> list[BaseTool]:
    """Return the three structured tools for the assistant.

    Returns:
        The tool list, in registration order:
        ``search_finance_handbook``, ``get_invoice``, ``send_email``.
    """
    return [search_finance_handbook, get_invoice, send_email]
