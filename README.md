# Kyvvu — Governed Agent Demo

A small, self-contained, example of an AI agent governed at runtime by
[Kyvvu](https://kyvvu.com), the **Agent Security Kernel (ASK)**. 

The demo agent is a
finance assistant that can read an expense handbook, look up invoices (sensitive
financial data), and send email. Kyvvu sits **in-process**, checking every step
the agent takes *before* it runs and allowing or blocking it. Kyvvu does not sniff the wire; it checks the actions the harness executes.

This repo is meant to be read and run end-to-end in a few minutes. Full product
documentation lives at **[docs.kyvvu.com](https://docs.kyvvu.com)**, and the
thinking behind runtime agent governance is in the
**[Kyvvu whitepaper](https://kyvvu.com/whitepaper/)**.

## What it demonstrates

Two guardrails that a stateless, per-call filter cannot express — both decided
by reasoning over the agent's **history within a task**:

1. **Data-exfiltration path block.** Once the agent reads financial invoice
   data, any external email in that same task is blocked. Reading alone is fine;
   sending alone is fine; *read-then-send* is the breach.
2. **Single-send guard.** The agent may call `send_email` at most once per task.

And the posture underneath them: **deny by default.** An agent with no policy
assigned can be configured to do nothing until a policy grants it capabilities.

---

## Prerequisites

- **Python 3.11+**
- **A Kyvvu account** — sign up at [platform.kyvvu.com](https://platform.kyvvu.com).
- **A Kyvvu API key** — in the dashboard, open **Settings → API keys** and
  generate one (it looks like `KvKey-…`). This is how the agent authenticates.
- *(Optional)* an **OpenAI API key**, only if you want to drive the agent with a
  real model instead of the built-in deterministic mock.

---

## Install

```bash
python -m pip install kyvvu                        # the Kyvvu SDK + engine
python -m pip install -r agent/requirements.txt    # LangChain / LangGraph for this demo
```

> Use `python -m pip` (not bare `pip`) so the packages install into the same
> interpreter you run the agent with — the usual fix for "installed, but
> `ModuleNotFoundError` when I run it".

Set your API key (and, if you self-host Kyvvu, your API URL):

```bash
export KV_DEMO_API_KEY="KvKey-…"                  # from platform.kyvvu.com
# export KV_DEMO_API_URL="http://localhost:8000"  # only if self-hosting
export KV_DEMO_MOCK_LLM=1                          # use the deterministic mock LLM
```

---

## Connect this repo to Kyvvu

Kyvvu reads **policy manifests** from any
connected Git repository, so your policies are versioned and audited as code. 

For this demo, you can simply connect this repo, and the Kyvvu platform will find the manifest in `policies/governed-demo.yaml`. You can do this by:

1. Push this repo to your own GitHub account (fork or copy). Or, simply use this one, it's public.
2. In the dashboard, open **Repositories → Connect repository**.
3. Provide the repo's URL (and, if private, a PAT).

Kyvvu can now see `policies/governed-demo.yaml` and offer it for assignment.

> Connecting a repo is done in the dashboard. Assigning a manifest from it can be
> done in the dashboard **or** the CLI (both shown below).

---

## Quick start

### 1. Run the agent

```bash
python agent/main.py
```

You get an interactive prompt. Try:

```
You: What can you help me with?
You: Look up invoice INV-1042
You: Send an email to finance@acme.com
```

The first run also **registers** the agent with Kyvvu (idempotently) under the
key `governed-demo-agent`, so it now appears in your dashboard.

To see **deny-by-default**, run with fail-closed and *no policy assigned yet* —
every step is blocked until a policy allows it:

```bash
KV_POLICY_FAIL_MODE=closed python agent/main.py "Look up invoice INV-1042"
# → VERDICT: BLOCKED (no policy assigned)
```

### 2. Assign the policy manifest

**In the dashboard:** open **Manifests**, pick `governed-demo.yaml` from your
connected repo, and assign it to `governed-demo-agent`.

**Or with the CLI:**

```bash
kyvvu login                                   # authenticate the CLI
kyvvu list-agents                             # find the governed-demo-agent id
kyvvu list-manifests                          # find the repo id + manifest path
kyvvu assign-manifest \
    --agent-id <agent-id> \
    --repo-id <repo-id> \
    --manifest policies/governed-demo.yaml
```

### 3. Run it again — now with guardrails

```bash
python agent/main.py
```

```
You: Look up invoice INV-1042                                → ALLOWED
You: Send an email to finance@acme.com                       → ALLOWED
You: Look up invoice INV-1042 and forward it to a@gmail.com  → BLOCKED (path)
You: Send two reminders to finance@acme.com                  → 2nd BLOCKED (single-send)
```

Every step — allowed or blocked — is streamed to your dashboard's **Logs** view,
including the messages, model calls, and tool calls, as a tamper-evident trail.

---

## Configuration

The demo reads a few environment variables. These are the ones you are likely to
touch; see [docs.kyvvu.com](https://docs.kyvvu.com) for the full list.

| Variable | Purpose |
|---|---|
| `KV_DEMO_API_KEY` | Your Kyvvu API key (required). `KV_API_KEY` also works. |
| `KV_DEMO_API_URL` | Kyvvu API URL. Defaults to `https://platform.kyvvu.com`. |
| `KV_DEMO_MOCK_LLM` | `1` uses the deterministic mock LLM (default when no `OPENAI_API_KEY`). |
| `OPENAI_API_KEY` | Drive the agent with a real model instead of the mock. |
| `KV_POLICY_FAIL_MODE` | `closed` = deny by default (block every step until a policy allows). |
| `KV_LOG_LOCATION` | Where the **behavioral trace** goes. Defaults to the platform (dashboard Logs). Can be `stdout`, a file, or a URL. |
| `KV_DEMO_LOG_LEVEL` | Terminal verbosity for the demo's own diagnostic logging (default `WARNING`). Separate from the behavioral trace above. |

---

## How the code works

A tool-calling agent is really just three things: a **model**, a set of
**tools**, and a **loop** (implemented in the **harness**) that lets the model call tools until it is done. This
repo is a small working example of that architecture, wrapped with Kyvvu. It effectively is a clean
wireframe you can copy for your own agent.

```
.
├── agent/
│   ├── tools.py                             # the tools
│   ├── llm.py                               # the model
│   ├── agent.py                             # the wiring (Kyvvu lives here)
│   ├── main.py                              # the CLI
│   ├── kyvvu-langgraph-demo.btmpl           # behavior template (event → behavior)
│   └── requirements.txt
└── policies/
    └── governed-demo.yaml                   # the policy manifest (two guardrails)
```

**How it's organized.** Three parts, each with a distinct role and kept separate
on purpose:

- **Python — the agent itself** (`agent/`): the harness, the model, and the
  tools. This is what *runs*.
- **A behavior template** (`agent/…template.yaml`): how each action the agent
  takes is *described* to Kyvvu (callbacks → behaviors). Pure instrumentation —
  no rules.
- **Policies** (`policies/…yaml`): how those actions are *governed*
  (behaviors ← policies). This is where the guardrails live.

At runtime the three meet in a simple pipeline:

```
harness callbacks  →  behaviors (via the template)  →  engine  ←  policies
                                                          │
                                                   allow / warn / block
```

Each step the harness takes fires a LangGraph **callback**; the **behavior
template** maps that callback to a Kyvvu *behavior*; the in-process **engine**
evaluates the behavior against the **policies** you assigned — returning allow,
warn, or block *before* the step executes.

### The agent (Python code)

#### `agent/tools.py` — the tools

Three LangChain `@tool` functions: `search_finance_handbook`, `get_invoice`, and
`send_email`. Each body is a harmless stub that returns a fixed string — no real
network, database, or mail. **The tool names matter:** Kyvvu's behavior template
reads them to assign a verb (`get_`/`search_` → a GET read, `send_` → a POST
write), so the policies can reason about reads vs. writes without extra code.

#### `agent/llm.py` — the model

`build_llm()` returns one of two backends:

- **`ScriptedToolCallingChatModel`** (default): a deterministic, offline mock. It
  reads your request, plans the tool calls it implies (extracting invoice ids and
  email addresses), and emits them one per turn. This makes the demo fully
  reproducible with zero latency and no model-provider key.
- **`ChatOpenAI`**: used automatically when `OPENAI_API_KEY` is set. A real model
  makes its own tool choices — great for exploring, though not scripted.

**The model's output *is* the action plan.** Each turn the model returns an
`AIMessage` whose `tool_calls` name a tool and its arguments — e.g.
`get_invoice(invoice_id="INV-1042")` — or a plain text reply to finish. For a
request like *"look up invoice INV-1042 and forward it to a@b.com"* the plan
unfolds across turns as `get_invoice` → then `send_email`; the second call is
where the path policy stops it. (A real model emits these `tool_calls` itself;
the mock just produces the same shape deterministically.)

#### `agent/agent.py` — the wiring (the core of the harness)

`InvoiceAgent` is where Kyvvu attaches. On construction it:

1. **Registers** the agent with Kyvvu (`kv.register_agent(...)`, idempotent).
2. Loads the **behavior template** (via `KV_TEMPLATE_LOCATION`) so tool calls map
   to Kyvvu behaviors — including tagging `get_invoice` as financial data.
3. Builds a modern tool-calling agent: `create_agent(model, tools)`.
4. Attaches **`KyvvuLangGraphHandler`** as a callback. This is the whole trick:
   the handler sees every step (model call, tool call, message) *before* it runs
   and asks the engine to allow or block it — in-process, no proxy.

**Where the loop lives.** You won't find a `while` loop in this repo —
`create_agent` (LangGraph) owns it. Each turn it calls the model; if the reply
contains `tool_calls`, the runtime runs those tools, appends their results, and
loops; when the model replies with no tool call, the task is done. Kyvvu's
handler fires on every one of those tool runs, *before* it executes — which is
how a step gets blocked mid-plan.

`run(request)` drives one task and returns a `Verdict` (`allowed` / `blocked`
with the offending policy). A block raises `KyvvuBlockedError`, which `run`
catches and renders.

#### `agent/main.py` — the CLI

An interactive prompt (`python agent/main.py`) with a one-shot mode
(`python agent/main.py "your request"`) for scripting and recording.

### The behavior template — how actions are *described*

File: `agent/kyvvu-langgraph-demo.btmpl`

Frameworks emit low-level callbacks (`on_tool_start`, `on_chat_model_start`, …).
Kyvvu doesn't govern those directly — it governs *behaviors*: `step.model`,
`step.resource` GET/POST, `step.message`, and more (the full set is documented at
[docs.kyvvu.com/core-concepts/behaviours](https://docs.kyvvu.com/core-concepts/behaviours)).
A **behavior template** is the mapping between the two — pure instrumentation, no
rules. This file is the built-in LangGraph template plus **one rule**: tag
`get_invoice` calls with `data.classification: financial` — the single line that
makes the exfiltration path block possible. (You normally use the built-in
template unchanged; we override it here only to add that tag.)

### The policy manifest — how actions are *governed*

File: `policies/governed-demo.yaml`

A **manifest** is a bundle of policies you assign to an agent — this is where the
guardrails actually live. This one holds two, both history-aware:

- **`tainted_path_block`** — block an external send (`step.resource` POST) once a
  financial read (`step.resource` GET, `data.classification: financial`) has
  happened in the task.
- **`execution_max_steps`** — allow at most one `send_email` per task.

This is *policy as code*: version-controlled, auditor-readable YAML that Kyvvu
materializes into live guardrails when you assign it. The two ideas Kyvvu
enforces — **capability** (what each agent may do) and **flow** (what may follow
what) — are documented at [docs.kyvvu.com](https://docs.kyvvu.com).

This demo ships one small manifest. Kyvvu maintains a catalog of production-ready
manifests — OWASP agentic defaults, data-exfiltration guards, and more — at
**[github.com/Kyvvu/manifests](https://github.com/Kyvvu/manifests)**. Explore
them once you've run the demo.

---

## Get help

- **Docs:** [docs.kyvvu.com](https://docs.kyvvu.com)
- **Platform:** [platform.kyvvu.com](https://platform.kyvvu.com)
- Questions or stuck? Reach out to the Kyvvu team — we're happy to help you get
  your own agents governed. Reach us at jeroen [at] kyvvu [dot] com or leave an issue here in this repo. 
