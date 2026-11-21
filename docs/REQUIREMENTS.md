# EvalGate — Requirements

**Research question:** When can an AI that rewrites another agent's operating procedures be trusted?

EvalGate is a small, reproducible agent platform for home-services call handling (HVAC/plumbing).
A single service agent handles conversations, a deterministic harness guards its actions, simulated
callers stress-test it, an Agent Manager proposes procedure fixes, and a trust evaluation measures
whether those fixes generalize or just game the tests.

---

## 1. Design principles

1. **One agent per conversation.** Full history + relevant procedures in context. No handoffs.
2. **Constrain outcomes, not paths.** The conversation is free-form; determinism lives at the tool
   boundary and on irreversible actions.
3. **The LLM proposes, the harness disposes.** No tool with side effects runs without passing guard checks.
4. **Instructions over primitives.** Business policy is SOP text, not code paths.
5. **Everything is replayable.** Every conversation produces a trace that can be re-scored and replayed.
6. **The improver never grades itself.** The Agent Manager cannot edit scenarios, graders, or the trust gate.

## 2. Components

| # | Component | Responsibility |
|---|-----------|----------------|
| C1 | Agent runtime | Async conversation loop, streaming output, tool calling, SOP loading |
| C2 | Action guard | Validate/authorize tool calls; confirmation, idempotency, transactional writes |
| C3 | Business backend | Mock CRM + scheduling DB (customers, technicians, slots, appointments) |
| C4 | Caller simulator | LLM-played callers with hidden ground truth and behavior traits |
| C5 | Scenario suite + graders | ~150 scenarios, train/held-out split, deterministic + LLM-judge scoring |
| C6 | Agent Manager | Cluster failures, propose SOP edits, create immutable agent versions |
| C7 | Trust evaluation | Generalization gap, regressions, reward-hacking detection, trust gate |
| C8 | Trace viewer (stretch) | Minimal React UI over traces and version comparisons |

## 3. Functional requirements

### C1 Agent runtime
- FR1.1 Each session holds: full message history, loaded SOP ids, tool-call log, agent version id.
- FR1.2 Responses stream token-by-token over WebSocket (`/ws/session`) and a CLI.
- FR1.3 Uses an OpenAI-compatible chat API (configurable `base_url`, `model`) so any hosted
  open-weight model (Qwen, GLM, Llama, Kimi) works. A scripted fake LLM is used in tests.
- FR1.4 SOP library: ~50 procedures as Markdown with front-matter (id, title, triggers, priority).
  Baseline mode loads all SOPs into the system prompt; retrieval modes are a later experiment.
- FR1.5 Max tool-call steps per turn (default 6) to prevent loops; on hitting the limit the agent
  must escalate, not keep calling.
- FR1.6 Per-tool timeouts and bounded retries for read-only tools.
- FR1.7 (Milestone 2) Speculative execution: read-only tools may start before the user turn is
  complete and are cancelled (`asyncio` cancellation) if the turn changes. Never for side-effect tools.

### C2 Action guard
- FR2.1 All tool arguments validated against Pydantic schemas; invalid calls return a structured
  error to the model rather than raising.
- FR2.2 Tools are classified `read_only` or `side_effect`.
- FR2.3 Side-effect tools (`book_appointment`, `cancel_appointment`, `reschedule_appointment`)
  require a **confirmation token** from `propose_booking`/`propose_change`. The guard checks:
  - the token exists, belongs to this session, and is not expired;
  - at least one caller turn occurred after the proposal was read back;
  - the token's details are unchanged since read-back;
  - the slot still exists and is free (checked inside the write transaction);
  - the customer exists.
- FR2.4 Idempotency: each side-effect call carries an idempotency key derived from
  (session, confirmation token). A repeated call returns the original result and creates nothing.
- FR2.5 Every guard decision (allow / deny + reason) is written to the trace.
- FR2.6 Consent withdrawal: once the caller declines a proposal, its token is revoked.

### C3 Business backend
- FR3.1 Entities: Customer, Address, Technician (skills: hvac, plumbing, electrical), Slot, Appointment,
  IdempotencyRecord.
- FR3.2 Postgres in docker-compose for local runs; SQLite for tests. Same SQLAlchemy models.
- FR3.3 Deterministic seed data (fixed RNG seed) so every eval run starts from identical state.
- FR3.4 DB-level constraints: one active appointment per slot; unique idempotency key.

### C4 Caller simulator
- FR4.1 A scenario defines: persona, hidden ground truth (true issue, address, availability,
  willingness to book), behavior traits, and an opening line.
- FR4.2 Traits include: gradual reveal, mid-call correction, interruption, off-topic question,
  changes availability, withdraws consent, gives wrong info then corrects, emergency.
- FR4.3 The simulator only knows what the scenario gives it; it never sees agent internals.
- FR4.4 Conversations end on: booking done, caller hangs up, escalation, or turn limit.

### C5 Scenarios and graders
- FR5.1 ~150 scenarios, tagged by trait and failure category; fixed **train / held-out** split
  (held-out is never shown to the Agent Manager).
- FR5.2 Deterministic graders over final DB state + trace: correct service, correct address,
  slot matches availability, confirmation obtained, no duplicate side effects, no forbidden tools,
  escalated when required, no booking after consent withdrawn.
- FR5.3 LLM-judge graders only for soft qualities (clarity, repetition). Reported separately.
- FR5.4 Each scenario runs N times (default 3) to measure variance; report mean and pass^k.

### C6 Agent Manager
- FR6.1 Input: failed traces from the **train** split only.
- FR6.2 Clusters failures by grader + tag, picks the top cluster, and asks an LLM to propose a
  minimal SOP edit (diff) with a rationale.
- FR6.3 Each accepted proposal creates a new immutable `AgentVersion` (SOP snapshot + prompt + model).
- FR6.4 The Manager has no write access to scenarios, graders, or the trust gate.

### C7 Trust evaluation
- FR7.1 For every proposal, run baseline and candidate on train + held-out.
- FR7.2 Report: train delta, held-out delta, **generalization gap** (train delta − held-out delta),
  regressions (previously passing scenarios that now fail), cost and latency deltas.
- FR7.3 **Reward-hacking probe:** an experiment where the Manager is optimized against a planted,
  exploitable LLM judge; measure whether judge score rises while deterministic scores do not.
- FR7.4 **Trust gate:** a rule (thresholds on held-out delta, regressions, variance) that decides
  auto-promote / needs-human / reject. Evaluate its precision against ground truth from the full suite.

## 4. Non-functional requirements
- NFR1 `make test` passes with no API keys (fake LLM).
- NFR2 Reproducible: fixed seeds, pinned model name, versioned scenarios, results stored as JSONL.
- NFR3 Latency recorded per turn: time to first token, total turn time, tool time.
- NFR4 Cost recorded per conversation (tokens × configured price).
- NFR5 Python 3.12+, asyncio throughout, type-checked, ruff-formatted, CI on GitHub Actions.

## 5. Metrics

| Metric | Definition |
|--------|------------|
| Task success | All deterministic graders pass for a scenario |
| Booking accuracy | Correct service + address + valid slot, when booking was the right outcome |
| Correction recovery | Success rate on scenarios with mid-call corrections |
| Policy violation rate | Fraction of conversations with ≥1 SOP/guard violation |
| Duplicate-action rate | Side effects executed more than once per intent |
| Guard deny rate | Denied side-effect calls / attempted |
| pass^k | Probability all k runs of a scenario succeed |
| TTFT p50/p95 | Time to first token per agent turn |
| Cost per success | Total cost / successful conversations |
| Generalization gap | Train improvement − held-out improvement for a proposal |
| Gate precision | Of auto-promoted proposals, fraction that truly improve the full suite without regressions |

## 6. Milestones

1. **M1 — Runtime + guard (week 1):** data layer, seed, SOPs, tools, guard, agent loop, WebSocket + CLI,
   tests including the duplicate-booking-after-timeout test.
2. **M2 — Simulator + evals (week 1–2):** caller simulator, 150 scenarios, graders, runner, baseline report.
   Speculative execution.
3. **M3 — Agent Manager + trust (week 2–3):** failure clustering, proposals, versioning, trust metrics,
   reward-hacking probe, trust gate, write-up.
4. **Stretch:** trace viewer, context-provisioning experiment (all-in-context vs retrieved vs prefetch
   SOPs: accuracy vs TTFT), streaming voice.

## 7. Out of scope
Real telephony, real customer data, payments, multi-tenant auth, production deployment.
