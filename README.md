# EvalGate

If you let an LLM rewrite another agent's instructions, how do you know the rewrite actually helped?

That's what I'm trying to figure out here. I built a phone agent for a made-up HVAC and plumbing company. It talks to customers, figures out what's broken, and books a technician. The plan is to have simulated callers try to break it, then let a second model read the failures and rewrite the agent's procedures. Every rewrite has to get through an eval gate before it ships: does it fix the failures, does it also hold up on scenarios it never saw, and did it break anything else?

The full spec is in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

## What's built so far

Right now it's the agent and the harness around it. Simulated callers and the improvement loop are next.

The agent is one model with the whole conversation and ~25 company procedures in context. No router, no handoffs between sub-agents. I went this way because every handoff loses context, and callers change their minds constantly ("wait, it's the water heater, not the furnace").

The part I care most about is that the model can't directly do anything irreversible. To book a visit it has to:

1. call `propose_booking`, which checks everything and returns a token, but writes nothing
2. read the details back to the caller
3. wait for the caller to reply, then call `confirm_booking(token)`

Confirm only takes the token, so what gets booked is exactly what the caller heard. The tests cover the cases I'd worry about in production:

- confirming before the caller has said yes gets blocked
- a confirm that succeeds but times out, then gets retried, still makes one booking
- two calls racing for the same slot can't both get it
- a model stuck in a tool-calling loop hands off to a human instead of spinning
- interrupting the agent mid-sentence doesn't corrupt the conversation

## Running it

```bash
uv sync
cp .env.example .env    # add a key for any OpenAI-compatible API (Together, Fireworks, Groq)
make test               # no key needed, uses a scripted fake model
make chat               # talk to the agent yourself; add --tools to see what it calls
```

SQLite by default. `make db` if you want Postgres.
