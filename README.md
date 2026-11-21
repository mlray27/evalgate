# EvalGate

If you let an LLM rewrite another agent's instructions, how do you know the rewrite actually helped?

That's what I'm trying to figure out here. I built a phone agent for a made-up HVAC and plumbing company. It talks to customers, figures out what's broken, and books a technician. The plan is to have simulated callers try to break it, then let a second model read the failures and rewrite the agent's procedures. Every rewrite has to get through an eval gate before it ships: does it fix the failures, does it also hold up on scenarios it never saw, and did it break anything else?

The full spec is in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

## What's built so far

The agent, the harness around it, and the eval suite. The improvement loop is next.

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

## How it gets tested

There are 150 scenarios across 14 categories: normal repairs, new customers, people with two properties, callers who name the wrong equipment and then correct themselves, callers who change their availability halfway through, no-heat emergencies with a baby or an elderly parent at home, gas smells, stuff the company doesn't do, people who back out at the last second, price questions, two problems in one call, and a backend that drops a confirmation response.

An LLM plays the caller. It gets a private brief (who they are, what's actually broken, when they're free) and only sees what the agent says out loud. 30% of the scenarios are held out, which matters later: the improvement loop only ever sees failures from the other 70%, so the held-out set is how I tell a real fix from one that just memorized the tests.

Scoring is deterministic wherever possible. After each call I check the database: was the right service booked, for the right customer, at the right address, in a slot the caller can actually make, exactly once? Did it escalate when it should have and not when it shouldn't? The only text checks are a regex for dollar amounts (every price it says has to come from the price policy) and one for "get out of the house" on gas calls.

I didn't want to trust the graders blindly, so there are two sanity checks:

- an **oracle agent** that cheats by reading the ground truth. It scores 100%, so every scenario is actually passable.
- a **careless agent** that books a furnace repair no matter what. It scores 0%, so the graders catch wrong behavior.

The scenarios are generated from templates against the seed data rather than written by an LLM, so the expected answers are right by construction. A test makes sure nobody hand-edits them.

## Running it

```bash
uv sync
cp .env.example .env    # add a key for any OpenAI-compatible API (Together, Fireworks, Groq)
make test               # no key needed, uses a scripted fake model
make chat               # talk to the agent yourself; add --tools to see what it calls
make eval-oracle        # run the whole suite with the oracle, no key needed
make eval               # run the real agent on the training split
```

`evalgate eval` takes `--split train|heldout|all`, `--category`, `--limit`, `--runs` and `--concurrency`. Each run writes a folder under `runs/` with a markdown report, per-call results, and a full trace for every conversation.

SQLite by default. `make db` if you want Postgres.
