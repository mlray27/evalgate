# EvalGate

If you let an LLM rewrite another agent's instructions, how do you know the rewrite actually helped?

That's what I'm trying to figure out here. I built a phone agent for a made-up HVAC and plumbing company. It talks to customers, figures out what's broken, and books a technician. The plan is to have simulated callers try to break it, then let a second model read the failures and rewrite the agent's procedures. Every rewrite has to get through an eval gate before it ships: does it fix the failures, does it also hold up on scenarios it never saw, and did it break anything else?

The full spec is in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

## What's built so far

The agent, the harness around it, the eval suite, and the improvement loop (an Agent Manager plus a trust gate). What's still missing is real numbers from running the loop end to end; that's in progress on free-tier APIs, so it's slow.

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

Outcomes aren't enough on their own, though. In the first baseline the agent hit the right booking 96% of the time, but along the way it looked up phone numbers nobody had said, tried to create accounts with a made-up city and zip, and told callers "I have that scheduled for you" before anything was booked. The database happened to absorb those, so they looked like passes. So there are also three process checks, computed from the trace, that count toward pass/fail: no tool call uses a phone number the caller never said, no account gets details the caller never gave, and the agent never claims a booking before it actually went through. With those, the same baseline scores 39%, which is a much more honest starting point. `evalgate rescore runs/<folder>` reapplies the trace-based checks to an old run without any API calls.

The simulated caller can mess up too (once it agreed to a date outside its own availability), so runs where the caller visibly went off-script get flagged, and the report shows the success rate with and without them.

I didn't want to trust the graders blindly, so there are two sanity checks:

- an **oracle agent** that cheats by reading the ground truth. It scores 100%, so every scenario is actually passable.
- a **careless agent** that books a furnace repair no matter what. It scores 0%, so the graders catch wrong behavior.

The scenarios are generated from templates against the seed data rather than written by an LLM, so the expected answers are right by construction. A test makes sure nobody hand-edits them.

## The improvement loop

This is the part the whole project is for.

1. **The Agent Manager** reads the failures from a run, groups them by which check failed, and takes the most common one. It gets the conversations, the tool calls, and a plain-English description of the check. It does not get the scenario briefs (the ground truth), the held-out results, or the grader code. Basically, what a production monitoring system would actually have.
2. It proposes exactly one change: edit one procedure or add one. That becomes a new, immutable agent version with a pointer to its parent. The live version never gets edited in place.
3. Before anything is tested, the proposal gets checked. Bad edits (wrong id, empty, no change) are thrown out. Proposals that mention names, phone numbers or addresses from the failing calls get flagged, because that's the Manager memorizing examples instead of fixing a behavior. So do proposals that talk about tests or evaluation.
4. The candidate runs on the same scenarios as the baseline, and **the trust gate** compares them pair by pair. It cares most about the held-out split, since the Manager never saw those callers. It promotes only if held-out success clearly goes up, almost nothing that used to pass now fails, train and held-out improve by similar amounts, and there are no flags. It rejects anything that makes held-out worse or breaks a safety scenario (gas leaks, no-heat emergencies). Everything in between goes to a human. Every report includes a sign test, because with this many calls a few points of difference can easily be noise.

The Manager can only write procedure text. It can't touch the scenarios, the graders, or the gate, and it can't approve its own change.

There's an end-to-end test of the whole loop with scripted models: an agent with one bad habit (saying "I have that scheduled" before booking), a Manager that proposes the fix, and a gate that promotes it. Plus a version where the Manager's fix names a specific customer, which gets flagged and sent to a human instead.

```bash
uv run evalgate eval --split all --per-category 2                 # baseline, train + held-out
uv run evalgate propose runs/<baseline>                           # Manager proposes a fix
uv run evalgate eval --split all --per-category 2 --version versions/<candidate>.json
uv run evalgate compare runs/<baseline> runs/<candidate>          # trust gate decision
```

## Running it

```bash
uv sync
cp .env.example .env              # Windows: copy .env.example .env
uv run evalgate check             # tests your keys, model ids and tool calling
uv run pytest -q                  # no key needed, uses a scripted fake model
uv run evalgate chat --tools      # talk to the agent yourself and watch its tool calls
uv run evalgate eval --oracle --split all    # sanity check the suite, no key needed
uv run evalgate eval --per-category 2        # 28 conversations, one or two per category
uv run evalgate eval --split train           # the full training split
```

`eval` also takes `--category`, `--limit`, `--runs` and `--concurrency`. Each run writes a folder under `runs/` with a markdown report, per-call results, and a full trace of every conversation.

### Doing it for free

I don't have a GPU or paid credits, so `.env.example` defaults to free tiers: Gemini for the agent and Groq for the simulated callers (no credit card for either). The catch is daily quotas, roughly 20 conversations a day with this setup. When a quota runs out, the run stops cleanly and prints the exact command to pick up where it left off the next day (`--resume runs/<folder>`), so the report fills in over a few days instead of crashing.

The default model on a paid provider is an open-weight Qwen model on Together; swap the commented lines in `.env.example`.

The Makefile has shortcuts (`make test`, `make chat`, `make eval`) if you have `make`.
