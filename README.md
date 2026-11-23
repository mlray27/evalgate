# EvalGate

If you let an LLM rewrite another agent's instructions, how do you know the rewrite actually helped?

That's the question I wanted to poke at. So I built a phone agent for a made-up HVAC and plumbing company, a bunch of fake callers to stress it, and a loop where a second model tries to fix the agent's procedures. Every fix has to get through an eval gate before it ships.

## How I got here

**One agent, not a graph.** Callers change their minds constantly ("wait, it's the water heater, not the furnace"). Every handoff between specialist agents loses context, so I went with one model that sees the whole call plus ~25 company procedures as plain text.

**The model doesn't get to book things directly.** It calls `propose_booking`, reads the details back, and only after the caller says yes can it call `confirm_booking(token)`. Confirm only takes the token, so what gets booked is exactly what the caller heard. Tests cover the stuff I'd worry about in prod: retries after a lost response still make one booking, two calls racing for a slot can't both win, and a model stuck in a tool loop hands off to a human.

**Then I needed a way to know if it's any good.** 150 scenarios across 14 categories (corrections, emergencies, gas leaks, people backing out, price questions...), 30% held out. An LLM plays the caller from a private brief. Scoring checks the database, not the vibes: right service, right customer, right address, a slot they can make, exactly once.

**First result was 96%, and it was wrong.** Reading the transcripts, the agent was looking up phone numbers nobody gave it, inventing zip codes, and telling people "I have that scheduled for you" before booking anything. The database happened to absorb all of it. So I added checks on the trace itself, and the same calls dropped to 39%. Biggest lesson so far: outcome-only evals can make a sloppy agent look great.

**The fake callers broke too.** One model kept hanging up mid-sentence, another wrote the agent's lines for it. So hang-ups have to actually sound like goodbyes now, and runs where the caller goes off-script get flagged instead of counted against the agent.

**Then the actual point: letting a model fix the agent.** The Agent Manager reads the training failures (transcripts and which check failed, never the ground truth or the held-out results) and proposes one procedure edit. That becomes a new version. The trust gate compares it to the old one, mostly on held-out callers the Manager never saw. It promotes, sends to a human, or rejects. It also flags fixes that mention specific customers, since that's memorizing, not fixing.

**Where it's at.** The first proposal went after the "scheduled too early" habit. Early numbers say it didn't fix it: the agent still says "scheduled" during the read-back. Which is kind of the point of having a gate. Full comparison is running now (slowly, on free tiers).

## Running it

```bash
uv sync
cp .env.example .env               # free Gemini + Groq keys, see the file
uv run evalgate check              # make sure keys and models work
uv run pytest -q                   # no keys needed
uv run evalgate chat --tools       # talk to the agent yourself

uv run evalgate eval --split all --per-category 2                         # baseline
uv run evalgate propose runs/<baseline>                                   # Manager proposes a fix
uv run evalgate eval --split all --per-category 2 --version versions/<id>.json
uv run evalgate compare runs/<baseline> runs/<candidate>                  # gate decision
```

Free tiers run out fast. When a run hits the daily limit it stops cleanly and prints the `--resume` command for tomorrow. Full spec in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).
