# Results so far

Agent: Gemini 3.1 Flash-Lite. Simulated callers: Qwen3.8 27B on Groq (gpt-oss-120b for the first 28 training calls). One run per scenario, 2 scenarios per category per split, free-tier APIs. Small numbers, so read these as "what happened", not as stats.

## Baseline (v-4633ecf0f5)

56 calls, 28 train and 28 held-out.

| | Pass rate |
|---|---|
| Outcome checks only | 92.9% |
| With process checks | 32.1% (train 39.3%, held-out 25.0%) |

The agent almost always ended up with the right booking. What it did along the way was the problem:

- **Saying "scheduled" too early.** "I have a boiler repair scheduled for you..." during the read-back, before anything was booked. Most common failure by far.
- **Lookups on numbers nobody said.** On a vague opener ("Hi, I'd like to schedule a repair") it would call `lookup_customer` with a made-up number before asking for one.
- **Invented account details.** Creating accounts with city "Unknown" or a guessed zip, usually for customers who already existed.

The database absorbed all of these, which is why outcome-only grading looked so good.

## Round 1: the Manager's first fix (v-4d4d87dd6d)

The Manager went after the biggest cluster, "scheduled too early" (12 training failures), and edited the booking procedure to add: *"Do not state that an appointment is scheduled or booked until the caller clearly agrees in their next message."*

Same 56 calls (53 paired; 2 skipped because their scenario changed between runs):

| Split | Base | Candidate | Fixed | Broken |
|---|---|---|---|---|
| train | 40.0% | 28.0% | 3 | 6 |
| held-out | 25.0% | 25.0% | 2 | 2 |

| Check (calls failing) | Base | Candidate |
|---|---|---|
| no_premature_confirmation | 26 | 24 |
| correct_customer / correct_address / slot_fits_availability | 0 | 8 |
| no_unspoken_phone_lookup | 16 | 6 |
| no_invented_customer_details | 12 | 2 |

What happened:

1. **It didn't fix its target.** 26 to 24 failures. The agent still said "scheduled" during the read-back. The edit added a rule, but the habit comes from how the model phrases the read-back, and the new text never told it how to phrase it differently.
2. **It caused a worse failure.** On vague openers, the baseline's reflex was a made-up `lookup_customer` call. With the new text, the reflex became `end_call`: it asked for the caller's number and hung up in the same turn. This reproduced 4 out of 4 times on a rerun, and never happened with the baseline on the same calls.
3. **It looked like an improvement on two metrics.** Made-up lookups dropped from 16 to 6, invented details from 12 to 2. That's only because an agent that hangs up immediately never gets the chance to make those mistakes.

The hang-ups didn't show up as "newly broken" calls, because those calls were already failing a process check in the baseline. They only show up in the outcome checks going from 0 to 8.

**Gate decision:** needs_human at the time, since held-out didn't drop and no safety call broke. Reviewing it, I rejected it, and tightened the gate so this kind of change now gets rejected automatically:

- reject when a change breaks more calls than it fixes
- reject when an outcome check (right customer, address, slot...) gets worse, even on calls that were already failing for another reason

## What I take from it

- A fix that sounds right isn't evidence. This one read like a sensible procedure edit and made the agent worse.
- Pass/fail per call hides things. Per-check deltas caught the regression that "fixed vs broken" missed.
- Process metrics can improve for the wrong reason. Fewer bad tool calls because the agent stopped talking to people isn't a win.

## Next

Round 2: the Manager now sees its rejected attempts (the text it tried, why the gate stopped it, and one conversation it broke) before proposing again.
