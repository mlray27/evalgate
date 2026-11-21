"""Simulated caller. An LLM plays the customer from a scenario brief it treats as private truth.

The simulator never sees the agent's tools, procedures, or internal state, only what the agent
says out loud. It ends the call by putting [HANGUP] at the end of a goodbye message.

Models sometimes tack [HANGUP] onto a message mid-call (seen with gpt-oss on Groq), which cut
real conversations short. So a hang-up only counts if the message actually reads like a goodbye:
a farewell word and no question. Otherwise the tag is dropped and the call continues.
"""

from __future__ import annotations

import re

from evalgate.llm import LLM, Completion, TextDelta
from evalgate.scenarios import Scenario

HANGUP = "[HANGUP]"
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
FAREWELL_RE = re.compile(
    r"\b(bye|goodbye|good night|thanks|thank you|thank u|take care|have a (good|great|nice))\b",
    re.I,
)


def is_real_hangup(text: str) -> bool:
    """A goodbye, not a mid-call answer: has a farewell word and asks nothing."""
    return bool(FAREWELL_RE.search(text)) and "?" not in text


def caller_system_prompt(s: Scenario, company: str) -> str:
    return (
        f"You are role-playing a customer phoning {company}, a heating, cooling, and plumbing "
        f"company. You are talking to their phone agent.\n\n"
        f"Your personality: {s.persona}.\n\n"
        f"Your situation (private; reveal details naturally, mostly when asked):\n{s.brief}\n\n"
        "How to talk:\n"
        "- Write only what you say out loud. One or two short sentences, like a real phone call.\n"
        "- No stage directions, no narration, no lists.\n"
        "- Answer what the agent asks. Don't dump every fact at once.\n"
        "- If the agent reads back details that don't match your situation (wrong address, wrong "
        "problem, a time you're not available), correct them.\n"
        "- Only agree to a time that fits your availability.\n"
        "- If the agent asks you to confirm a booking that is correct, agree plainly.\n"
        "- Never say you are an AI or that this is a test.\n"
        f"- Hang up only when the call is truly over: the agent has confirmed your booking, told "
        f"you someone will call back, or you have decided to end the call. Then say a short "
        f"goodbye and put {HANGUP} at the very end.\n"
        f"- Never write {HANGUP} while you are still giving information, asking a question, or "
        f"waiting for the agent to do something. Most of your messages must NOT contain it.\n"
    )


class CallerSimulator:
    def __init__(self, scenario: Scenario, llm: LLM, company: str) -> None:
        self.scenario = scenario
        self.llm = llm
        self.hung_up = False
        # From the caller's point of view, the agent is the "user" and the caller is "assistant".
        self.messages: list[dict] = [
            {"role": "system", "content": caller_system_prompt(scenario, company)},
            {"role": "assistant", "content": scenario.opening},
        ]
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.ignored_hangups = 0
        self.last_raw = ""
        self.ignored_hangup = False

    @property
    def opening(self) -> str:
        return self.scenario.opening

    async def reply(self, agent_text: str) -> str:
        self.messages.append({"role": "user", "content": agent_text or "(silence)"})
        text: list[str] = []
        async for ev in self.llm.stream(self.messages, []):
            if isinstance(ev, TextDelta):
                text.append(ev.text)
            elif isinstance(ev, Completion):
                self.prompt_tokens += ev.prompt_tokens
                self.completion_tokens += ev.completion_tokens
        raw = "".join(text)
        said = THINK_RE.sub("", raw).strip()  # some models leak reasoning
        self.last_raw = raw
        self.ignored_hangup = False
        if HANGUP in said:
            said = said.replace(HANGUP, "").strip()
            if is_real_hangup(said):
                self.hung_up = True
            else:
                self.ignored_hangup = True
                self.ignored_hangups += 1
        self.messages.append({"role": "assistant", "content": said})
        return said
