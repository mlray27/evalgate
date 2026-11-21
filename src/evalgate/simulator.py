"""Simulated caller. An LLM plays the customer from a scenario brief it treats as private truth.

The simulator never sees the agent's tools, procedures, or internal state, only what the agent
says out loud. It ends the call by putting [HANGUP] at the end of its message.
"""

from __future__ import annotations

from evalgate.llm import LLM, Completion, TextDelta
from evalgate.scenarios import Scenario

HANGUP = "[HANGUP]"


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
        f"- When the call is over (booking confirmed and recapped, told someone will call back, or "
        f"you decide to end it), say a short goodbye and end your message with {HANGUP}.\n"
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
        said = "".join(text).strip()
        if HANGUP in said:
            self.hung_up = True
            said = said.replace(HANGUP, "").strip()
        self.messages.append({"role": "assistant", "content": said})
        return said
