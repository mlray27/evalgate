from datetime import date
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVALGATE_", env_file=".env", extra="ignore")

    llm_base_url: str = "https://api.together.ai/v1"
    llm_api_key: str = ""
    llm_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
    llm_temperature: float = 0.2
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 4  # SDK retries for connection errors and 5xx
    llm_reasoning_effort: str = ""  # e.g. "low"; empty = provider default
    rate_limit_patience_s: float = 300.0

    # Zero-setup default; point at Postgres from docker-compose for realistic runs.
    database_url: str = "sqlite+aiosqlite:///evalgate.db"

    # A fixed "today" keeps seed data, availability and evals reproducible.
    # Mid-January: furnace calls are emergencies.
    today: date = date(2026, 1, 12)
    company_name: str = "Brightline Heating & Plumbing"
    seed: int = 7

    max_tool_steps_per_turn: int = 6
    read_tool_timeout_s: float = 3.0
    read_tool_retries: int = 1
    proposal_ttl_turns: int = 6

    trace_dir: str = "traces"

    # Simulated callers. Defaults to the agent's endpoint and model if left empty.
    sim_base_url: str = ""
    sim_api_key: str = ""
    sim_model: str = ""
    sim_temperature: float = 0.7
    sim_reasoning_effort: str = "low"  # the caller is role-play, not puzzles; saves tokens
    max_caller_turns: int = 16
    conversation_timeout_s: float = 1200.0  # rate-limit waits can take minutes
    eval_concurrency: int = 4
    runs_dir: str = "runs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
