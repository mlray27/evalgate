"""Wiring: build the LLM, database and default agent version from settings."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from evalgate.config import Settings
from evalgate.db import Customer, create_all, make_engine, make_sessionmaker
from evalgate.llm import LLM, OpenAICompatLLM
from evalgate.seed import seed
from evalgate.sops import AgentVersion, load_library


def build_llm(settings: Settings) -> LLM:
    if not settings.llm_api_key:
        raise RuntimeError("Set EVALGATE_LLM_API_KEY (see .env.example).")
    return OpenAICompatLLM(
        settings.llm_base_url,
        settings.llm_api_key,
        settings.llm_model,
        settings.llm_temperature,
        settings.llm_timeout_s,
        settings.llm_max_retries,
        settings.llm_reasoning_effort,
        settings.rate_limit_patience_s,
    )


def build_sim_llm(settings: Settings) -> LLM:
    key = settings.sim_api_key or settings.llm_api_key
    if not key:
        raise RuntimeError("Set EVALGATE_LLM_API_KEY (or EVALGATE_SIM_API_KEY).")
    return OpenAICompatLLM(
        settings.sim_base_url or settings.llm_base_url,
        key,
        settings.sim_model or settings.llm_model,
        settings.sim_temperature,
        settings.llm_timeout_s,
        settings.llm_max_retries,
        settings.sim_reasoning_effort,
        settings.rate_limit_patience_s,
    )


def build_manager_llm(settings: Settings) -> LLM:
    key = settings.manager_api_key or settings.llm_api_key
    if not key:
        raise RuntimeError("Set EVALGATE_LLM_API_KEY (or EVALGATE_MANAGER_API_KEY).")
    return OpenAICompatLLM(
        settings.manager_base_url or settings.llm_base_url,
        key,
        settings.manager_model or settings.llm_model,
        settings.manager_temperature,
        60.0,
        settings.llm_max_retries,
        settings.llm_reasoning_effort,
        settings.rate_limit_patience_s,
    )


def default_version(settings: Settings) -> AgentVersion:
    return AgentVersion(sops=load_library(), model=settings.llm_model, note="baseline")


async def open_db(settings: Settings) -> tuple[AsyncEngine, async_sessionmaker]:
    engine = make_engine(settings.database_url)
    sm = make_sessionmaker(engine)
    await create_all(engine)
    async with sm() as s:
        n = await s.scalar(select(func.count()).select_from(Customer))
    if not n:
        await seed(engine, sm, settings.today, settings.seed)
    return engine, sm
