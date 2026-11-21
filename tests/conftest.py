import pytest

from evalgate.config import Settings
from evalgate.db import make_engine, make_sessionmaker
from evalgate.seed import seed
from evalgate.sops import AgentVersion, load_library


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
        llm_api_key="",
        trace_dir=str(tmp_path / "traces"),
    )


@pytest.fixture
async def sm(settings):
    engine = make_engine(settings.database_url)
    maker = make_sessionmaker(engine)
    await seed(engine, maker, settings.today, settings.seed)
    yield maker
    await engine.dispose()


@pytest.fixture
def version() -> AgentVersion:
    return AgentVersion(sops=load_library(), model="scripted")
