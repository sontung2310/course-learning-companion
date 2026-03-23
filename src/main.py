import logging
import tracemalloc
from contextlib import asynccontextmanager

from fastapi import FastAPI
from nemoguardrails import LLMRails, RailsConfig

from src.routers.api import api_router
from src.services.memory.long_term_memory import LongTermMemoryService
from src.services.users import Users
from src.settings import APP_CONFIGS, SETTINGS
from src.services.agents import LearningOrchestrator

# LearningOrchestrator is imported inside lifespan so `import src.main` does not load CrewAI/torch.

tracemalloc.start()


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[type-arg]
        return (
            record.args is not None
            and len(record.args) >= 3
            and list(record.args)[2] not in ["/health", "/ready"]
        )


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

_startup_log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # _startup_log.info("Startup: PostgreSQL (long-term memory tables)…")
    # long_term_memory = LongTermMemoryService()
    # await long_term_memory.initialize()

    # _startup_log.info("Startup: LearningOrchestrator (CrewAI; first load can take a while)…")
    # from src.rails import actions as rails_actions
    # from src.services.agents import LearningOrchestrator

    # app.state.learning_orchestrator = LearningOrchestrator()
    # rails_actions.set_learning_orchestrator(app.state.learning_orchestrator)
    # app.state.users_service = Users()

    # _startup_log.info("Startup: NeMo Guardrails (RailsConfig + LLMRails)…")
    # config = RailsConfig.from_path("src/rails")
    # app.state.rails = LLMRails(config)

    # _startup_log.info("Startup complete — accepting requests.")
    long_term_memory = LongTermMemoryService()
    await long_term_memory.initialize()

    app.state.learning_orchestrator = LearningOrchestrator()
    app.state.users_service = Users()
    yield


app = FastAPI(**APP_CONFIGS, lifespan=lifespan)


@app.get("/health", include_in_schema=False)
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", include_in_schema=False)
async def readycheck() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(
    api_router,
    prefix=SETTINGS.API_V1_STR,
)

