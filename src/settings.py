from pydantic_settings import BaseSettings
from pydantic import SecretStr
from typing import Any
import os
from dotenv import load_dotenv

BASEDIR = os.path.abspath(os.path.dirname(__file__))

# Connect the path with your '.env' file name
load_dotenv(os.path.join(BASEDIR, "../.env"))


class Settings(BaseSettings):
    """Settings for the application."""

    ROOT_PATH: str = ""
    API_V1_STR: str = "/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8055
    ENVIRONMENT: str = "development"

    # OpenAI API key
    OPENAI_API_KEY: SecretStr
    OPENAI_BASE_URL: str = "http://localhost:4000"
    OPENAI_TEMPERATURE: float = 0.7

    # Milvus configuration
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: str = "19530"
    MILVUS_COLLECTION: str = "langchain_docs"

    # PostgreSQL configuration for long-term memory
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "postgres"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: SecretStr  # optional for local dev; set in .env for real DB

    # Redis configuration for short-term memory
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: SecretStr | None = None

    # Redis database for caching
    REDIS_DB_CACHE: int = 1

    # Langfuse configuration
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_SECRET_KEY: SecretStr | None = None
    LANGFUSE_HOST: str | None = None


SETTINGS = Settings()  # type: ignore

APP_CONFIGS: dict[str, Any] = {
    "title": "Course Learning Assistant",
    "root_path": SETTINGS.ROOT_PATH,
}
