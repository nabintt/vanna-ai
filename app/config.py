from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = Field(default="local-vanna-ai", validation_alias="APP_NAME")
    app_host: str = Field(default="0.0.0.0", validation_alias="APP_HOST")
    app_port: int = Field(default=8000, validation_alias="APP_PORT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    ollama_host: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_HOST")
    ollama_model: str = Field(default="llama3.2", validation_alias="OLLAMA_MODEL")
    ollama_embed_model: str | None = Field(default=None, validation_alias="OLLAMA_EMBED_MODEL")
    ollama_timeout: float = Field(default=120.0, validation_alias="OLLAMA_TIMEOUT")
    ollama_keep_alive: str | None = Field(default="15m", validation_alias="OLLAMA_KEEP_ALIVE")
    ollama_num_ctx: int = Field(default=4096, validation_alias="OLLAMA_NUM_CTX")

    db_type: str = Field(default="postgres", validation_alias="DB_TYPE")
    db_host: str = Field(default="localhost", validation_alias="DB_HOST")
    db_port: int = Field(default=5432, validation_alias="DB_PORT")
    db_name: str = Field(default="", validation_alias="DB_NAME")
    db_user: str = Field(default="", validation_alias="DB_USER")
    db_password: str = Field(default="", validation_alias="DB_PASSWORD")

    train_on_start: bool = Field(default=True, validation_alias="TRAIN_ON_START")
    allow_bootstrap_sample_data: bool = Field(
        default=True,
        validation_alias="ALLOW_BOOTSTRAP_SAMPLE_DATA",
    )
    max_result_rows: int = Field(default=200, validation_alias="MAX_RESULT_ROWS")
    vanna_top_k: int = Field(default=10, validation_alias="VANNA_TOP_K")

    chroma_path: Path = Field(default=BASE_DIR / "data" / "chroma", validation_alias="CHROMA_PATH")
    training_data_dir: Path = Field(
        default=BASE_DIR / "data" / "training",
        validation_alias="TRAINING_DATA_DIR",
    )
    business_glossary_path: Path = Field(
        default=BASE_DIR / "data" / "training" / "business_glossary.md",
        validation_alias="BUSINESS_GLOSSARY_PATH",
    )
    example_pairs_path: Path = Field(
        default=BASE_DIR / "data" / "training" / "example_question_sql.json",
        validation_alias="EXAMPLE_PAIRS_PATH",
    )
    bootstrap_state_path: Path = Field(
        default=BASE_DIR / "data" / "training" / "bootstrap_state.json",
        validation_alias="BOOTSTRAP_STATE_PATH",
    )

    @property
    def normalized_ollama_model(self) -> str:
        return normalize_ollama_model_name(self.ollama_model)

    @property
    def normalized_ollama_embed_model(self) -> str | None:
        if not self.ollama_embed_model:
            return None
        return normalize_ollama_model_name(self.ollama_embed_model)

    @property
    def normalized_db_type(self) -> str:
        return self.db_type.strip().lower()

    @property
    def database_target(self) -> str:
        return (
            f"{self.normalized_db_type}://{self.db_user or '<unset>'}"
            f"@{self.db_host}:{self.db_port}/{self.db_name or '<unset>'}"
        )

    @property
    def sqlalchemy_url(self) -> str:
        self.validate_database_config()
        driver = {
            "postgres": "postgresql+psycopg2",
            "mysql": "mysql+pymysql",
        }.get(self.normalized_db_type)
        if driver is None:
            raise ValueError(
                f"Unsupported DB_TYPE '{self.db_type}'. Supported values are 'postgres' and 'mysql'."
            )

        username = quote_plus(self.db_user)
        password = quote_plus(self.db_password)
        database = quote_plus(self.db_name)
        return f"{driver}://{username}:{password}@{self.db_host}:{self.db_port}/{database}"

    def ensure_directories(self) -> None:
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.training_data_dir.mkdir(parents=True, exist_ok=True)

    def validate_database_config(self) -> None:
        missing = [
            name
            for name, value in {
                "DB_NAME": self.db_name,
                "DB_USER": self.db_user,
                "DB_PASSWORD": self.db_password,
            }.items()
            if not str(value).strip()
        ]
        if missing:
            raise ValueError(
                "Missing required database settings: "
                + ", ".join(missing)
                + ". Update your .env file before starting the server."
            )


def normalize_ollama_model_name(model_name: str) -> str:
    model_name = model_name.strip()
    if not model_name:
        raise ValueError("OLLAMA_MODEL cannot be empty.")
    if ":" not in model_name:
        return f"{model_name}:latest"
    return model_name


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
