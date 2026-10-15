from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
# Export .env into os.environ so LangChain / OpenAI / Tavily clients pick keys up too.
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    app_name: str = "Atlas"

    openai_api_key: str = ""
    tavily_api_key: str = ""
    pinecone_api_key: str = ""

    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    pinecone_index: str = "ai-agent-studio"
    pinecone_namespace: str = "default"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 5
    max_refinements: int = 1  # how many times the agent may rewrite the query and retry

    data_dir: Path = BASE_DIR / "data"

    def missing_keys(self) -> list[str]:
        names = {
            "OPENAI_API_KEY": self.openai_api_key,
            "TAVILY_API_KEY": self.tavily_api_key,
            "PINECONE_API_KEY": self.pinecone_api_key,
        }
        return [k for k, v in names.items() if not v]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
