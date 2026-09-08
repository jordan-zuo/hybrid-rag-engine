from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Enterprise Hybrid RAG Engine (Embedded Low-RAM)"

    QDRANT_STORAGE_PATH: str = "./qdrant_local_data"
    COLLECTION_NAME: str = "enterprise_knowledge"
    VECTOR_SIZE: int = 384

    CACHE_SIMILARITY_THRESHOLD: float = 0.92

    DENSE_MODEL_NAME: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    DENSE_TOP_K: int = 20
    SPARSE_TOP_K: int = 20

    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    # Generation provider (hot path): llm7.io verified 200 on
    # mistral-Nemo-Instruct-2407 / codestral-latest. GROQ_* kept as fallback.
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.llm7.io/v1"
    LLM_MODEL: str = "mistral-Nemo-Instruct-2407"

    # Judge provider (eval-time only, 6 sequential calls): Groq qwen3.8-27b
    # reasons well; its 1000 OTPM budget is plenty at max_tokens=300.
    # JUDGE_API_KEY falls back to GROQ_API_KEY when empty.
    JUDGE_API_KEY: str = ""
    JUDGE_BASE_URL: str = "https://api.groq.com/openai/v1"
    JUDGE_MODEL: str = "qwen/qwen3.8-27b"

    RRF_K: int = 60

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def gen_api_key(self) -> str:
        return self.LLM_API_KEY or self.GROQ_API_KEY

    @property
    def gen_base_url(self) -> str:
        if self.LLM_API_KEY:
            return self.LLM_BASE_URL
        return self.GROQ_BASE_URL

    @property
    def gen_provider(self) -> str:
        return "llm7" if self.LLM_API_KEY else "groq"

    @property
    def judge_api_key(self) -> str:
        return self.JUDGE_API_KEY or self.GROQ_API_KEY

    @property
    def judge_base_url(self) -> str:
        return self.JUDGE_BASE_URL

    @property
    def effective_api_key(self) -> str:
        return self.gen_api_key

    @property
    def effective_base_url(self) -> str:
        return self.gen_base_url


settings = Settings()
