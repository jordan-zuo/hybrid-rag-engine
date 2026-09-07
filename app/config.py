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
    LLM_MODEL: str = "llama-3.3-70b-versatile"

    RRF_K: int = 60

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
