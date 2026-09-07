from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.engine import RAGEngine, RAGResponse

SAMPLE_DOCUMENTS: list[dict[str, Any]] = [
    {
        "id": "q3-financials",
        "metadata": {"title": "Q3 Financials", "doc_type": "finance"},
        "text": (
            "Q3 Financials: Enterprise revenue was $48.2 million, up 12% year over year. "
            "Gross margin was 71.3% and operating margin was 18.4%. "
            "Cash and equivalents totaled $126 million at quarter end. "
            "Net new ARR was $6.1 million, with remaining performance obligations of $91 million. "
            "The finance committee approved a $4.0 million buyback authorization."
        ),
    },
    {
        "id": "security-policy",
        "metadata": {"title": "Security Policy", "doc_type": "security"},
        "text": (
            "Security Policy: Production access requires hardware-backed MFA. "
            "API keys must be rotated every 90 days and stored only in the secrets manager. "
            "Data at rest is encrypted with AES-256 and data in transit uses TLS 1.3. "
            "The control environment is SOC 2 Type II certified. "
            "Critical vulnerabilities must be patched within 24 hours of vendor disclosure."
        ),
    },
    {
        "id": "sla-guarantee",
        "metadata": {"title": "SLA Guarantee", "doc_type": "legal"},
        "text": (
            "SLA Guarantee: The platform commits to 99.9% monthly uptime excluding announced maintenance. "
            "Priority-1 incidents have a 15-minute response target and a 4-hour restoration target. "
            "If monthly uptime falls below 99.9%, customers receive a 10% service credit. "
            "If monthly uptime falls below 99.0%, the credit increases to 25%. "
            "Credits must be requested within 30 days of the affected invoice."
        ),
    },
]


class IngestDocument(BaseModel):
    id: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestDocument]


class IngestResponse(BaseModel):
    ingested: int
    documents: int


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    top_fused: int = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = RAGEngine()
    engine.index_documents(SAMPLE_DOCUMENTS)
    app.state.engine = engine
    yield


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)


def get_engine() -> RAGEngine:
    return app.state.engine


@app.get("/health")
def health() -> dict[str, Any]:
    engine = get_engine()
    return {"status": "ok", "mode": "embedded-zero-docker", **engine.stats()}


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    if not payload.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    engine = get_engine()
    ingested = engine.index_documents([doc.model_dump() for doc in payload.documents])
    return IngestResponse(ingested=ingested, documents=engine.retriever.document_count())


@app.post("/query", response_model=RAGResponse)
def query(payload: QueryRequest) -> RAGResponse:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return get_engine().query(
        query=payload.query,
        top_k=payload.top_k,
        top_fused=payload.top_fused,
    )
