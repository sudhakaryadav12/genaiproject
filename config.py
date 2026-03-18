
# =====================
# File: src/vector_chunks_sdk/config.py
# =====================
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
import os

# class LLMConfig(BaseModel):
#     provider: str = Field(..., description="e.g., openai, azure_openai, custom")
#     endpoint: str = Field(...)
#     api_key: str = Field(...)
#     model: str = Field(...)
#     api_version: str = Field(...)
#     prompt_template: Optional[str] = Field(
#         default="",
#         description="Jinja-like template where {{query}} and {{chat_history}} can appear"
#     )
#     extra: Dict[str, Any] = Field(default_factory=dict)

class LLMConfig(BaseModel):
    provider: str
    endpoint: str
    api_key: str
    model: str
    api_version: str
    prompt_template: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

class EmbeddingConfig(BaseModel):
    provider: str
    endpoint: str
    api_key: str
    api_version: str
    model: str
    dim: int
    extra: Dict[str, Any] = Field(default_factory=dict)

class RerankConfig(BaseModel):
    endpoint: str
    api_key: str
    model: str
    prompt_template: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

class DBConfig(BaseModel):
    kind: str = Field(..., description="chroma | milvus | azure_ai_search")
    collection_name: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)

class GlobalConfig(BaseModel):
    llm: LLMConfig
    embeddings: EmbeddingConfig
    vectordb: DBConfig
    rerank: Optional[RerankConfig] = None
    @staticmethod
    def from_env(prefix: str = "VCS") -> "GlobalConfig":
        """Load from environment variables (e.g., VCS_LLM_PROVIDER)."""
        llm = LLMConfig(
            provider=os.getenv(f"{prefix}_LLM_PROVIDER", "openai"),
            endpoint=os.getenv(f"{prefix}_LLM_ENDPOINT", ""),
            api_key=os.getenv(f"{prefix}_LLM_API_KEY", ""),
            model=os.getenv(f"{prefix}_LLM_MODEL", ""),
            prompt_template=os.getenv(f"{prefix}_LLM_PROMPT", "")
        )
        emb = EmbeddingConfig(
            provider=os.getenv(f"{prefix}_EMB_PROVIDER", "openai"),
            endpoint=os.getenv(f"{prefix}_EMB_ENDPOINT", ""),
            api_key=os.getenv(f"{prefix}_EMB_API_KEY", ""),
            model=os.getenv(f"{prefix}_EMB_MODEL", ""),
            dim=int(os.getenv(f"{prefix}_EMB_DIM", "1536")),
        )
        db = DBConfig(
            kind=os.getenv(f"{prefix}_DB_KIND", "chroma"),
            collection_name=os.getenv(f"{prefix}_DB_COLLECTION", None),
            params={}
        )
        rerank = RerankConfig(
            endpoint=os.getenv(f"{prefix}_RERANK_ENDPOINT", ""),
            api_key=os.getenv(f"{prefix}_RERANK_API_KEY", ""),
            model=os.getenv(f"{prefix}_RERANK_MODEL", ""),
            prompt_template=os.getenv(f"{prefix}_RERANK_PROMPT", "")
        )
        return GlobalConfig(llm=llm, embeddings=emb, vectordb=db, rerank=rerank)
