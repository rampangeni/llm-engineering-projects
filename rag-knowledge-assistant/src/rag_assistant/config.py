"""Central configuration, read from environment variables (and .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PIPELINES = ("basic", "pro")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_data_dir() -> Path:
    return PROJECT_ROOT / "data"


@dataclass
class Settings:
    pipeline: str = "basic"
    data_dir: Path = field(default_factory=_default_data_dir)
    basic_llm_model: str = "openai/gpt-4.1-nano"
    pro_llm_model: str = "openai/gpt-4.1-nano"
    judge_llm_model: str = "openai/gpt-4.1-nano"
    basic_embedding: str = "hf:all-MiniLM-L6-v2"
    pro_embedding: str = "openai:text-embedding-3-large"
    # Basic: LangChain splitter parameters and retrieval depth
    chunk_size: int = 500
    chunk_overlap: int = 200
    basic_retrieval_k: int = 10
    # Pro: fetch RETRIEVAL_K per query, keep FINAL_K after reranking
    pro_retrieval_k: int = 20
    pro_final_k: int = 10
    pro_workers: int = 3
    pro_avg_chunk_chars: int = 100
    embed_batch_size: int = 100

    @property
    def knowledge_base_dir(self) -> Path:
        return self.data_dir / "knowledge-base"

    @property
    def tests_file(self) -> Path:
        return self.data_dir / "tests.jsonl"

    def db_dir(self, pipeline: str) -> Path:
        return self.data_dir / "vector_db" / pipeline

    def llm_model(self, pipeline: str) -> str:
        return self.pro_llm_model if pipeline == "pro" else self.basic_llm_model

    def embedding_spec(self, pipeline: str) -> str:
        return self.pro_embedding if pipeline == "pro" else self.basic_embedding

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(override=True)
        env = os.environ
        s = cls()
        s.pipeline = env.get("RAG_PIPELINE", s.pipeline).lower()
        if env.get("RAG_DATA_DIR"):
            s.data_dir = Path(env["RAG_DATA_DIR"]).expanduser().resolve()
        s.basic_llm_model = env.get("BASIC_LLM_MODEL", s.basic_llm_model)
        s.pro_llm_model = env.get("PRO_LLM_MODEL", s.pro_llm_model)
        s.judge_llm_model = env.get("JUDGE_LLM_MODEL", s.judge_llm_model)
        s.basic_embedding = env.get("BASIC_EMBEDDING", s.basic_embedding)
        s.pro_embedding = env.get("PRO_EMBEDDING", s.pro_embedding)
        if s.pipeline not in PIPELINES:
            raise ValueError(f"RAG_PIPELINE must be one of {PIPELINES}, got {s.pipeline!r}")
        return s
