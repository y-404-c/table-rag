"""全局配置：默认值 + .env 覆盖（前缀 TR_，嵌套用 __）。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LLMSettings(BaseModel):
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3.8-flash"
    api_key: str = ""
    # 额度耗尽后自动轮换的 key 列表（逗号 / 分号 / 换行分隔）；为空则只用 api_key
    api_keys: str = ""
    temperature: float = 0.0
    max_tokens: int = 1200
    timeout_s: float = 90.0
    max_retries: int = 2
    # qwen4-max 默认开思考链，会吃 completion token 且偶发空输出；默认关掉
    enable_thinking: bool = False

    def resolved_keys(self) -> list[str]:
        from tablerag.llm.keypool import parse_keys

        keys = parse_keys(self.api_keys)
        if not keys and self.api_key.strip():
            keys = [self.api_key.strip()]
        return keys


class EmbeddingSettings(BaseModel):
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "text-embedding-v3"
    api_key: str = ""
    batch_size: int = 10
    timeout_s: float = 60.0


class RerankSettings(BaseModel):
    url: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    model: str = "gte-rerank-v2"
    api_key: str = ""
    enabled: bool = True
    timeout_s: float = 60.0


class CacheSettings(BaseModel):
    # sqlite | redis | memory
    backend: str = "sqlite"
    url: str = "redis://localhost:6379/0"
    ttl_s: int = 7 * 24 * 3600


class RetrievalSettings(BaseModel):
    bm25_top_k: int = 20
    vector_top_k: int = 20
    rrf_k: int = 60
    final_top_k: int = 8
    hyde: bool = False
    max_hyde_rounds: int = 2
    rerank_truncate_chars: int = 1200


class RenderSettings(BaseModel):
    """文档化渲染的扰动参数。固定 seed，保证可复现。"""

    seed: int = 42
    page_breaks: bool = True
    merged_cells: bool = True
    two_column: bool = True
    whitespace: bool = True
    font_size: float = 8.0


class AgentSettings(BaseModel):
    max_evidence_rounds: int = 2
    compute_enabled: bool = True
    trace_sink: str = "jsonl"  # jsonl（本地 trace）；langfuse 未接入


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TR_",
        env_nested_delimiter="__",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    render: RenderSettings = Field(default_factory=RenderSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)

    data_dir: Path = PROJECT_ROOT / "data"
    var_dir: Path = PROJECT_ROOT / "var"
    reports_dir: Path = PROJECT_ROOT / "reports"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def rendered_dir(self) -> Path:
        return self.data_dir / "rendered"

    @property
    def splits_path(self) -> Path:
        return self.data_dir / "splits.json"

    @property
    def db_path(self) -> Path:
        return self.var_dir / "tablerag.db"

    @property
    def chroma_dir(self) -> Path:
        return self.var_dir / "chroma"

    @property
    def trace_dir(self) -> Path:
        return self.var_dir / "traces"

    def ensure_dirs(self) -> None:
        for path in (self.raw_dir, self.rendered_dir, self.var_dir, self.chroma_dir,
                     self.trace_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)

    def api_keys_resolved(self) -> Settings:
        """embedding / rerank 未单独配 key 时复用 LLM 的 key（同一百炼账号）。"""
        if not self.embedding.api_key:
            self.embedding.api_key = self.llm.api_key
        if not self.rerank.api_key:
            self.rerank.api_key = self.llm.api_key
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings.api_keys_resolved()
