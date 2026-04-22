from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


class Settings:
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
    CLEAN_DATA_DIR: Path = PROJECT_ROOT / "data" / "clean"
    OUTPUT_DATA_DIR: Path = PROJECT_ROOT / "data" / "output"
    LOG_DIR: Path = PROJECT_ROOT / "logs"

    SELENIUM_HEADLESS: bool = _env_bool("SELENIUM_HEADLESS", True)
    CRAWL_DELAY: float = _env_float("CRAWL_DELAY", 1.5)
    REQUEST_TIMEOUT: int = int(_env_float("REQUEST_TIMEOUT", 30))

    VNEXPRESS_BASE_URL: str = os.getenv(
        "VNEXPRESS_BASE_URL", "https://vnexpress.net/khoa-hoc-cong-nghe"
    )
    VNEXPRESS_CATEGORY: str = os.getenv("VNEXPRESS_CATEGORY", "khoa-hoc-cong-nghe")


    # LLM Analyzer
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash")
    LLM_TEMPERATURE: float = _env_float("LLM_TEMPERATURE", 0.2)
    LLM_MAX_TOKENS: int = int(_env_float("LLM_MAX_TOKENS", 2048))
    LLM_TIMEOUT: int = int(_env_float("LLM_TIMEOUT", 120))

    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    OLLAMA_API_BASE: str = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")


    #Ollama-specific tuning

    OLLAMA_NUM_CTX: int = int(_env_float("OLLAMA_NUM_CTX", 8192))
    OLLAMA_NUM_PREDICT: int = int(_env_float("OLLAMA_NUM_PREDICT", 2048))

    # Near-greedy sampling for deterministic JSON emission.
    OLLAMA_TOP_P: float = _env_float("OLLAMA_TOP_P", 0.9)
    OLLAMA_TOP_K: int = int(_env_float("OLLAMA_TOP_K", 40))
    OLLAMA_REPEAT_PENALTY: float = _env_float("OLLAMA_REPEAT_PENALTY", 1.05)

    # Warn when an Ollama prompt reaches this fraction of ``num_ctx`` —
    # past this point the server silently truncates the head of the prompt.
    PREFLIGHT_CTX_RATIO: float = _env_float("PREFLIGHT_CTX_RATIO", 0.9)

    # Conservative VN-corpus estimate (real value 2.5–3.0 chars/token).
    PREFLIGHT_CHARS_PER_TOKEN: float = _env_float("PREFLIGHT_CHARS_PER_TOKEN", 2.5)

    LLM_CACHE_DIR: Path = PROJECT_ROOT / ".llm_cache"
    LLM_CACHE_ENABLED: bool = _env_bool("LLM_CACHE_ENABLED", True)

    # Keyword extraction
    KEYWORD_TOP_K: int = int(_env_float("KEYWORD_TOP_K", 15))
    TFIDF_MIN_DF: int = int(_env_float("TFIDF_MIN_DF", 3))
    TFIDF_MAX_DF: float = _env_float("TFIDF_MAX_DF", 0.6)

    STOPWORDS_FILE: Path = Path(os.getenv("STOPWORDS_FILE", str(PROJECT_ROOT / "resources" / "vietnamese-stopwords.txt")))
    TFIDF_NGRAM_MIN: int = int(_env_float("TFIDF_NGRAM_MIN", 2))
    TFIDF_NGRAM_MAX: int = int(_env_float("TFIDF_NGRAM_MAX", 3))
    NER_MIN_ARTICLE_COUNT: int = int(_env_float("NER_MIN_ARTICLE_COUNT", 2))

    ANALYSIS_DATA_DIR: Path = PROJECT_ROOT / "data" / "analysis"

    # Reporter
    REPORT_TOP_ARTICLES: int = int(_env_float("REPORT_TOP_ARTICLES", 5))
    REPORT_TOP_KEYWORDS: int = int(_env_float("REPORT_TOP_KEYWORDS", 10))
    # Recency boost: scores scale by (1 - RECENCY_DECAY)^days_old;
    # set to 0 for a pure keyword-match ranking.
    REPORT_RECENCY_DECAY: float = _env_float("REPORT_RECENCY_DECAY", 0.08)
    REPORT_DIVERSITY_MAX_OVERLAP: int = int(
        _env_float("REPORT_DIVERSITY_MAX_OVERLAP", 3)
    )
    REPORT_USE_LLM_SUMMARY: bool = _env_bool("REPORT_USE_LLM_SUMMARY", True)

    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"

    def ensure_dirs(self) -> None:
        for d in (
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.CLEAN_DATA_DIR,
            self.OUTPUT_DATA_DIR,
            self.ANALYSIS_DATA_DIR,
            self.LLM_CACHE_DIR,
            self.LOG_DIR,
            self.REPORTS_DIR,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
