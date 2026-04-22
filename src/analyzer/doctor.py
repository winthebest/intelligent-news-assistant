"""``python -m src.analyzer.doctor`` — one-shot sanity check for the LLM
setup. Prints host capacity (RAM/GPU), recommends an Ollama checkpoint,
probes the configured endpoint for reachability and installed models, and
preflights the NER prompt against ``num_ctx``. Run once after checkout to
surface silent-truncation / dead-endpoint failure modes before a real
analysis spends minutes producing nonsense."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from config import settings

from .keyword_ner import _format_articles_block, _load_prompt_template
from .resources import probe_host

logger = logging.getLogger(__name__)


_OK = "[OK]"
_WARN = "[WARN]"
_FAIL = "[FAIL]"


def _hr(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def _ollama_tags() -> list[str] | None:
    """Return the list of model tags installed on the Ollama server, or None."""
    import urllib.error
    import urllib.request

    url = settings.OLLAMA_API_BASE.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        logger.debug("Ollama /api/tags probe failed: %s", e)
        return None
    models = data.get("models", [])
    return [m.get("name", "") for m in models if m.get("name")]


def _check_ollama(host_summary: Dict[str, Any]) -> None:
    _hr("Ollama server")
    print(f"  Endpoint        : {settings.OLLAMA_API_BASE}")

    tags = _ollama_tags()
    if tags is None:
        print(f"  {_FAIL} Not reachable. Start the server with `ollama serve`.")
        host_summary["ollama_reachable"] = False
        return

    host_summary["ollama_reachable"] = True
    print(f"  {_OK} Reachable. {len(tags)} model(s) installed:")
    for t in sorted(tags):
        print(f"       - {t}")

    # Is the currently configured model present?
    if settings.LLM_MODEL.startswith("ollama/"):
        wanted = settings.LLM_MODEL[len("ollama/") :]
        # Accept exact tag OR a quantisation variant ("qwen2.5:14b-instruct"
        # matches "qwen2.5:14b-instruct-q5_K_M").
        installed = any(t == wanted or t.startswith(wanted + ":") for t in tags)
        if installed:
            print(f"  {_OK} LLM_MODEL '{wanted}' is installed.")
        else:
            print(f"  {_FAIL} LLM_MODEL '{wanted}' NOT installed.")
            print(f"       Fix: ollama pull {wanted}")
        host_summary["ollama_model_installed"] = installed


def _preflight_ner_prompt() -> None:
    _hr("NER prompt preflight (Ollama only)")
    clean_path = settings.CLEAN_DATA_DIR / "articles.json"
    if not clean_path.exists():
        print(f"  {_WARN} {clean_path} not found; skip.")
        return
    try:
        articles: List[Dict[str, Any]] = json.loads(
            clean_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as e:
        print(f"  {_WARN} Could not read articles.json: {e}")
        return

    template = _load_prompt_template()
    block = _format_articles_block(articles)
    prompt = template.format(
        n=len(articles),
        min_count=settings.NER_MIN_ARTICLE_COUNT,
        top_k=settings.KEYWORD_TOP_K,
        articles_block=block,
    )
    est_tokens = int(len(prompt) / settings.PREFLIGHT_CHARS_PER_TOKEN)
    budget = int(settings.OLLAMA_NUM_CTX * settings.PREFLIGHT_CTX_RATIO)

    print(f"  Articles in corpus : {len(articles)}")
    print(f"  Prompt length      : {len(prompt):,} chars")
    print(
        f"  Estimated tokens   : ~{est_tokens:,} "
        f"(using {settings.PREFLIGHT_CHARS_PER_TOKEN} chars/token)"
    )
    print(
        f"  OLLAMA_NUM_CTX     : {settings.OLLAMA_NUM_CTX:,} "
        f"(budget = {budget:,} @ {int(settings.PREFLIGHT_CTX_RATIO*100)}%)"
    )

    if not settings.LLM_MODEL.startswith("ollama/"):
        print(f"  {_OK} Gemini/OpenAI provider: context window is not a concern.")
        return

    if est_tokens <= budget:
        margin = budget - est_tokens
        print(f"  {_OK} Fits comfortably (margin: ~{margin:,} tokens).")
    else:
        deficit = est_tokens - budget
        suggested = max(settings.OLLAMA_NUM_CTX * 2, est_tokens * 2)
        for size in (4096, 8192, 12288, 16384, 24576, 32768):
            if size >= suggested:
                suggested = size
                break
        print(
            f"  {_FAIL} Prompt exceeds budget by ~{deficit:,} tokens. "
            f"Ollama WILL truncate silently."
        )
        print(f"       Fix: set OLLAMA_NUM_CTX={suggested} in .env")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    _hr("Host capacity")
    cap = probe_host()
    print(
        f"  RAM   : {cap.ram_gb:.1f} GB" if cap.ram_gb else "  RAM   : unknown"
    )
    if cap.gpu_vram_gb:
        print(f"  GPU   : {cap.gpu_name} ({cap.gpu_vram_gb:.1f} GB VRAM)")
    else:
        print("  GPU   : none detected (Ollama will run on CPU)")
    recommended = cap.recommend_ollama_model()
    print(f"  Recommended Ollama model: {recommended}")

    _hr("LLM configuration")
    print(f"  LLM_MODEL         : {settings.LLM_MODEL}")
    print(f"  LLM_TEMPERATURE   : {settings.LLM_TEMPERATURE}")
    if settings.LLM_MODEL.startswith("ollama/"):
        print(f"  OLLAMA_NUM_CTX    : {settings.OLLAMA_NUM_CTX}")
        print(f"  OLLAMA_NUM_PREDICT: {settings.OLLAMA_NUM_PREDICT}")
        print(f"  OLLAMA_TOP_P      : {settings.OLLAMA_TOP_P}")
    elif settings.LLM_MODEL.startswith("gemini/"):
        key = "set" if settings.GEMINI_API_KEY else "MISSING"
        print(f"  GEMINI_API_KEY    : {key}")

    host_summary: Dict[str, Any] = {}
    if settings.LLM_MODEL.startswith("ollama/"):
        _check_ollama(host_summary)

    _preflight_ner_prompt()

    _hr("Done")
    print("  Use the recommendations above to tune .env, then run:")
    print("    python -m src.analyzer.run_analyzer")
    print()


if __name__ == "__main__":
    main()
