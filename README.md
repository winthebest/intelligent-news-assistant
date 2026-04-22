# Intelligent News Assistant

A 4-stage pipeline that crawls Vietnamese tech news from **VnExpress**, cleans & dedups the corpus, extracts trending keywords (**TF-IDF + LLM-NER** hybrid), and emits a weekly Markdown report with **Executive Summary**, **Trending Keywords**, and **Highlighted News**.

**Stack.** Python 3.11 · Selenium · rapidfuzz · scikit-learn · litellm · Ollama (`qwen2.5:14b-instruct`) or Gemini.

**Docs.** See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for the full architecture, design decisions, ablation study, and evaluation.

---

## Quick start

```bash
# 1. Install
conda create -n pca_test python=3.11 -y
conda activate pca_test
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# edit .env — set LLM_MODEL to ollama/... or gemini/...

# 3. (Optional) Preflight: RAM/VRAM probe, Ollama reachability, prompt-token budget
python -m src.analyzer.doctor

# 4. Crawl — URLs first (newest-first), then content with 7-day window + auto-stop
python -m src.crawler.url_crawler    --start-page 1 --end-page 5
python -m src.crawler.content_crawler --days 7 --max-old-streak 5

# 5. Filter — schema validation + exact URL dedup + fuzzy title dedup
python -m src.filter.run_filter

# 6. Analyze — TF-IDF + LLM-NER + weighted merge
python -m src.analyzer.run_analyzer

# 7. Report — Markdown + JSON weekly brief
python -m src.reporter.run_reporter
# or without LLM executive summary (offline / deterministic fallback):
#   python -m src.reporter.run_reporter --no-llm

# 8. (Optional) Set-level ablation: TF-IDF vs NER vs merged
python -m scripts.benchmark_ablation
```

---

## Configuration

All tunables live in `config/config.py` and are overridable via `.env`. Most-used settings:

```dotenv
# LLM provider — swap ollama ↔ gemini with no code change
LLM_MODEL=ollama/qwen2.5:14b-instruct
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_NUM_CTX=8192
LLM_TEMPERATURE=0.2
LLM_CACHE_ENABLED=true

# TF-IDF baseline
TFIDF_MIN_DF=3
TFIDF_MAX_DF=0.6
TFIDF_NGRAM_MIN=2
TFIDF_NGRAM_MAX=3
STOPWORDS_FILE=resources/vietnamese-stopwords.txt

# Keyword fusion
KEYWORD_TOP_K=15
NER_MIN_ARTICLE_COUNT=2

# Phase 4 — reporter
REPORT_TOP_ARTICLES=5
REPORT_TOP_KEYWORDS=10
REPORT_RECENCY_DECAY=0.08
REPORT_DIVERSITY_MAX_OVERLAP=3
REPORT_USE_LLM_SUMMARY=true
```

See [`.env.example`](.env.example) for the complete list with inline comments.

---

## Outputs

| Stage | Path | Content |
|---|---|---|
| Crawler | `data/raw/vnexpress_links.csv` | Canonical URLs, newest-first |
| Crawler | `data/raw/vnexpress_articles.json` | Full articles (title / description / content / dates / …) |
| Filter | `data/clean/articles.json` | Deduped + NFC-normalised + schema-validated |
| Filter | `data/clean/dedup_report.json` | Audit log of every drop |
| Analyzer | `data/analysis/keywords_tfidf.json` | Statistical baseline |
| Analyzer | `data/analysis/keywords_ner.json` | LLM-NER proper nouns |
| Analyzer | `data/analysis/keywords_final.json` | Merged top-K |
| Analyzer | `data/analysis/analyzer_report.json` | Run metadata + latencies |
| Reporter | `reports/weekly_report_<date>.md` | **Human-facing weekly brief** |
| Reporter | `reports/weekly_report_<date>.json` | Structured twin |
| Benchmark | `reports/benchmark_ablation.{md,json}` | TF-IDF vs NER vs merged |

A rendered sample: [`reports/weekly_report_2026-04-22.md`](reports/weekly_report_2026-04-22.md).

---

## Repository layout

```
Test_PCA/
├── config/                 # Settings class + .env loader
├── prompts/                # LLM prompt templates (ner_titles.txt, executive_summary.txt)
├── resources/              # Static lexicons (vietnamese-stopwords.txt, ~1.9K entries)
├── src/
│   ├── crawler/            # url_crawler.py, content_crawler.py, crawl_pipeline.py
│   ├── filter/             # cleaner.py, dedup.py, run_filter.py
│   ├── analyzer/           # TF-IDF + LLM-NER + merger + llm_client + doctor
│   └── reporter/           # highlighter + summarizer + formatter + run_reporter
├── scripts/
│   └── benchmark_ablation.py   # set-level ablation TF-IDF vs NER vs merged
├── data/
│   ├── raw/                # vnexpress_links.csv + vnexpress_articles.json
│   ├── clean/              # articles.json + dedup_report.json
│   └── analysis/           # keywords_{tfidf,ner,final}.json + analyzer_report.json
├── reports/                # weekly_report_<date>.{md,json} + benchmark_ablation.{md,json}
├── .llm_cache/             # SHA256-keyed LLM response cache
├── .env.example
├── requirements.txt
├── README.md               # ← you are here
└── TECHNICAL_REPORT.md     # architecture, decisions, ablation, evaluation
```

---

## Requirements

- **Python** 3.11
- **Chrome** + matching `chromedriver` (for Selenium)
- **Ollama** v0.3+ — if using local LLM; otherwise a `GEMINI_API_KEY` works the same
- **Hardware** — ~16 GB RAM is enough for `qwen2.5:7b-instruct`; `14b` wants a GPU with ≥ 10 GB VRAM or 32 GB system RAM for CPU inference. `python -m src.analyzer.doctor` probes the host and recommends a checkpoint size.

---

## Reproducibility

Every LLM call is cached under `.llm_cache/<sha256>.json`, keyed by `(model, temperature, response_format, system prompt, user prompt)`. Given the same raw corpus and warm cache, Phase 3 and Phase 4 re-emit byte-identical output — useful for ablation and for CI.

---

## Crawling from scratch

The repo ships a pre-crawled corpus (`data/raw/…`) so graders can go straight to Phase 2–4. Both crawlers are **incremental by design**:

- `url_crawler.py` reads the existing `vnexpress_links.csv` and appends only URLs it hasn't seen (canonical-URL deduplication).
- `content_crawler.py` reads the existing `vnexpress_articles.json` and only fetches URLs not already present.

That means re-running them on top of existing artifacts just **tops up** the corpus, not replaces it. To force a full fresh crawl, clear the stale artifacts first:

### What to delete

| Path | Effect of deleting |
|---|---|
| `data/raw/vnexpress_links.csv` | Forces URL rediscovery from VnExpress listing pages |
| `data/raw/vnexpress_articles.json` | Forces content re-fetch for every URL |
| `data/clean/*.json` | Removes stale Phase 2 output (articles.json, dedup_report.json) |
| `data/analysis/*.json` | Removes stale Phase 3 output (4 keyword files) |
| `reports/weekly_report_*.{md,json}` | Removes stale Phase 4 report |
| `reports/benchmark_ablation.{md,json}` | Removes stale ablation report |
| `.llm_cache/` *(optional)* | Forces cold LLM calls. Keep it if you only want to re-verify Phase 3–4 determinism without paying again for LLM tokens. |

### One-liners

**PowerShell (Windows — this repo's default shell)**:

```powershell
# Full clean slate (keeps .llm_cache — drop it too if you want cold LLM calls)
Remove-Item -Force -ErrorAction SilentlyContinue `
    data\raw\vnexpress_links.csv, `
    data\raw\vnexpress_articles.json
Remove-Item -Force -Recurse -ErrorAction SilentlyContinue `
    data\clean\*.json, `
    data\analysis\*.json, `
    reports\weekly_report_*.md, `
    reports\weekly_report_*.json, `
    reports\benchmark_ablation.md, `
    reports\benchmark_ablation.json

# Optional: also drop LLM cache to force cold calls
# Remove-Item -Force -Recurse .llm_cache
```

**Bash (Linux / macOS / Git Bash)**:

```bash
rm -f data/raw/vnexpress_links.csv data/raw/vnexpress_articles.json
rm -f data/clean/*.json data/analysis/*.json
rm -f reports/weekly_report_*.md reports/weekly_report_*.json
rm -f reports/benchmark_ablation.md reports/benchmark_ablation.json
# optional:
# rm -rf .llm_cache
```

Then re-run the full pipeline from [Quick start step 4](#quick-start) onwards:

```bash
python -m src.crawler.url_crawler    --start-page 1 --end-page 5
python -m src.crawler.content_crawler --days 7 --max-old-streak 5
python -m src.filter.run_filter
python -m src.analyzer.run_analyzer
python -m src.reporter.run_reporter
python -m scripts.benchmark_ablation   # optional
```

> **Tip.** If you only want to re-crawl content (e.g., the cleaner was improved) but keep the URL list, delete **only** `vnexpress_articles.json` and skip `url_crawler`.

---

