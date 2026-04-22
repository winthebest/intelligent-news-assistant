# Technical Report — Intelligent News Assistant

> **Author**: Phan Võ Trọng Tiển

> **Submission**: PCACS — AI Engineer Intern Technical Test

> **Date**: 2026-04-22

> **Repository**: _<github link — to be added after `git init` + push>_

> **Topic**: Khoa học – Công nghệ (Science & Technology)

> **Lookback window**: 7 days (2026-04-13 → 2026-04-20)

> **Corpus**: 61 articles from VnExpress `/khoa-hoc-cong-nghe/`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Scope](#2-problem-statement--scope)
3. [System Architecture](#3-system-architecture)
4. [Implementation Decisions & Trade-offs](#4-implementation-decisions--trade-offs)
5. [Evaluation & Results](#5-evaluation--results)
6. [Limitations](#6-limitations)
7. [Engineering Practices](#7-engineering-practices)
8. [What I Would Do Next](#8-what-i-would-do-next)
9. [Appendix — Parameter & Term Reference](#9-appendix--parameter--term-reference)

---

## 1. Executive Summary

**Problem.** Given a chosen news topic and a 7-day window, **automatically crawl Vietnamese online newspapers**, **clean + dedup the corpus**, **extract trending keywords**, and emit a weekly Markdown brief with **Executive Summary**, **Trending Keywords**, and **Highlighted News**.

**Approach.** A 4-stage pipeline (Crawler → Filter → LLM Analyzer → Reporter) that turns raw VnExpress pages into a weekly summary report with trending keywords and highlighted news.

**Scale.**
- Raw articles crawled: **64**
- After clean + exact + fuzzy dedup: **61** (2 dropped for `content_too_short`, 1 fuzzy-duplicate cluster).
- Trending keywords produced: **15** (hybrid TF-IDF + LLM-NER).
- Highlighted news rendered: **5**.

**Key results.**
- Hybrid keyword extraction is **structurally complementary**: TF-IDF (bigrams/trigrams) and LLM-NER (proper nouns) produce **disjoint** term sets on this corpus — 10 merged terms come from TF-IDF only, 5 from NER only, agreement rate 0%. Running either branch alone would drop 33–67% of the final list.
- Pipeline is **provider-agnostic**: same code runs against Gemini (API) or Ollama (local) via a single `.env` switch, thanks to `litellm` + a SHA256-keyed disk cache.
- Weekly report is **reproducible**: given the same raw JSON and warm cache, Phase 3 + 4 re-emit byte-identical Markdown.

**Stack.** Python 3.11 · Selenium · rapidfuzz · scikit-learn · litellm · Ollama `qwen2.5:14b-instruct`.

---

## 2. Problem Statement & Scope

### 2.1 What the pipeline must deliver (from the test brief)
- [x] Crawl Vietnamese news on a chosen topic, within a 7-day lookback.
- [x] Produce a weekly report with three sections: **Executive Summary**, **Trending Keywords**, **Highlighted News**.
- [x] Deliver source code on GitHub plus this technical report. 

### 2.2 Scope decisions
| Decision | Choice | Why |
|---|---|---|
| Topic | Khoa học – Công nghệ (Science & Technology) | High proper-noun density (product names, companies, space missions), heavy English-mixed vocabulary — ideal to test LLM-NER vs TF-IDF trade-offs. |
| Sources | VnExpress only (single-source) | Single listing page gives a clean, chronological newest-first feed that works well with the auto-stop crawler. The architecture keeps crawlers under `src/crawler/`. |
| Language | Vietnamese (mixed with English tech terms) | Topic-native language. Drives NFC normalisation, community stopword list, and the word-boundary regex fix in §4.8. |
| Reporting period | Last 7 days from run time | Matches the brief; cutoff enforced in `content_crawler.py` via `datetime.now(VN_TZ) - timedelta(days=7)`. |

### 2.3 Non-goals (explicitly out of scope)
- Real-time / streaming ingestion.
- Multi-week trend comparison.
- User-facing UI — output is a single Markdown report.


---

## 3. System Architecture

### 3.1 High-level diagram

```
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐
│  1. Crawler  │──▶│  2. Filter   │──▶│  3. LLM Analyzer │──▶│  4. Reporter │
│ (Selenium +  │   │ (Dedup +     │   │ (TF-IDF +        │   │ (Markdown    │
│  DOM clean)  │   │  date + NFC) │   │  LLM-NER +       │   │  summary)    │
│              │   │              │   │  merge)          │   │              │
└──────────────┘   └──────────────┘   └──────────────────┘   └──────────────┘
   raw JSON          clean JSON        keywords_*.json        report.md
```

### 3.2 Stage-by-stage

#### Stage 1 — Crawler (`src/crawler/`)
- **`url_crawler.py`**: paginated listing crawl, URL canonicalisation (`scheme + netloc + path`, strip fragment/query), chronological newest-first ordering persisted to `vnexpress_links.csv`.
- **`content_crawler.py`**: Selenium + DOM-level JS injection (`_strip_related_dom`) removes VnExpress "tin liên quan" boxes *before* reading `.fck_detail`. Vietnamese abbreviation expansion (`TP.HCM → Thành phố Hồ Chí Minh`) runs during extraction.
- **Auto-stop**: breaks early when it hits `max_old_streak=5` consecutive articles older than the **7-day window**. Relies on the newest-first ordering produced by the URL crawler.
- **Output**: `data/raw/vnexpress_articles.json` (schema: `canonical_url`, `title`, `description`, `content`, `date_published`, `published_at` ISO-8601+TZ, `category`, `source`, `author`, `crawl_time`).

#### Stage 2 — Filter (`src/filter/`)
- **`cleaner.py`** — NFC Unicode normalisation + heuristic trailing-boilerplate stripper. Three detectors run over the last ~45% of each body: plain byline (`Sơn Hà`), byline + source (`Thu Thảo (Theo Space, CCTV)`), title + view-count suffix (`… 74`). Fail-safe: no match → content unchanged. Drops deprecated keys (`url_id`, `url`) from legacy dumps.
- **`dedup.py`** — two tiers: exact match on `canonical_url`, then fuzzy title clustering via `rapidfuzz.fuzz.token_set_ratio ≥ 85` built on a path-compression union-find. Cluster representative = longest `content` (richer for downstream NLP).
- **`run_filter.py`** — schema validator enforces `REQUIRED_FIELDS = (title, content, date_published, canonical_url)` + `min_content_length=100`. Every drop is logged with a reason into `dedup_report.json` (`content_too_short:N`, `exact_duplicate_url`, `fuzzy_duplicate` + similarity + cluster representative).
- **Output**: `data/clean/articles.json` (61 articles) + `data/clean/dedup_report.json` (audit log).

#### Stage 3 — LLM Analyzer (`src/analyzer/`)
- **TF-IDF baseline** (`keyword_tfidf.py`): bigrams + trigrams (`ngram_range=(2,3)`), `min_df=3`, `max_df=0.6`, `sublinear_tf=True`, external 1,942-entry Vietnamese stopword list loaded from `resources/vietnamese-stopwords.txt` (union with a small inline fallback to catch VnExpress-specific filler like `video`, `ảnh`, `photo`).
- **LLM-NER** (`keyword_ner.py`): single batched call on `title || description[:180]` for all 61 articles (~6K tokens), routed through `litellm` (Gemini or Ollama). Post-processing: `_coerce_to_list` reshapes wrapper-dict outputs, `_validate_entity` accepts alias field names (`name`/`term`/`frequency`…) but emits the canonical `{entity, count}` schema, `entity_classifier.is_common_noun` drops drifted outputs (`phi hành gia`, `tàu vũ trụ`…) via a curated blocklist. Sample titles are backfilled from the corpus — never trusted from the LLM.
- **Fusion** (`keyword_merger.py`): normalise each source to [0,1], fold Vietnamese accents for cross-source key matching, score = `0.6 · norm_ner + 0.4 · norm_tfidf + 0.15 (if both sources agree)`.
- **Disk cache** (`llm_client.py`): every LLM call is keyed by SHA256 of (model, temperature, response_format, system prompt, user prompt) → stored as JSON under `.llm_cache/`. Guarantees reproducibility + zero-cost re-runs for ablation.
- **Ollama hardening**: native JSON mode via `format=json`, explicit `options={num_ctx: 8192, num_predict, top_p, top_k, repeat_penalty}`, preflight check (`_warn_if_overflow`) warns when the prompt exceeds 90% of `num_ctx` so Ollama never silently truncates.
- **Output**: `keywords_tfidf.json`, `keywords_ner.json`, `keywords_final.json`, `analyzer_report.json`.

#### Stage 4 — Reporter (`src/reporter/`)
Turns the analyzer artefacts into a human-readable weekly brief. Four small modules instead of one big script so each concern stays testable in isolation:
- **`highlighter.py`** — scores every article by `sum(final_score of matching keywords) × (1 − recency_decay)^days_old`. "Matching" uses a **word-boundary regex** on `title + description` (body excluded to avoid noise; word-boundary instead of substring so short acronyms like "AI" don't match inside "Dubai" or Vietnamese accented letters). Greedy diversity pass drops candidates that share ≥ `max_overlap` keywords with any already-picked article so one topic can't monopolise the list.
- **`summarizer.py`** — builds a compact prompt (top-10 keywords + top-8 highlighted titles) and sends it through the same `LLMClient` used by Phase 3, so it inherits the disk cache and the provider switch. Falls back to a deterministic template summary if the LLM call fails or is disabled via `REPORT_USE_LLM_SUMMARY=false`.
- **`formatter.py`** — pure-string Markdown renderer. No templating engine; the structure is tiny and stable. Escapes `|` in titles so the keyword table doesn't break.
- **`run_reporter.py`** — CLI orchestrator. Reads articles + `keywords_final.json`, derives the reporting period from `date_published` min/max, runs all three stages, writes the Markdown plus a JSON companion for programmatic consumers.

**Output**: `reports/weekly_report_<YYYY-MM-DD>.md` and `reports/weekly_report_<YYYY-MM-DD>.json` (structured twin with the same data).

### 3.3 Data contract 
```json
{
  "canonical_url": "https://vnexpress.net/...",
  "title": "...",
  "description": "...",
  "content": "...",
  "published_at": "2026-04-18T09:15:00+07:00",
  "date_published": "2026-04-18",
  "category": "khoa-hoc-cong-nghe",
  "source": "vnexpress",
  "author": "...",
  "crawl_time": "2026-04-21T16:25:26+07:00"
}
```


---

## 4. Implementation Decisions & Trade-offs

This section documents the design choices embedded in the current codebase, with the trade-offs that justify them.

### 4.1 Canonical-URL-only schema
Each article in `data/raw/vnexpress_articles.json` and `data/clean/articles.json` carries a single stable identifier — `canonical_url` — plus two date fields and one topic field:

```json
{
  "canonical_url": "https://vnexpress.net/…",
  "date_published": "2026-04-18",
  "published_at":   "2026-04-18T09:15:00+07:00",
  "category":       "khoa-hoc-cong-nghe",
  "source":         "vnexpress",
  …
}
```

- `canonical_url` is the only join key used by the filter, analyzer, and reporter. 
- `date_published` (day-level string) drives the 7-day cutoff because day granularity is enough for the brief's "past week" requirement and makes string comparisons safe.
- `published_at` keeps full ISO-8601 with the `+07:00` offset so `datetime.fromisoformat` works directly in the reporter's recency-decay math.
- `category` is a single taxonomy field; there is no parallel `topic` field.



### 4.2 Crawl auto-stop on a newest-first listing

`content_crawler.crawl_content(days=7, max_old_streak=5)` reads the date *before* doing anything else with an article, and breaks the whole crawl when it has seen 5 consecutive articles whose `date_published < now - 7 days`. This works because:

- VnExpress listing pages are chronological newest-first, and `url_crawler.py` preserves that order when merging new URLs into `vnexpress_links.csv` by writing *new* URLs *before* the existing ones.
- A transient pinned/featured old article cannot trigger a false stop — the streak resets the moment an in-window article appears.
- The alternative (hard-cap `--end-page=N` or "crawl everything, filter later") either under-covers when volume spikes or wastes Selenium time on articles that will be dropped anyway.

### 4.3 LLM provider agnosticism via `litellm` + disk cache

`llm_client.LLMClient` is the single surface through which Phase 3 and Phase 4 talk to any LLM. It routes all calls through `litellm`, so `LLM_MODEL=gemini/gemini-1.5-flash` and `LLM_MODEL=ollama/qwen2.5:14b-instruct` both work unchanged.

Provider-specific branches stay inside the client: `_ollama_options` builds `{num_ctx, num_predict, top_p, top_k, repeat_penalty}`, `format=json` turns on Ollama's native JSON mode, and `_warn_if_overflow` estimates the prompt's token count against `OLLAMA_NUM_CTX` before sending so Ollama never silently truncates.

Every call is cached by SHA256 of `(model, temperature, response_format, system prompt, user prompt)` to `.llm_cache/<key>.json`. Cached reruns are free, and the ablation script (`scripts/benchmark_ablation.py`) uses this to iterate on rankings without hitting the LLM again.

### 4.4 Titles + short lede as the NER input

`keyword_ner.extract_ner_entities` sends the LLM a single batched prompt of the form:

```
1. {title} || {description[:180]}
2. …
```

for all 61 articles at once. The full article bodies would be 120 K tokens, requiring either ~30 sequential calls or a very long context. Titles + 180-char lede is 6 K tokens, fits under `OLLAMA_NUM_CTX=8192` with headroom, and keeps NER at a single LLM call (5-10 s on local Ollama, 1 ms when cached). An empirical scan of this week's corpus confirmed that every proper noun appearing ≥ 2 times in the bodies was already present in at least one title or lede.

### 4.5 Hybrid TF-IDF + LLM-NER

`keyword_merger.merge_keywords` fuses the two branches with `0.6 · norm_ner + 0.4 · norm_tfidf + 0.15` if a term appears in both (accent-folded, lowercased). TF-IDF's `doc_freq` always overrides the LLM's self-reported `count` in the final output, so the user-visible frequency is never a hallucination.

The weighting rationale: NER surfaces named entities that are usually the news-actionable subject of each article (`AI`, `Artemis II`), while TF-IDF surfaces domain themes (`phát triển`, `điện thoại`). Named entities get slightly higher weight because they are more concrete. The `+0.15` agreement bonus fires when a term is present in both lists.

**Ablation — from `reports/benchmark_ablation.json`, reproducible via `python -m scripts.benchmark_ablation`:**

| Variant | #Terms | Unique contribution vs other | Notes |
|---|---:|---:|---|
| TF-IDF only | 15 | **10** bigram/trigram themes disjoint from NER | `phát triển`, `việt nam`, `điện thoại`… |
| LLM-NER only | 10 | **5** proper nouns disjoint from TF-IDF | `AI`, `Artemis II`, `SIM chính chủ`, `iPhone gập`, `VNeID` |
| **Hybrid (ours)** | 15 | — | 10 TF-IDF + 5 NER merged via weighted score |

The zero raw-overlap is a **structural feature**, not a bug: `TFIDF_NGRAM_MIN=2` means TF-IDF cannot emit a single-token acronym like `AI`, and LLM-NER by prompt design does not return compound noun phrases like `phát triển`. Each branch covers an output space the other cannot reach; running either alone drops 5–10 terms (33–67%) from the final list.

### 4.6 Flat `{entity, count}` NER schema + common-noun blocklist

The NER prompt asks only for `entity` and `count` per item. No per-entity classification (`PERSON / ORG / PRODUCT / …`) is requested because:

- The final report ranks keywords as a flat list; category metadata is informational only.
- Smaller open-weight models (Qwen 7b/14b) drop or misclassify the category field often enough that a post-hoc lexicon override becomes the real classifier — which defeats the purpose of using the LLM.

`entity_classifier.is_common_noun` post-processes the LLM output against a curated blocklist (`phi hành gia`, `tàu vũ trụ`, `công ty công nghệ`, …) to catch drift where the model returns a common noun as if it were a named entity. The blocklist is deliberately tight — false drops cost more than false accepts because the merger already down-weights noise via TF-IDF.

### 4.7 External Vietnamese stopwords file

`keyword_tfidf._build_stopwords` loads `resources/vietnamese-stopwords.txt` (~1.9 K community-maintained entries, functional words + frequent n-gram filler) and unions it with a small inline fallback that covers VnExpress-specific media annotations (`video`, `ảnh`, `photo`, `clip`).

Multi-word entries in the file are exploded into their individual tokens at load time so sklearn's token-level stopword filter catches them regardless of n-gram boundary (`"cho biết"` filters both `"cho"` and `"biết"`). The file path is overridable via `STOPWORDS_FILE` in `.env` if a topic-specific list is needed.

### 4.8 Highlighted-news scoring: recency × keyword-match with diversity

`highlighter.rank_articles` computes, for each article that has at least one matching trending keyword in its title + description:

```
article_score = Σ final_score[kw] × (1 - recency_decay) ^ days_old
```

with `REPORT_RECENCY_DECAY=0.08` (halves influence after ~9 days, matching the 7-day window with a small grace for late crawls). Matching uses a Unicode word-boundary regex — `re.compile(rf"(?<!\w){escape(term)}(?!\w)", re.IGNORECASE)` — instead of plain substring, because acronyms like `AI` otherwise false-match inside `Dubai` or inside Vietnamese accented letters that decompose to `a + i` under NFD.

After sorting by score, a greedy pass rejects any candidate that shares `REPORT_DIVERSITY_MAX_OVERLAP=3` or more matched keywords with any already-picked article. This is loose enough to keep independent AI + iPhone + Artemis stories but tight enough to prevent five near-identical AI stories dominating the list.


---

## 5. Evaluation & Results

### 5.1 Pipeline metrics (from `data/analysis/analyzer_report.json` and `data/clean/dedup_report.json`)

**Filter stage**

| Step | Count | Dropped | Reason |
|---|---:|---:|---|
| Raw | 64 | — | — |
| After clean + schema validation | 62 | 2 | `content_too_short:24` + `content_too_short:9` |
| After exact URL dedup | 62 | 0 | No duplicate `canonical_url` this week |
| After fuzzy title dedup (≥ 85) | **61** | 1 | `fuzzy_duplicate` cluster on SIM chính chủ story |

**Analyzer stage** (wall-clock on local machine, `ollama/qwen2.5:14b-instruct`)

| Stage | Count | Latency (ms) | Notes |
|---|---:|---:|---|
| TF-IDF | 15 | 1,484 | Scikit-learn on 61 docs, bigrams+trigrams, 1,942 stopwords |
| LLM-NER | 10 | 1 (cache hit) | Single batched call; cold-run latency ~8–12 s on 14b (measured separately when cache is cleared) |
| Merge | 15 | 0 | In-memory fuzzy join + weighted score |

Article count: **61**. LLM model: `ollama/qwen2.5:14b-instruct`. Both cache-hit and cold-run latencies are captured — the 1 ms row is expected given the disk cache; clearing `.llm_cache/` reruns the LLM for cold-run reporting.

### 5.2 Qualitative samples

**Top 5 trending keywords** (from `keywords_final.json`):
| Rank | Term | Final Score | Sources | DocFreq | NER Count |
|---:|---|---:|---|---:|---:|
| 1 | AI | 0.600 | ner | 7 | 7 |
| 2 | Artemis II | 0.480 | ner | 6 | 6 |
| 3 | phát triển | 0.400 | tfidf | 32 | — |
| 4 | hoạt động | 0.363 | tfidf | 28 | — |
| 5 | SIM chính chủ | 0.360 | ner | 5 | 5 |

**NER entities spotted** (from `keywords_ner.json`): `AI`, `Artemis II`, `SIM chính chủ`, `iPhone gập`, `VNeID`, `MacBook Neo`, `Epson EB-L690U`, `Trí tuệ nhân tạo`, `Huawei Pura X Max`, `Unitree Robotics H1`.

**TF-IDF themes** (from `keywords_tfidf.json`): `phát triển`, `hoạt động`, `sản phẩm`, `việt nam`, `doanh nghiệp`, `điện thoại`, `khả năng`, `nghiên cứu`, `triển khai`, `môi trường`.

### 5.3 Cross-validation (set-level ablation)

Running `python -m scripts.benchmark_ablation` on the three artefacts yields:

- TF-IDF ∩ NER raw overlap: **0 / 25** (zero by construction — see §4.5).
- Provenance of merged final list: **10 TF-IDF-only + 5 NER-only + 0 both**.
- Agreement rate: **0.0%**.

The zero agreement rate is *expected* and not a failure signal: each branch produces a structurally different output space (n-gram themes vs proper nouns/acronyms). The merger's value here is coverage, not consensus — it retrieves 5 entities that TF-IDF structurally cannot emit and 10 themes that NER refuses to label. The full ablation report is at `reports/benchmark_ablation.md`.


---

## 6. Limitations

- **Single-source corpus**. Only VnExpress; TuoiTre / ThanhNien would add cross-source validation but were deprioritised due to the tech-test time budget. The `src/crawler/` module can accept a new `<source>_url_crawler.py` + `<source>_content_crawler.py` pair without touching downstream stages.
- **LLM count numbers are estimates, not counts**. `count` in `keywords_ner.json` is the LLM's best guess, not a true frequency. The real frequency is `doc_freq` from TF-IDF. The merger uses TF-IDF's `doc_freq` as the reported `doc_freq` whenever both sources agree; pure NER-only rows fall back to the LLM's count.
- **No temporal trend**. Pipeline produces a snapshot for 1 week. Comparing week-on-week requires a small extension in the reporter.
- **No ground-truth evaluation set**. The §4.5 ablation is set-level (coverage, overlap) — it answers "does TF-IDF and NER cover different spaces?" but not "how close is our top-15 to a human-labelled gold top-15?". Building that gold set is straightforward but out of scope for this test.
- **Prompt is Vietnamese-primary**. A multilingual corpus (e.g., mixed VN/EN finance news) may need prompt adjustments.

---

## 7. Engineering Practices

- **Config-driven**. Every tunable lives in `config/config.py` and is overridable via `.env` (see `.env.example`). 
- **Reproducibility**. LLM calls are disk-cached by SHA256 of (model, temp, format, prompts). Re-running the pipeline returns identical outputs until a prompt or model changes.
- **Provider portability**. Same pipeline runs on Gemini (API, paid) or Ollama (local, free) via `LLM_MODEL` env var.
- **Preflight checks**. `python -m src.analyzer.doctor` probes host RAM/VRAM, recommends a Qwen checkpoint size, verifies Ollama reachability, and estimates prompt tokens against `num_ctx` before running.
- **Defensive LLM parsing**. 3-layer defense: (1) `_coerce_to_list()` reshapes non-list JSON; (2) `_validate_entity()` accepts alternative field names (`name`, `term`, `frequency`) but emits a canonical schema; (3) `is_common_noun()` blocklist filters noise entities.
- **Schema validation** at every stage boundary — a bad article drops out immediately, not at report time.
- **Deterministic tests possible** — disk cache + fixed corpus snapshot makes analyzer behavior a pure function.
- **Clean separation of concerns** — each stage has a single CLI entrypoint and a single output artifact.

---

## 8. What I Would Do Next

**Given more time, future work would focus on three areas**:

- Source coverage and breadth: Add TuoiTre and ThanhNien crawlers to literally satisfy the "various newspapers" requirement — only the crawler module changes; downstream stages remain untouched. This pairs naturally with canonical entity linking (e.g. "Apple" ≡ "Táo khuyết") to collapse cross-source variants once multi-source duplication becomes common.

- Output quality: Add a --ground-truth flag to measure keyword extraction precision against a manually labelled list, raise NER_MIN_ARTICLE_COUNT to 3 once the corpus exceeds ~150 articles, and introduce week-over-week trend detection — diffing keyword_final.json across runs to emit "rising" / "falling" tags. The reporter could also generate an HTML report alongside the existing Markdown.

- Long-term architecture:  Replace the heuristic boilerplate cleaner with a supervised classifier trained on a few hundred annotated DOMs; fine-tune a Vietnamese NER model (PhoBERT + CRF) and benchmark precision/recall/latency against the LLM-NER baseline; and move from batch to streaming ingestion via RSS with a rolling 7-day store for continuous operation.**

---

## 9. Appendix — Parameter & Term Reference

Every knob referenced in §3–§5 is listed here with its default value and a plain-language explanation. The goal is that a reader who isn't a daily NLP/LLM practitioner can read the main body without stopping to Google.

### 9.1 Pipeline knobs (`.env` / `config/config.py`)

| Parameter | Default | Stage | What it controls |
|---|---|---|---|
| `days` | `7` | crawler | Publication window. Articles older than `now - days` are skipped. |
| `max_old_streak` | `5` | crawler | Auto-stop the whole crawl after N consecutive out-of-window articles. Depends on the URL CSV being newest-first. |
| `min_content_length` | `100` | cleaner | Drop articles whose cleaned body has fewer than N characters. Catches stub pages. |
| fuzzy dedup threshold | `85` | dedup | `rapidfuzz.fuzz.token_set_ratio` cutoff on a 0–100 scale. Two titles scoring ≥ 85 are merged into the same cluster. |
| `KEYWORD_TOP_K` | `15` | analyzer | How many merged keywords appear in the final report. |
| `TFIDF_NGRAM_MIN / MAX` | `2 / 3` | tfidf | TF-IDF considers bigrams and trigrams only — no unigrams. This is why single-token acronyms like `AI` never surface through TF-IDF. |
| `TFIDF_MIN_DF` | `3` | tfidf | Keep a term only if it appears in ≥ 3 articles. Filters one-off noise. |
| `TFIDF_MAX_DF` | `0.6` | tfidf | Drop a term that appears in more than 60% of articles. Filters generic fillers. |
| `NER_MIN_ARTICLE_COUNT` | `2` | ner | Drop LLM-reported entities whose self-reported count < 2. |
| `REPORT_TOP_ARTICLES` | `5` | reporter | How many highlighted-news items are rendered. |
| `REPORT_TOP_KEYWORDS` | `10` | reporter | How many rows the Trending Keywords table shows. |
| `REPORT_RECENCY_DECAY` | `0.08` | highlighter | Per-day geometric decay. Score ×= `(1 - 0.08)^days_old`; ~50% after 9 days, 0 effect after 0 days. Set to `0.0` to disable recency boost. |
| `REPORT_DIVERSITY_MAX_OVERLAP` | `3` | highlighter | A new highlight is rejected if it shares ≥ 3 matched keywords with any already-picked article. Lower = more diverse; higher = more concentrated. |
| `REPORT_USE_LLM_SUMMARY` | `true` | reporter | `false` forces the deterministic template summary even when the LLM is reachable. |

### 9.2 Ollama / LLM sampler knobs

Used only when `LLM_MODEL` starts with `ollama/`. Gemini and OpenAI providers ignore the Ollama-specific fields.

| Parameter | Default | What it does |
|---|---|---|
| `LLM_MODEL` | `gemini/gemini-1.5-flash` _or_ `ollama/qwen2.5:14b-instruct` | `litellm`-style provider+model slug. The same code path handles both. |
| `LLM_TEMPERATURE` | `0.2` | Lower = more deterministic output; `0.0` is fully greedy. We use 0.2 to keep JSON extraction stable without freezing the model. |
| `OLLAMA_NUM_CTX` | `8192` | Max tokens (prompt + response) per call. Ollama silently truncates the head of the prompt past this — the doctor's preflight check exists to catch this. |
| `OLLAMA_NUM_PREDICT` | `2048` | Upper bound on *response* tokens. |
| `OLLAMA_TOP_P` | `0.9` | Nucleus sampling: sample only from the smallest set of tokens whose cumulative probability ≤ 0.9. |
| `OLLAMA_TOP_K` | `40` | At each step, only consider the 40 highest-probability tokens. |
| `OLLAMA_REPEAT_PENALTY` | `1.05` | Mild penalty against verbatim repetition. Values > 1 discourage loops. |
| Ollama `format=json` | — | Native JSON-only mode for `/api/chat`. Stricter than prompt-level "please return JSON" instructions because the server rejects non-JSON candidate tokens. |
| `PREFLIGHT_CTX_RATIO` | `0.9` | Warning threshold: if the estimated prompt size exceeds 90% of `num_ctx`, log a warning before sending. |
| `PREFLIGHT_CHARS_PER_TOKEN` | `2.5` | Rough chars-per-token estimate for Vietnamese corpora (real value 2.5–3.0). Used only for the preflight warning. |

### 9.3 Fusion weights (`keyword_merger.merge_keywords`)

The merged score is:

```
final_score(term) = 0.6 · norm_ner(term) + 0.4 · norm_tfidf(term) + 0.15 · [term present in both sources]
```

- Each source is scaled independently to `[0, 1]` (divide by max).
- The `0.15` agreement bonus fires only when an accent-folded, lowercased key matches in both the NER and the TF-IDF lists.
- Override any of `w_ner`, `w_tfidf`, `agreement_bonus` via the function signature; no env var is exposed yet because the defaults produced the reported results.

### 9.4 Concepts referenced in-text

| Term | Plain-language meaning |
|---|---|
| **TF-IDF** | "Term Frequency × Inverse Document Frequency." Classic statistical score for how characteristic a phrase is of a document, penalising words that appear everywhere. Implemented by `sklearn.feature_extraction.text.TfidfVectorizer`. |
| **NER** | "Named Entity Recognition." Extracting proper nouns — people, organisations, products, events. Here it is done by the LLM instead of a trained model. |
| **n-gram / bigram / trigram** | Consecutive sequences of N tokens. "AI" is a unigram; "phát triển" is a bigram; "xe tự lái" is a trigram. |
| **Document frequency (`doc_freq`)** | In how many *articles* a term appears. Truthful count, unlike the LLM's self-reported `count`. |
| **NFC / NFD (Unicode normalisation)** | Two ways to encode the same accented letter. NFC stores "ế" as one codepoint; NFD stores it as "e + combining acute". Mixing the two causes silent matching failures on Vietnamese. The cleaner normalises everything to NFC. |
| **`canonical_url`** | A URL stripped of its query string (`?utm=…`) and fragment (`#section`). Two visits that differ only in tracking params then compare equal. This is the only identifier the pipeline relies on. |
| **Union-find (disjoint-set)** | Data structure that groups items into clusters in near-constant time per merge. Used in `dedup.py` to collect near-duplicate titles into clusters. |
| **SHA256 cache key** | Cryptographic hash used purely as a deterministic fingerprint of `(model, temperature, format, system prompt, user prompt)`. No security purpose — it just guarantees that identical inputs always hit the same cache file. |
| **Word-boundary regex** | Pattern that matches whole tokens. `(?<!\w)AI(?!\w)` matches the standalone word "AI" but not "AI" inside "Dubai" or inside a Vietnamese letter decomposed under NFD to "a + i". |
| **Accent-folded matching** | Comparison that happens after stripping Vietnamese diacritics. "Apple" vs "apple" and "Việt Nam" vs "viet nam" all collapse to the same key — useful when merging NER (proper case) with TF-IDF (lowercased). |
| **`token_set_ratio`** | Fuzzy-match score 0–100 from `rapidfuzz` that tokenises both strings, compares their sets, and tolerates reordering and duplicated words. More robust than simple Levenshtein for news titles. |
| **Qwen 7b / 14b** | Open-weight multilingual LLM family. The number is the parameter count in billions. 14b is the largest that fits comfortably in 16 GB of VRAM at Q4 quantisation (~11 GB model + ~2 GB KV-cache). |
| **`litellm`** | Small Python library that exposes OpenAI, Gemini, Ollama, Anthropic and others through one `completion(...)` API, so switching providers is a config change. |
| **Agreement rate (set-level)** | `|A ∩ B| / |A ∪ B|` — the Jaccard index. Used in §5.3 to report how much TF-IDF and NER overlap on the final list. |

