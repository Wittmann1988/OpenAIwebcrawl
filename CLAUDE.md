# Project Architecture — Web Q&A with Embeddings

## Overview

RAG (Retrieval-Augmented Generation) pipeline that crawls a website, generates
vector embeddings for the content, and answers natural-language questions using
semantic search + OpenAI completions.

**Target site:** `openai.com` (configurable via `domain` / `full_url` at top of script)

## File Layout

```
OpenAIwebcrawl/
├── web-qa.py            # Main pipeline script (single-file, 13 steps)
├── web-qa.ipynb         # Jupyter notebook version (same logic, cell-per-step)
├── requirements.txt     # Pinned Python 3.11 dependencies
├── text/<domain>/       # [generated] Raw crawled pages as .txt files
├── processed/
│   ├── scraped.csv      # [generated] Cleaned text with filenames
│   └── embeddings.csv   # [generated] Text chunks + ada-002 embeddings
└── .claude/
    ├── settings.json    # Claude Code hook config
    └── hooks/
        └── session-start.sh  # Installs deps + flake8 on web sessions
```

## Pipeline Stages (web-qa.py)

The script is organized as sequential numbered steps. Each step depends on the
output of the previous one.

| Step | Lines     | Function / Purpose                                         |
|------|-----------|------------------------------------------------------------|
| 1    | 1-28      | Imports, constants, `domain`/`full_url` config             |
| 1b   | 31-43     | `HyperlinkParser` — HTMLParser subclass to extract hrefs   |
| 2    | 50-71     | `get_hyperlinks(url)` — fetch page, return all links       |
| 3    | 78-108    | `get_domain_hyperlinks(domain, url)` — filter same-domain  |
| 4    | 115-168   | `crawl(url)` — BFS crawler, saves pages to `text/<domain>/`|
| 5    | 175-180   | `remove_newlines(serie)` — pandas Series text cleaner      |
| 6    | 188-206   | Load .txt files → DataFrame → `processed/scraped.csv`      |
| 7    | 213-222   | Tokenize with tiktoken `cl100k_base`, count tokens per row |
| 8    | 228-267   | `split_into_many(text, max_tokens=500)` — chunk long texts |
| 9    | 291-293   | Rebuild DataFrame from chunks, re-count tokens             |
| 10   | 302-304   | Generate embeddings (`text-embedding-ada-002`) → CSV       |
| 11   | 310-313   | Load embeddings from CSV, parse back to numpy arrays       |
| 12   | 319-391   | `create_context()` + `answer_question()` — RAG Q&A core   |
| 13   | 397-399   | Example queries                                            |

## Data Flow

```
openai.com
    │  BFS crawl (Step 4)
    ▼
text/openai.com/*.txt          Raw page text
    │  Clean + concat (Steps 5-6)
    ▼
processed/scraped.csv          title | text
    │  Tokenize + chunk (Steps 7-9)
    ▼
DataFrame in memory             text | n_tokens  (max 500 tokens/chunk)
    │  Embed (Step 10)
    ▼
processed/embeddings.csv       text | n_tokens | embeddings (1536-dim)
    │  Cosine similarity search (Step 12)
    ▼
Answer via text-davinci-003    Context window ≤ 1800 tokens
```

## Key Functions

### `crawl(url)` — Step 4
BFS using `collections.deque`. Tracks visited URLs in a `set`. Saves each page
as `text/<domain>/<url_slug>.txt`. Skips JS-only pages.

### `split_into_many(text, max_tokens=500)` — Step 8
Sentence-level splitting (splits on `. `). Accumulates sentences until the
token budget is exceeded, then starts a new chunk. Skips individual sentences
that exceed `max_tokens`.

### `create_context(question, df, max_len=1800)` — Step 12
Embeds the question, computes cosine distances against all stored embeddings,
then grabs the closest chunks until the combined token count hits `max_len`.
Returns chunks joined by `\n\n###\n\n`.

### `answer_question(df, model, question, ...)` — Step 12
Wraps `create_context` with a completion call. Prompt template instructs the
model to say "I don't know" when context is insufficient.

## OpenAI Models Used

| Purpose            | Model                      | Notes                           |
|--------------------|----------------------------|---------------------------------|
| Embeddings         | `text-embedding-ada-002`   | 1536 dimensions, cl100k_base    |
| Q&A completions    | `text-davinci-003`         | Legacy completions API          |
| Tokenizer          | `cl100k_base` (tiktoken)   | Matches ada-002 token counting  |

**Note:** Both models use the legacy `openai` v0.26 SDK (`openai.Embedding.create`,
`openai.Completion.create`). Migrating to v1.x would require switching to
`client.embeddings.create` / `client.completions.create`.

## Key Dependencies

| Package          | Role                                         |
|------------------|----------------------------------------------|
| `openai==0.26.1` | API client (legacy SDK)                      |
| `tiktoken`       | Token counting for chunking budget           |
| `beautifulsoup4` | HTML → text extraction during crawl          |
| `pandas`         | DataFrame storage for text + embeddings      |
| `numpy`          | Array ops for embedding vectors              |
| `scikit-learn`   | `distances_from_embeddings` (cosine distance)|
| `matplotlib`     | Token distribution histograms                |
| `requests`       | HTTP fetching in crawl step                  |

## Dev Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Lint
flake8 web-qa.py

# Run full pipeline (requires OPENAI_API_KEY env var)
python web-qa.py
```

## Gotchas

- **API key:** Must set `openai.api_key` or `OPENAI_API_KEY` env var before
  running — the script has the assignment commented out (line 24).
- **Rate limits:** Step 10 embeds every chunk sequentially. Large sites will
  hit rate limits. No retry/backoff is implemented.
- **Legacy SDK:** Uses `openai==0.26.1`. The `openai.Embedding.create` and
  `openai.Completion.create` APIs are deprecated in v1.x+.
- **JavaScript pages:** The crawler cannot render JS. It detects the
  "You need to enable JavaScript" string and skips those pages silently.
- **No incremental crawl:** Re-running `crawl()` overwrites all text files.
  The `seen` set is in-memory only.
- **URL normalization:** Trailing slashes are stripped, but query params and
  fragments are not deduplicated.
