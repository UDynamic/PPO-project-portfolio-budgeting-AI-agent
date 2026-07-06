# Literature Intelligence Pipeline
### RAG-based Literature Review System for RL × Project Portfolio Management Thesis

---

## Overview

This pipeline transforms a raw folder of academic PDFs into a structured, queryable knowledge base purpose-built for a Master's thesis on **Reinforcement Learning for Project Portfolio Management Budgeting**.

The system is designed to be re-run safely at any time. Drop new PDFs in, run the scripts in order, and only new files get processed.

---

## Pipeline Architecture

```
Raw PDFs
    │
    ▼
┌─────────────────────────────┐
│  1. Readability Filter      │  readability_test.py
│  Checks text extractability │
│  OCR quality, embedability  │
└────────────┬────────────────┘
             │
    ┌────────┴────────┐
    │                 │
  PASS              FAIL
    │                 │
    ▼                 ▼
lit_archive_passed/  lit_archive_failed/
                     (fix or discard)
             │
             ▼
┌─────────────────────────────┐
│  2. Relevance Filter        │  relevance_filter.py
│  Semantic similarity score  │
│  + Keyword scoring          │
└────────────┬────────────────┘
             │
    ┌────────┼────────┐
    │        │        │
    ▼        ▼        ▼
highly_   valuable/ irrelevant/
relevant/
             │
             ▼
┌─────────────────────────────┐
│  3. Paper Structurer        │  paper_structurer.py
│  LLM extracts thesis-       │
│  specific JSON from each    │
│  paper (Groq API, free)     │
└────────────┬────────────────┘
             │
             ▼
    structured/*.json
    (knowledge base)
```

---

## Folder Structure

```
rag_for_litreview/
│
├── .env                          ← API keys (never commit this)
│
├── lit_archive/                  ← DROP RAW PDFs HERE
│
├── lit_archive_failed/           ← Failed readability (scanned/encrypted)
│
├── lit_archive_passed/           ← Passed readability test
│   ├── highly_relevant/          ← Top relevance tier
│   │   ├── structured/           ← JSON knowledge base (highly relevant)
│   │   └── structured_failed/    ← Failed structuring logs
│   │
│   ├── valuable/                 ← Medium relevance tier
│   │   ├── structured/           ← JSON knowledge base (valuable)
│   │   └── structured_failed/    ← Failed structuring logs
│   │
│   └── irrelevant/               ← Low relevance (ignore)
│
├── relevance_results.json        ← Full ranking log from filter step
│
├── 1. readability_test.py
├── 2. relevance_filter.py
└── 3. paper_structurer.py
```

---

## Setup

### 1. Install dependencies

```bash
python -m pip install pymupdf sentence-transformers requests python-dotenv
```

### 2. Create `.env` file at project root

```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com) — no credit card required.

### 3. Prepare your PDFs

- Drop all raw PDF files into `lit_archive/`
- Rename files consistently before starting: `AuthorYear_Topic.pdf`

---

## Step-by-Step Usage

### Step 1 — Readability Test

**Script:** `1. readability_test.py`

**What it does:**
- Opens every PDF in `lit_archive/`
- Runs 3 tests on each file:
  - **Test 1 — Extraction:** Can PyMuPDF extract text? What ratio of pages are readable?
  - **Test 2 — Quality:** Is the extracted text meaningful? Checks alpha-character ratio and average word length to catch garbled OCR
  - **Test 3 — Embedding:** Can sentence-transformers generate a valid embedding from the text?
- Moves passing PDFs to `lit_archive_passed/`
- Moves failing PDFs to `lit_archive_failed/`

**Run:**
```bash
python "1. readability_test.py"
```

**Output:**
```
Found 4 PDFs to test
======================================================================
Testing: some_paper.pdf
  Test 1 - Extraction : PASS | Readable pages: 0.91 | Chars: 93763
  Test 2 - Quality    : PASS | Alpha ratio: 0.82 | Avg word len: 5.59
  Test 3 - Embedding  : PASS | Dim: 384
  → Overall: ✓ PASS — moving to ready
======================================================================
SUMMARY: 3 passed | 1 failed
```

**Fixing failed files:**

| Failure type | Cause | Fix |
|---|---|---|
| Test 1 fails, 0 chars | Scanned PDF | Run OCR via PDF24 or OCRmyPDF |
| Test 1 fails, error | Encrypted/DRM | Re-download via university library proxy |
| Test 2 fails, low alpha ratio | Math-heavy paper | Lower threshold to 0.45 in script |
| Test 2 fails, garbled text | Poor OCR quality | Better OCR tool needed |
| Test 3 fails | Corrupted file | Re-download |

**Threshold tuning** — if math-heavy papers fail Test 2, lower this line:
```python
passed = alpha_ratio > 0.45 and 3 < avg_word_len < 12  # default 0.6
```

---

### Step 2 — Relevance Filter

**Script:** `2. relevance_filter.py`

**What it does:**
- Reads all PDFs from `lit_archive_passed/`
- Scores each paper using two signals combined 50/50:
  - **Semantic similarity:** Embeds paper text and computes cosine similarity against 10 thesis-specific query sentences covering RL, EVM, portfolio management, S-curves, MDP, etc.
  - **Keyword scoring:** Weighted hits against domain-specific high-value terms (RL, CPI, SPI, portfolio, S-curve, milestone, stochastic...) and medium-value terms, with penalty for off-domain keywords (medical, biology, etc.)
- Classifies each paper into one of three tiers:
  - `highly_relevant/` — combined score ≥ 0.45
  - `valuable/` — combined score ≥ 0.30
  - `irrelevant/` — below threshold
- Moves PDFs into the corresponding subfolders
- Saves full ranked log to `relevance_results.json`

**Run:**
```bash
python "2. relevance_filter.py"
```

**Output:**
```
Found 47 PDFs to classify
======================================================================
[1/47] reinforcement_learning_portfolio.pdf
  Semantic : 0.621 | Keyword : 0.512 | Combined : 0.567
  Top keywords : ['reinforcement learning', 'portfolio', 'budget allocation']
  → HIGHLY_RELEVANT

RANKED PAPERS (highest to lowest relevance):
─────────────────────────────────────────────
  0.567 ███████████          [HI] reinforcement_learning_portfolio.pdf
  0.489 █████████            [HI] stochastic_project_budget.pdf
  0.312 ██████               [VA] resource_constrained_scheduling.pdf
  0.201 ████                 [IR] image_segmentation_neural.pdf
```

**Threshold tuning** — adjust these two lines at the top of the script:
```python
HIGHLY_RELEVANT_THRESHOLD = 0.45   # raise to be stricter
VALUABLE_THRESHOLD        = 0.30   # lower to rescue borderline papers
```

**Adding new papers** — safe to re-run. However, since files are *moved* not copied, re-running on an empty `lit_archive_passed/` root will find nothing. New papers must go through Step 1 first.

---

### Step 3 — Paper Structurer

**Script:** `3. paper_structurer.py`

**What it does:**
- Processes all PDFs in both `highly_relevant/` and `valuable/`
- For each PDF:
  1. Extracts abstract, introduction, and conclusion (falls back to first 2 pages)
  2. Sends extracted text to Groq API (free, `llama-3.3-70b-versatile`)
  3. Parses the JSON response with 4-layer fallback handling
  4. Saves structured JSON to `structured/` subfolder alongside the PDF
- Skips files that already have a JSON (safe to re-run)
- Skips files that previously failed (delete `_FAILED.txt` to retry)
- Logs failed extractions to `structured_failed/`

**Run:**
```bash
python "3. paper_structurer.py"
```

**Output:**
```
============================================================
Paper Structurer
Model  : llama-3.3-70b-versatile
Folders: 2
============================================================

── HIGHLY_RELEVANT ──────────────────────────────────────
  [1/12] Fleming2016_EarnedValueManagement.pdf
    ✓ Extracted 5832 chars
    ✓ Saved → ./lit_archive_passed/highly_relevant/structured/Fleming2016.json

── VALUABLE ─────────────────────────────────────────────
  [1/8] Archer1999_ProjectPortfolioSelection.pdf
    → Already structured — skipping

============================================================
TOTAL SUMMARY
  Structured : 18
  Failed     : 0
  Skipped    : 2
  Total PDFs : 20
============================================================
```

**JSON output schema — per paper:**
```json
{
  "metadata": {
    "title_guess": "Earned Value Management: A Project Control Tool",
    "year_guess": "2016",
    "problem": "Tracking project cost and schedule performance",
    "method": "EVM indices CPI and SPI with forecasting",
    "domain": "construction, defense, IT",
    "level": "single project",
    "model_type": "statistical",
    "data_type": "empirical"
  },
  "calibration_params": {
    "cpi_stats": "mean 0.89, std 0.12, lognormal distribution",
    "spi_stats": "mean 0.85, stabilizes after 20% completion",
    "cost_overrun": "15-30% across construction projects",
    "schedule_overrun": "20% average delay",
    "project_type": "construction, defense",
    "sample_size": 120,
    "scurve_params": "beta distribution a=1.5 b=3.0",
    "efficiency_params": null,
    "horizon": "monthly"
  },
  "thesis_relevance": {
    "supports_environment_design": true,
    "supports_rl_agent": false,
    "supports_problem_motivation": true,
    "supports_calibration": true,
    "key_finding": "CPI stabilizes after 20% project completion making it a reliable predictor",
    "gap_identified": "No automated decision framework exists for portfolio-level budget allocation"
  },
  "key_variables": ["CPI", "SPI", "BAC", "EAC", "EV", "AC", "PV"],
  "_source": {
    "filename": "Fleming2016_EarnedValueManagement.pdf",
    "folder": "./lit_archive_passed/highly_relevant",
    "time": "2026-07-06T18:30:00",
    "model": "llama-3.3-70b-versatile"
  }
}
```

**Adding new PDFs — protocol:**

```
1. Drop new PDFs into lit_archive/
2. Run:  python "1. readability_test.py"
3. Run:  python "2. relevance_filter.py"
         (source_folder must point to lit_archive_passed/ root)
4. Run:  python "3. paper_structurer.py"
         (only new files get processed — existing JSONs are skipped)
```

**Retrying failed files:**
```bash
# Delete the failure log to allow retry on next run
rm lit_archive_passed/highly_relevant/structured_failed/paper_FAILED.txt
python "3. paper_structurer.py"
```

---

## Re-run Safety Summary

| Script | Safe to re-run? | Behaviour on existing files |
|---|---|---|
| `readability_test.py` | ✓ Yes | Re-tests all files in inbox folder |
| `relevance_filter.py` | ⚠ Careful | Moves files — inbox must have files |
| `paper_structurer.py` | ✓ Yes | Skips any PDF with existing JSON |

---

## Switching Models

To change the LLM used for structuring, edit one line in `3. paper_structurer.py`:

```python
MODEL = "llama-3.3-70b-versatile"      # best quality (default)
MODEL = "llama-3.1-8b-instant"          # fastest, highest daily limit
MODEL = "deepseek-r1-distill-qwen-32b"  # if available on your Groq account
```

Check currently available models:
```bash
curl -s https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY" | python -m json.tool | grep '"id"'
```

---

## What Comes Next

With structured JSONs in place, the next modules to build on top of this pipeline are:

- **Query engine** — ask questions across all structured papers in plain English
- **Literature review generator** — section-by-section drafting grounded in retrieved JSON fields
- **Calibration extractor** — pull all `calibration_params` fields across papers to feed RL environment parameter distributions (CPI, SPI, S-curve shape, overrun rates)
- **Gap analysis** — aggregate `gap_identified` fields to build the thesis motivation section

---

## Dependencies

| Package | Purpose |
|---|---|
| `pymupdf` | PDF text extraction |
| `sentence-transformers` | Local embeddings for relevance scoring |
| `requests` | Groq API calls |
| `python-dotenv` | `.env` file loading |

```bash
python -m pip install pymupdf sentence-transformers requests python-dotenv
```

---

## Notes

- Never commit `.env` to version control — add it to `.gitignore`
- All LLM extraction is approximate — verify calibration parameters against source PDFs before using in thesis
- The `structured/` JSONs are the ground truth for downstream querying — treat them as your thesis knowledge base