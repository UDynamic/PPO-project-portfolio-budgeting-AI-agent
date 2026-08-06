"""
relevance_filter.py
-------------------
Filters and classifies academic PDFs by relevance to your thesis topic.
Uses embedding similarity + keyword scoring to rank papers into:
  - highly_relevant/
  - valuable/
  - irrelevant/

Usage:
    python relevance_filter.py

Requirements:
    pip install pymupdf sentence-transformers
"""

import fitz  # PyMuPDF
import os
import shutil
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# ─────────────────────────────────────────────
# CONFIG — edit paths if needed
# ─────────────────────────────────────────────
INPUT_FOLDER        = "./lit_archive"                          # drop all raw PDFs here
PASSED_FOLDER       = "./lit_archive_passed"                   # from readability test
HIGHLY_RELEVANT     = "./lit_archive_passed/highly_relevant"
VALUABLE            = "./lit_archive_passed/valuable"
IRRELEVANT          = "./lit_archive_passed/irrelevant"
RESULTS_LOG         = "./lit_archive_passed/relevance_results.json"

# Thresholds — tune these after first run
HIGHLY_RELEVANT_THRESHOLD = 0.45   # combined score above this → highly relevant
VALUABLE_THRESHOLD        = 0.30   # above this → valuable
# below VALUABLE_THRESHOLD → irrelevant

os.makedirs(HIGHLY_RELEVANT, exist_ok=True)
os.makedirs(VALUABLE,        exist_ok=True)
os.makedirs(IRRELEVANT,      exist_ok=True)

# ─────────────────────────────────────────────
# THESIS DEFINITION
# What your thesis is about — used for semantic similarity
# ─────────────────────────────────────────────
THESIS_QUERIES = [
    "project management",
    "project portfolio management",
    "project portfolio management budget allocation",
    "stochastic environment simulation project portfolio optimization",
    "earned value management CPI SPI performance indices construction projects",
    "S-curve expenditure profile beta distribution project cost modeling",
    "Markov decision process resource allocation under uncertainty",
    "autonomous agent budget decision making contractor portfolio",
    "project cost overrun schedule delay empirical distribution",
    "milestone payment advance payment retention project contract cash flow",
    "project portfolio selection optimization multi-project management",
    "reinforcement learning sequential decision making operations management",
]

# ─────────────────────────────────────────────
# KEYWORD SCORING
# Direct keyword hits boost the relevance score
# ─────────────────────────────────────────────
HIGH_VALUE_KEYWORDS = [
    "project portfolio", "portfolio management", "budget allocation",
    "earned value", "evm", "cpi", "spi", "cost performance", "schedule performance",
    "s-curve", "expenditure profile", "cost overrun", "schedule overrun",
    "milestone", "advance payment", "retention", "contract", "epc",
    "resource allocation", "multi-project", "portfolio optimization",
    "stochastic", "uncertainty", "simulation", "gymnasium", "openai gym",
    "reinforcement learning", "markov decision", "mdp", "deep rl", "ppo",
]

MEDIUM_VALUE_KEYWORDS = [
    "project management", "project scheduling", "resource constrained",
    "project selection",
    "optimization", "heuristic", "metaheuristic", "genetic algorithm",
    "monte carlo", "risk", "probability distribution", "empirical",
    "construction", "infrastructure", "contractor", "stakeholder",
    "cost estimation", "forecasting", "planning", "control",
    "neural network", "machine learning", "deep learning", "agent",
    "reward", "policy", "value function", "q-learning",
]

NEGATIVE_KEYWORDS = [
    "medical", "clinical", "hospital", "patient", "drug", "cancer",
    "agriculture", "crop", "farming", "biology", "chemistry",
    "image recognition", "natural language", "sentiment analysis",
    "autonomous driving", "robotics manipulation", "game playing atari",
]


# ─────────────────────────────────────────────
# STEP 1 — Extract text sample for scoring
# ─────────────────────────────────────────────
def extract_text_sample(pdf_path: str, max_chars: int = 4000) -> str:
    """Extract abstract + intro + conclusion for scoring."""
    try:
        with fitz.open(pdf_path) as doc:
            pages = [page.get_text() for page in doc]

        full_text  = "\n".join(pages)
        text_lower = full_text.lower()

        def get_section(keyword, length=1500):
            idx = text_lower.find(keyword)
            if idx == -1:
                return ""
            return full_text[idx: idx + length]

        abstract   = get_section("abstract",     1500)
        intro      = get_section("introduction", 1500)
        conclusion = get_section("conclusion",   1000)

        combined = "\n".join(filter(None, [abstract, intro, conclusion])).strip()

        if len(combined) < 200:
            combined = "\n".join(pages[:3])

        return combined[:max_chars]

    except Exception as e:
        print(f"  ✗ Text extraction error: {e}")
        return ""


# ─────────────────────────────────────────────
# STEP 2 — Keyword score
# ─────────────────────────────────────────────
def keyword_score(text: str) -> tuple[float, list, list]:
    """
    Returns:
        score        : float 0.0 to 1.0
        hits_high    : list of matched high-value keywords
        hits_medium  : list of matched medium-value keywords
    """
    text_lower = text.lower()

    hits_high   = [kw for kw in HIGH_VALUE_KEYWORDS   if kw in text_lower]
    hits_medium = [kw for kw in MEDIUM_VALUE_KEYWORDS if kw in text_lower]
    hits_neg    = [kw for kw in NEGATIVE_KEYWORDS     if kw in text_lower]

    raw_score = (len(hits_high) * 2.0 + len(hits_medium) * 0.5) / \
                (len(HIGH_VALUE_KEYWORDS) * 2.0 + len(MEDIUM_VALUE_KEYWORDS) * 0.5)

    # Penalize negative keyword hits
    penalty = min(len(hits_neg) * 0.1, 0.4)
    score   = max(0.0, min(1.0, raw_score - penalty))

    return round(score, 3), hits_high, hits_medium


# ─────────────────────────────────────────────
# STEP 3 — Semantic similarity score
# ─────────────────────────────────────────────
def semantic_score(text: str, model, thesis_embeddings) -> float:
    """
    Embeds the paper text and computes max cosine similarity
    against all thesis query embeddings.
    """
    if not text.strip():
        return 0.0

    paper_embedding = model.encode(text, convert_to_tensor=True)

    scores = [
        float(util.cos_sim(paper_embedding, te))
        for te in thesis_embeddings
    ]

    return round(max(scores), 3)


# ─────────────────────────────────────────────
# STEP 4 — Combined score and classification
# ─────────────────────────────────────────────
def classify(combined_score: float) -> str:
    if combined_score >= HIGHLY_RELEVANT_THRESHOLD:
        return "highly_relevant"
    elif combined_score >= VALUABLE_THRESHOLD:
        return "valuable"
    else:
        return "irrelevant"


def destination_folder(classification: str) -> str:
    return {
        "highly_relevant": HIGHLY_RELEVANT,
        "valuable":        VALUABLE,
        "irrelevant":      IRRELEVANT,
    }[classification]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_filtering(source_folder: str = PASSED_FOLDER):
    """
    source_folder: where your readability-passed PDFs are.
    Defaults to lit_archive_passed/ root (not subfolders).
    """
    # Find all PDFs in source folder (non-recursive)
    files = sorted([
        f for f in os.listdir(source_folder)
        if f.endswith(".pdf")
    ])

    if not files:
        print(f"\nNo PDFs found in: {source_folder}")
        print("Make sure your readability-passed PDFs are in that folder.")
        return

    print(f"\n{'='*65}")
    print(f"Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Encoding thesis queries...")
    thesis_embeddings = [
        model.encode(q, convert_to_tensor=True)
        for q in THESIS_QUERIES
    ]

    print(f"\nFound {len(files)} PDFs to classify")
    print(f"Thresholds: highly_relevant ≥ {HIGHLY_RELEVANT_THRESHOLD} | "
          f"valuable ≥ {VALUABLE_THRESHOLD}")
    print(f"{'='*65}\n")

    results      = []
    counts       = {"highly_relevant": 0, "valuable": 0, "irrelevant": 0}

    for i, filename in enumerate(files, 1):
        pdf_path = os.path.join(source_folder, filename)
        print(f"[{i}/{len(files)}] {filename[:65]}")

        # Extract text
        text = extract_text_sample(pdf_path)
        if not text:
            print("  ✗ Could not extract text — skipping\n")
            continue

        # Score
        kw_score, hits_high, hits_medium = keyword_score(text)
        sem_score = semantic_score(text, model, thesis_embeddings)

        # Combined: 50% semantic + 50% keyword
        combined = round((sem_score * 0.5) + (kw_score * 0.5), 3)

        # Classify
        label = classify(combined)
        dest  = destination_folder(label)

        # move to destination
        shutil.move(pdf_path, os.path.join(dest, filename))
        counts[label] += 1

        print(f"  Semantic : {sem_score:.3f} | "
              f"Keyword  : {kw_score:.3f} | "
              f"Combined : {combined:.3f}")
        print(f"  Top keywords : {hits_high[:5]}")
        print(f"  → {label.upper()}\n")

        results.append({
            "filename":       filename,
            "semantic_score": sem_score,
            "keyword_score":  kw_score,
            "combined_score": combined,
            "classification": label,
            "keywords_hit":   hits_high + hits_medium,
        })

    # Sort results by combined score descending
    results.sort(key=lambda x: x["combined_score"], reverse=True)

    # Save log
    log = {
        "run_timestamp": datetime.now().isoformat(),
        "source_folder": source_folder,
        "thresholds": {
            "highly_relevant": HIGHLY_RELEVANT_THRESHOLD,
            "valuable":        VALUABLE_THRESHOLD,
        },
        "summary": counts,
        "papers":  results,
    }

    with open(RESULTS_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    # Print summary
    print(f"\n{'='*65}")
    print(f"CLASSIFICATION COMPLETE")
    print(f"  Highly Relevant : {counts['highly_relevant']}")
    print(f"  Valuable        : {counts['valuable']}")
    print(f"  Irrelevant      : {counts['irrelevant']}")
    print(f"  Total processed : {sum(counts.values())}")
    print(f"\nFull results saved → {RESULTS_LOG}")
    print(f"{'='*65}")

    # Print ranked list
    print(f"\nRANKED PAPERS (highest to lowest relevance):")
    print(f"{'─'*65}")
    for r in results:
        bar = "█" * int(r["combined_score"] * 20)
        print(f"  {r['combined_score']:.3f} {bar:<20} [{r['classification'][:2].upper()}] "
              f"{r['filename'][:45]}")


if __name__ == "__main__":
    # Point this at wherever your readability-passed PDFs live
    # Change to HIGHLY_RELEVANT etc. if you want to re-filter a subfolder
    run_filtering(source_folder=PASSED_FOLDER)