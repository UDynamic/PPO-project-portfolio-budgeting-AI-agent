#!/usr/bin/env python3
"""
lit_synthesis.py

Generates a copy-pasteable LaTeX synthesis table from the same structured
JSON archive used by lit_review_dashboard.py.

Citation key resolution
-----------------------
Reads metadata.citation_key from each JSON file — that's it.
If the field is absent or empty the paper is flagged NOT FOUND in both
the .tex output and the console report so you can add it.

Add to any JSON file:
    {
      "metadata": {
        "citation_key": "Schulman2016",
        ...
      },
      ...
    }

USAGE
-----
    python lit_synthesis.py --base-dir /path/to/lit_archive_passed

OUTPUT
------
    <base-dir>/lit_review_dashboard_output/tables/synthesis_table.tex
"""

import argparse
import json
import re
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIG  (mirrors lit_review_dashboard.py)
# --------------------------------------------------------------------------

BASE_SUBDIRS = [
    Path("highly_relevant") / "structured",
    Path("valuable") / "structured",
]

SYM_FULL    = r"\checkmark"
SYM_PARTIAL = r"$\circ$"
SYM_NONE    = "--"

DIMENSION_RULES = {
    "portfolio_level": {
        "full":    ["portfolio", "project portfolio", "project selection",
                    "portfolio management", "portfolio optimiz"],
        "partial": ["multi-project", "multiple project", "program"],
    },
    "decision_type": {
        "selection":  ["project selection", "portfolio selection", "r&d selection",
                       "technology selection", "option", "investment select"],
        "scheduling": ["schedul", "resource-constrained project", "rcpsp", "makespan",
                       "time-constrained", "project planning", "task schedul"],
        "budgeting":  ["budget", "cash flow", "cashflow", "cash-flow", "cost control",
                       "earned value", "expenditure", "financing", "working capital",
                       "liquidity"],
    },
    "sequential_decisions": {
        "full":    ["sequential decision", "multi-stage", "multi-period", "dynamic decision",
                    "markov decision", "reinforcement learning", "policy", "stage-gate",
                    "time-step", "episode", "online decision"],
        "partial": ["iterative", "rolling horizon", "re-plan", "adaptive", "update",
                    "recourse", "revision"],
    },
    "stochastic_cash_flows": {
        "full":    ["stochastic cash", "stochastic cashflow", "random cash", "uncertain cash",
                    "cash flow uncertain", "cashflow uncertain", "probabilistic cash",
                    "monte carlo", "s-curve", "scurve"],
        "partial": ["stochastic", "uncertain", "probabilistic", "risk", "volatil",
                    "scenario", "sensitivity"],
    },
    "contract_mechanics": {
        "full":    ["contract mechanic", "milestone payment", "payment schedule",
                    "contract term", "payment term", "liquidated damage", "retention",
                    "progress payment", "contract cash", "invoicing", "billing"],
        "partial": ["contract", "payment", "invoice", "billing cycle", "advance payment",
                    "down payment"],
    },
    "termination_settlement": {
        "full":    ["termination", "early termination", "project abandon", "cancellation",
                    "wind-down", "settlement", "contract termination"],
        "partial": ["abandon", "discontinu", "exit option", "defer", "real option"],
    },
}

_CAPTION = (
    r"Representative literature mapped against modelling dimensions of the "
    r"present framework. \textit{Decision type}: S\,=\,selection, "
    r"Sc\,=\,scheduling, B\,=\,budgeting. "
    r"\checkmark~=~fully addressed;\; $\circ$~=~partially addressed;\; "
    r"{--}~=~not addressed."
)


# --------------------------------------------------------------------------
# LOADING
# --------------------------------------------------------------------------

def load_records(base_dir: Path) -> list[dict]:
    records = []
    for sub in BASE_SUBDIRS:
        folder = base_dir / sub
        if not folder.exists():
            print(f"  [warn] folder not found, skipping: {folder}")
            continue
        bucket = sub.parts[0]
        for fp in sorted(folder.glob("*.json")):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                rec["_bucket"]   = bucket
                rec["_filepath"] = str(fp)
                records.append(rec)
            except Exception as e:
                print(f"  [warn] failed to parse {fp}: {e}")
    return records


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

def extract_year(rec: dict) -> str:
    meta = rec.get("metadata", {}) or {}
    y    = str(meta.get("year_guess", ""))
    m    = re.search(r"(19|20)\d{2}", y)
    if m:
        return m.group()
    fname = (rec.get("_source", {}) or {}).get("filename", "")
    m = re.search(r"(19|20)\d{2}", fname)
    if m:
        return m.group()
    return ""


def record_text(rec: dict) -> str:
    meta    = rec.get("metadata", {}) or {}
    rel     = rec.get("thesis_relevance", {}) or {}
    keyvars = rec.get("key_variables", []) or []
    calib   = rec.get("calibration_params", {}) or {}
    parts   = [
        str(meta.get("problem",     "")),
        str(meta.get("method",      "")),
        str(meta.get("domain",      "")),
        str(meta.get("model_type",  "")),
        str(meta.get("title_guess", "")),
        str(rel.get("key_finding",    "")),
        str(rel.get("gap_identified", "")),
        " ".join(keyvars),
        " ".join(str(v) for v in calib.values()),
    ]
    return " ".join(parts).lower()


def resolve_citation_key(rec: dict) -> tuple[str, bool]:
    """
    Returns (key, found).
    Reads metadata.citation_key only — explicit, no guessing.
    """
    meta = rec.get("metadata", {}) or {}
    key  = (meta.get("citation_key") or "").strip()
    if key:
        return key, True
    return "NOT_FOUND", False


# --------------------------------------------------------------------------
# DIMENSION CLASSIFIERS
# --------------------------------------------------------------------------

def _contains_any(text: str, keywords: list) -> bool:
    return any(kw.lower() in text for kw in keywords)


def classify_binary(text: str, rules: dict) -> str:
    if _contains_any(text, rules.get("full", [])):
        return SYM_FULL
    if _contains_any(text, rules.get("partial", [])):
        return SYM_PARTIAL
    return SYM_NONE


def classify_decision_type(text: str) -> str:
    scores = {
        dtype: sum(text.count(kw.lower()) for kw in kws)
        for dtype, kws in DIMENSION_RULES["decision_type"].items()
    }
    best = max(scores.values())
    if best == 0:
        return "--"
    winners = [d for d, s in scores.items() if s == best]
    codes   = {"selection": "S", "scheduling": "Sc", "budgeting": "B"}
    return "/".join(codes[w] for w in winners)


# --------------------------------------------------------------------------
# ROW BUILDER
# --------------------------------------------------------------------------

def build_row(rec: dict) -> dict:
    text    = record_text(rec)
    key, ok = resolve_citation_key(rec)
    return {
        "bib_key":    key,
        "found":      ok,
        "portfolio":  classify_binary(text, DIMENSION_RULES["portfolio_level"]),
        "decision":   classify_decision_type(text),
        "sequential": classify_binary(text, DIMENSION_RULES["sequential_decisions"]),
        "stochastic": classify_binary(text, DIMENSION_RULES["stochastic_cash_flows"]),
        "contract":   classify_binary(text, DIMENSION_RULES["contract_mechanics"]),
        "termination":classify_binary(text, DIMENSION_RULES["termination_settlement"]),
        "bucket":     rec.get("_bucket", ""),
        "year":       extract_year(rec),
        "filename":   Path(rec.get("_filepath", "unknown")).name,
        "title_guess":(rec.get("metadata", {}) or {}).get("title_guess", ""),
    }


# --------------------------------------------------------------------------
# CONSOLE REPORT
# --------------------------------------------------------------------------

def print_report(rows: list[dict]) -> None:
    found   = [r for r in rows if r["found"]]
    missing = [r for r in rows if not r["found"]]

    print(f"\n{'='*60}")
    print(f"  CITATION KEY REPORT  ({len(found)} resolved / {len(missing)} missing)")
    print(f"{'='*60}")

    if found:
        print("\n  RESOLVED:")
        for r in sorted(found, key=lambda x: x["bib_key"].lower()):
            print(f"    ✓  {r['bib_key']:<35} ← {r['filename']}")

    if missing:
        print("\n  MISSING citation_key in metadata — add to these JSON files:")
        for r in missing:
            title = r["title_guess"] or "(no title_guess)"
            print(f"\n    ✗  {r['filename']}")
            print(f"       title : {title[:72]}")
            print(f"       year  : {r['year'] or '?'}")
            print(f'       fix   : add  "citation_key": "AuthorYear"  to metadata')

    print(f"\n{'='*60}\n")


# --------------------------------------------------------------------------
# LATEX RENDERER
# --------------------------------------------------------------------------

def render_latex(rows: list[dict]) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{_CAPTION}}}")
    lines.append(r"\label{tab:litmap}")
    lines.append(r"\renewcommand{\arraystretch}{1.3}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Work} &"
        r"  \textbf{Portfolio} &"
        r"  \textbf{Decision} &"
        r"  \textbf{Sequential} &"
        r"  \textbf{Stochastic} &"
        r"  \textbf{Contract} &"
        r"  \textbf{Termination} \\"
    )
    lines.append(
        r"  &"
        r"  \textbf{level} &"
        r"  \textbf{type} &"
        r"  \textbf{decisions} &"
        r"  \textbf{cash flows} &"
        r"  \textbf{mechanics} &"
        r"  \textbf{settlement} \\"
    )
    lines.append(r"\hline")

    def sort_key(r):
        bucket_order = 0 if r["bucket"] == "highly_relevant" else 1
        try:
            yr = int(r["year"]) if r["year"] else 9999
        except ValueError:
            yr = 9999
        return (bucket_order, yr)

    for r in sorted(rows, key=sort_key):
        if r["found"]:
            work_cell = rf"\citet{{{r['bib_key']}}}"
        else:
            # Visible placeholder + comment so the file still compiles
            work_cell = (
                f"% NOT FOUND: add citation_key to {r['filename']}\n"
                rf"  \textbf{{[REF MISSING]}}"
            )

        lines.append(
            f"{work_cell:<50} & "
            f"{r['portfolio']:<12} & "
            f"{r['decision']:<6} & "
            f"{r['sequential']:<12} & "
            f"{r['stochastic']:<12} & "
            f"{r['contract']:<12} & "
            f"{r['termination']} \\\\"
        )

    # Present-work row — always last
    lines.append(
        r"\textbf{Present work}                                & "
        r"\checkmark & "
        r"\textbf{B}  & "
        r"\checkmark & "
        r"\checkmark & "
        r"\checkmark & "
        r"\checkmark \\"
    )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX synthesis table from the structured JSON archive."
    )
    parser.add_argument(
        "--base-dir", type=str, default="lit_archive_passed",
        help="Folder containing 'highly_relevant' and 'valuable' subfolders."
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output folder (default: <base-dir>/lit_review_dashboard_output)."
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    out_dir  = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else base_dir / "lit_review_dashboard_output"
    )
    tbl_dir = out_dir / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading records from {base_dir} ...")
    records = load_records(base_dir)
    print(f"  loaded {len(records)} records")

    if not records:
        print("No records found — check --base-dir. Exiting.")
        return

    rows = [build_row(r) for r in records]

    print_report(rows)

    tex      = render_latex(rows)
    out_path = tbl_dir / "synthesis_table.tex"
    out_path.write_text(tex, encoding="utf-8")

    print(f"LaTeX table written to:\n  {out_path}")
    print("\nUse in your document:")
    print(r"  \input{tables/synthesis_table.tex}")


if __name__ == "__main__":
    main()