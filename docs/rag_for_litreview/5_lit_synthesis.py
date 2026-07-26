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

    # Limit to the 30 most informative rows (default):
    python lit_synthesis.py --base-dir /path/to/lit_archive_passed --max-rows 30

    # No limit:
    python lit_synthesis.py --base-dir /path/to/lit_archive_passed --max-rows 0

OUTPUT
------
    <base-dir>/lit_review_dashboard_output/tables/synthesis_table.tex

REQUIRED LaTeX PACKAGES
-----------------------
Add these to your document preamble (order matters for xcolor + colortbl):

    \\usepackage{booktabs}
    \\usepackage[table]{xcolor}
    \\usepackage{colortbl}
    \\usepackage{array}

The generated .tex file also contains the \\definecolor declarations as a
comment block at the top so you can move them to a shared style file if
preferred.
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

# Raw symbols — used internally for scoring/comparison logic only.
# The render layer wraps these in colour commands before writing LaTeX.
SYM_FULL    = r"\checkmark"
SYM_PARTIAL = r"$\circ$"
SYM_NONE    = "--"

# ── Colour palette ──────────────────────────────────────────────────────────
# Defined here so a single edit propagates everywhere.
# All names must be declared via \definecolor in _PREAMBLE_COLORS.
#
#   COL_FULL      green tick  — "fully addressed"
#   COL_PARTIAL   amber ring  — "partially addressed"
#   COL_NONE      mid-grey    — "not addressed"
#   COL_RARE_BG   steel-blue column tint (sequential / contract / termination)
#   COL_HEAD_BG   dark-navy header row background
#   COL_HEAD_FG   white text on dark header
#   COL_PW_BG     present-work row background (deep teal)
#   COL_ROW_ODD   alternating row tint (very light blue-grey)

COL_FULL     = "symgreen"
COL_PARTIAL  = "symamber"
COL_NONE     = "symgrey"
COL_RARE_BG  = "rarebg"
COL_HEAD_BG  = "headbg"
COL_HEAD_FG  = "headfg"
COL_PW_BG    = "pwbg"
COL_ROW_ODD  = "rowodd"

_PREAMBLE_COLORS = r"""% ── Synthesis-table colour definitions (auto-generated) ─────────────────────
% Required packages (add to your preamble if not already present):
%   \usepackage{booktabs}
%   \usepackage[table]{xcolor}
%   \usepackage{colortbl}
%   \usepackage{array}
%
\definecolor{symgreen}{HTML}{1A7A4A}   % full \checkmark
\definecolor{symamber}{HTML}{B45309}   % partial \circ
\definecolor{symgrey}{HTML}{9CA3AF}    % not addressed --
\definecolor{rarebg}{HTML}{EFF6FF}     % rare-column header tint (blue-50)
\definecolor{headbg}{HTML}{1E3A5F}     % header row background (dark navy)
\definecolor{headfg}{HTML}{FFFFFF}     % header row text (white)
\definecolor{pwbg}{HTML}{14532D}       % present-work row background (deep green)
\definecolor{rowodd}{HTML}{F8FAFC}     % alternating odd-row tint
% ─────────────────────────────────────────────────────────────────────────────
"""

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

# Columns that are rare in the literature and are most important to preserve
# (these directly highlight the gap that Present work fills).
_RARE_COLUMNS = {"sequential", "contract", "termination"}

# Decision-type codes — used to enforce variety in the kept set
_DECISION_CODES = {"S", "Sc", "B", "S/Sc", "S/B", "Sc/B"}


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
# ROW SELECTION  (enforces --max-rows limit)
# --------------------------------------------------------------------------

def _row_score(row: dict) -> int:
    """
    Informativeness score used to rank rows before selection.

    Points awarded:
      +3  per column that has a full \\checkmark  in a rare dimension
          (sequential, contract, termination) — these directly show
          the gap that Present work fills and must be well-represented
      +2  per column that has a full \\checkmark  in any dimension
      +1  per column that has a partial $\\circ$
      +1  if the decision type is a compound code (S/Sc, S/B, Sc/B)
          — these are rarer and add variety
    """
    score = 0
    for col in ("portfolio", "sequential", "stochastic", "contract", "termination"):
        val = row[col]
        if val == SYM_FULL:
            bonus = 3 if col in _RARE_COLUMNS else 2
            score += bonus
        elif val == SYM_PARTIAL:
            score += 1
    if "/" in row["decision"]:
        score += 1
    return score


def select_rows(rows: list[dict], max_rows: int) -> list[dict]:
    """
    Return up to *max_rows* rows chosen to maximise:
      1. Coverage of rare columns (sequential, contract, termination)
      2. Overall informativeness (_row_score)
      3. Decision-type variety — at least one row per code that exists
         in the full set, up to the budget

    When max_rows <= 0 the full list is returned unchanged.
    """
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows

    # Sort descending by score so the greedy pass picks the best first
    ranked = sorted(rows, key=_row_score, reverse=True)

    kept          = []
    seen_decisions = set()

    # ── Pass 1: guarantee at least one row per rare-column \checkmark ──────
    # Iterate rare columns in order of scarcity (keep the most distinctive).
    for col in _RARE_COLUMNS:
        if len(kept) >= max_rows:
            break
        for r in ranked:
            if r in kept:
                continue
            if r[col] == SYM_FULL:
                kept.append(r)
                seen_decisions.add(r["decision"])
                break   # one seed per rare column is enough

    # ── Pass 2: one representative per decision-type code ─────────────────
    present_decisions = {r["decision"] for r in ranked} - {"--"}
    for code in sorted(present_decisions):
        if len(kept) >= max_rows:
            break
        if code in seen_decisions:
            continue
        for r in ranked:
            if r in kept:
                continue
            if r["decision"] == code:
                kept.append(r)
                seen_decisions.add(code)
                break

    # ── Pass 3: fill remaining slots by descending score ──────────────────
    for r in ranked:
        if len(kept) >= max_rows:
            break
        if r not in kept:
            kept.append(r)

    print(
        f"  [select] {len(rows)} rows → kept {len(kept)} "
        f"(limit: {max_rows}, dropped: {len(rows) - len(kept)})"
    )
    return kept


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

def _colour_sym(raw: str) -> str:
    """
    Wrap a raw symbol (SYM_FULL / SYM_PARTIAL / SYM_NONE) in its
    semantic colour command.  The result is safe to embed in a table cell.
    """
    if raw == SYM_FULL:
        return rf"\textcolor{{{COL_FULL}}}{{\checkmark}}"
    if raw == SYM_PARTIAL:
        return rf"\textcolor{{{COL_PARTIAL}}}{{$\circ$}}"
    # SYM_NONE  →  light grey dash
    return rf"\textcolor{{{COL_NONE}}}{{--}}"


def _colour_decision(code: str) -> str:
    """
    Render the decision-type code in small-caps with a neutral dark colour
    so it stands out from the symbol columns without competing with them.
    If the code is '--' apply the same grey as SYM_NONE.
    """
    if code == "--":
        return rf"\textcolor{{{COL_NONE}}}{{--}}"
    return rf"\textsc{{\small {code}}}"


def render_latex(rows: list[dict]) -> str:
    """
    Emit a fully styled LaTeX table.

    Styling features
    ----------------
    * booktabs rules  (toprule / midrule / bottomrule)
    * Dark-navy header row with white bold text
    * Rare columns (Sequential, Contract, Termination) carry a persistent
      steel-blue column tint via columncolor so readers instantly see the
      gap the present work fills
    * Alternating light-grey row tints on body rows (rowcolors)
    * Semantic colour per symbol: green checkmark, amber circ, grey --
    * Present-work row: deep-green background, white bold text
    * cmidrule separator before the present-work row

    Column spec note
    ----------------
    The rare-column token is built with plain string concatenation:
        rare_col = r">{" + r"\columncolor{" + COL_RARE_BG + r"}}c"
    Do NOT use rf-string tricks like  rf">{{" + var + r"}}c"  — the brace
    escaping interacts with the inner braces of \columncolor{} and produces
    a triple closing brace (>{\columncolor{X}}}c) which causes the fatal
    LaTeX error: "array Error: >{..} at wrong position".
    """
    # ── Column spec ─────────────────────────────────────────────────────────
    # Each rare-column token must be exactly:  >{\columncolor{NAME}}c
    # Brace audit: 1x{ opens \columncolor arg, 1x} closes it
    #              1x{ opens >{  preamble,     1x} closes it  → 2 pairs only.
    rare_col = ">{" + "\\columncolor{" + COL_RARE_BG + "}}c"
    col_spec = (
        ">{\\" + "raggedright\\arraybackslash}p{3.8cm}"  # Work  (no rf-string)
        "c"                                               # Portfolio
        "c"                                               # Decision
        + rare_col                                        # Sequential ← rare
        + "c"                                             # Stochastic
        + rare_col                                        # Contract   ← rare
        + rare_col                                        # Termination← rare
    )

    L = []  # output lines

    # ── colour definitions (prepended so .tex is self-contained) ────────────
    L.append(_PREAMBLE_COLORS)

    # ── table environment ────────────────────────────────────────────────────
    L.append(r"\begin{table}[htbp]")
    L.append(r"  \centering")
    L.append(rf"  \caption{{{_CAPTION}}}")
    L.append(r"  \label{tab:litmap}")
    L.append(r"  \setlength{\tabcolsep}{6pt}")
    L.append(r"  \renewcommand{\arraystretch}{1.35}")
    L.append(rf"  \rowcolors{{3}}{{{COL_ROW_ODD}}}{{white}}")
    L.append(r"  \resizebox{\textwidth}{!}{%")
    L.append("  \\begin{tabular}{" + col_spec + "}")
    L.append(r"  \toprule")

    # ── header row — dark navy; \cellcolor overrides \columncolor on rare cols
    head_bg  = "\\rowcolor{" + COL_HEAD_BG + "}"
    head_col = "\\color{" + COL_HEAD_FG + "}"

    def hcell(text, rare=False):
        override = "\\cellcolor{" + COL_HEAD_BG + "}" if rare else ""
        return override + head_col + "\\textbf{" + text + "}"

    L.append(
        "  " + head_bg
        + hcell("Work") + " & "
        + hcell("Portfolio") + " & "
        + hcell("Decision") + " & "
        + hcell("Sequential", rare=True) + " & "
        + hcell("Stochastic") + " & "
        + hcell("Contract", rare=True) + " & "
        + hcell("Termination", rare=True) + " \\\\"
    )
    L.append(
        "  " + head_bg
        + hcell("") + " & "
        + hcell("level") + " & "
        + hcell("type") + " & "
        + hcell("decisions", rare=True) + " & "
        + hcell("cash flows") + " & "
        + hcell("mechanics", rare=True) + " & "
        + hcell("settlement", rare=True) + " \\\\"
    )
    L.append(r"  \midrule")

    # ── body rows ────────────────────────────────────────────────────────────
    def sort_key(r):
        bucket_order = 0 if r["bucket"] == "highly_relevant" else 1
        try:
            yr = int(r["year"]) if r["year"] else 9999
        except ValueError:
            yr = 9999
        return (bucket_order, yr)

    for r in sorted(rows, key=sort_key):
        if r["found"]:
            work_cell = "\\citet{" + r["bib_key"] + "}"
        else:
            work_cell = (
                "% NOT FOUND: add citation_key to " + r["filename"] + "\n"
                "  \\textbf{[REF MISSING]}"
            )

        L.append(
            "  " + work_cell + " & "
            + _colour_sym(r["portfolio"]) + " & "
            + _colour_decision(r["decision"]) + " & "
            + _colour_sym(r["sequential"]) + " & "
            + _colour_sym(r["stochastic"]) + " & "
            + _colour_sym(r["contract"]) + " & "
            + _colour_sym(r["termination"]) + " \\\\"
        )

    # ── present-work row ─────────────────────────────────────────────────────
    L.append(r"  \cmidrule{1-7}")
    pw_bg   = "\\rowcolor{" + COL_PW_BG + "}"
    pw_col  = "\\color{" + COL_HEAD_FG + "}"
    pw_cell = "\\cellcolor{" + COL_PW_BG + "}"

    def pwcell(text, bold=False):
        inner = "\\textbf{" + text + "}" if bold else text
        return pw_col + inner

    L.append(
        "  " + pw_bg
        + pwcell("Present work", bold=True) + " & "
        + pw_cell + pwcell("\\checkmark") + " & "
        + pwcell("\\textbf{B}") + " & "
        + pw_cell + pwcell("\\checkmark") + " & "
        + pwcell("\\checkmark") + " & "
        + pw_cell + pwcell("\\checkmark") + " & "
        + pw_cell + pwcell("\\checkmark") + " \\\\"
    )

    # ── close ─────────────────────────────────────────────────────────────────
    L.append(r"  \bottomrule")
    L.append(r"  \end{tabular}%")
    L.append(r"  }")
    L.append(r"\end{table}")

    return "\n".join(L)



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
    parser.add_argument(
        "--max-rows", type=int, default=25,
        help=(
            "Maximum number of literature rows in the table (default: 30). "
            "Rows are selected to maximise informativeness and decision-type "
            "variety while prioritising rare columns (sequential, contract, "
            "termination). Set to 0 to include all rows."
        ),
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

    # ── Apply row limit ────────────────────────────────────────────────────
    rows = select_rows(rows, max_rows=args.max_rows)

    print_report(rows)

    tex      = render_latex(rows)
    out_path = tbl_dir / "synthesis_table.tex"
    out_path.write_text(tex, encoding="utf-8")

    print(f"LaTeX table written to:\n  {out_path}")
    print("\nUse in your document:")
    print(r"  \input{tables/synthesis_table.tex}")


if __name__ == "__main__":
    main()