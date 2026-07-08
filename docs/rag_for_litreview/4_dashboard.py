#!/usr/bin/env python3
"""
lit_review_dashboard.py

Builds a literature-review dashboard (charts + tables) from a folder of
per-paper JSON extraction files, shaped like:

{
  "metadata": {...}, "calibration_params": {...}, "thesis_relevance": {...},
  "key_variables": [...], "_source": {...}
}

Target article: "Development of RL Agent for Project Portfolio Budgeting
under Cashflow Uncertainties"

USAGE
-----
    python lit_review_dashboard.py --base-dir /path/to/lit_archive_passed

It expects two subfolders under --base-dir (edit BASE_SUBDIRS below if your
structure differs):
    <base-dir>/highly_relevant/structured/*.json
    <base-dir>/valuable/structured/*.json

OUTPUT
------
    <base-dir>/lit_review_dashboard_output/figures/*.png   (9 individual charts
                                                             + 1 combined dashboard)
    <base-dir>/lit_review_dashboard_output/tables/papers_summary.csv
    <base-dir>/lit_review_dashboard_output/tables/gap_analysis_report.md

Only standard scientific-Python packages are required: pandas, numpy, matplotlib.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG — edit these if your folder names / research framing differ
# --------------------------------------------------------------------------

BASE_SUBDIRS = [
    Path("highly_relevant") / "structured",
    Path("valuable") / "structured",
]

# Buckets used to build the "research gap" matrix (rows = problem domain,
# cols = methodological approach). Edit freely to match your taxonomy.
ROW_KEYWORDS = {
    "Cash Flow / Budgeting":     ["cash flow", "cashflow", "budget", "financing", "liquidity", "cash-flow"],
    "Portfolio Selection":       ["portfolio selection", "project selection", "portfolio management", "project portfolio"],
    "Scheduling":                ["schedul"],
    "Resource Allocation":       ["resource-constrained", "resource allocation", "resource constraint", "resource-limited"],
    "Risk & Uncertainty":        ["uncertain", "risk", "stochastic", "volatil"],
    "Other PM Topic":            [],  # fallback bucket
}

COL_KEYWORDS = {
    "Reinforcement Learning":     ["reinforcement learning", "rl agent", "q-learning", "deep rl",
                                   "markov decision", "policy gradient", "deep reinforcement"],
    "Math / MCDM Optimization":   ["optimization", "mcdm", "multi-criteria", "multi-objective",
                                   "linear programming", "integer programming", "genetic algorithm",
                                   "heuristic", "metaheuristic"],
    "Simulation":                 ["simulation", "monte carlo", "s-curve", "scurve", "discrete-event"],
    "Statistical / ML Forecast":  ["forecast", "regression", "machine learning", "neural network",
                                   "statistical", "time series", "predictive model"],
    "Other Method":               [],  # fallback bucket
}

# The cell in the gap matrix representing YOUR contribution — used to draw
# a highlight box on the heatmap.
YOUR_ROW = "Cash Flow / Budgeting"
YOUR_COL = "Reinforcement Learning"

FIGSIZE_DPI = 300
COLOR_MAIN = "#2E5A87"
COLOR_ACCENT = "#C0554B"
COLOR_NEUTRAL = "#8A8A8A"
PALETTE = ["#2E5A87", "#4C8CB0", "#7FB3D5", "#C0554B", "#E0A458", "#6C9A6B", "#8A8A8A"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# --------------------------------------------------------------------------
# LOADING
# --------------------------------------------------------------------------

def load_records(base_dir: Path) -> list[dict]:
    """Walk the configured subfolders and load every .json file found."""
    records = []
    for sub in BASE_SUBDIRS:
        folder = base_dir / sub
        if not folder.exists():
            print(f"  [warn] folder not found, skipping: {folder}")
            continue
        # tag which top-level bucket this came from (highly_relevant / valuable)
        bucket = sub.parts[0]
        for fp in sorted(folder.glob("*.json")):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                rec["_bucket"] = bucket
                rec["_filepath"] = str(fp)
                records.append(rec)
            except Exception as e:
                print(f"  [warn] failed to parse {fp}: {e}")
    return records


def extract_year(rec: dict) -> "int | None":
    """Try metadata.year_guess first, then fall back to a 4-digit year in the filename."""
    meta = rec.get("metadata", {}) or {}
    y = meta.get("year_guess", "")
    if y:
        m = re.search(r"(19|20)\d{2}", str(y))
        if m:
            return int(m.group())
    fname = (rec.get("_source", {}) or {}).get("filename", "")
    m = re.search(r"(19|20)\d{2}", fname)
    if m:
        return int(m.group())
    return None


def classify(text: str, keyword_map: dict, fallback: str) -> str:
    text = (text or "").lower()
    best_bucket, best_score = fallback, 0
    for bucket, kws in keyword_map.items():
        if not kws:
            continue
        score = sum(text.count(kw) for kw in kws)
        if score > best_score:
            best_bucket, best_score = bucket, score
    return best_bucket


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        meta = rec.get("metadata", {}) or {}
        rel = rec.get("thesis_relevance", {}) or {}
        src = rec.get("_source", {}) or {}
        keyvars = rec.get("key_variables", []) or []

        combined_text = " ".join([
            str(meta.get("problem", "")),
            str(meta.get("method", "")),
            str(meta.get("domain", "")),
            str(meta.get("model_type", "")),
            " ".join(keyvars),
        ])

        rows.append({
            "filename": src.get("filename", Path(rec.get("_filepath", "unknown")).name),
            "bucket": rec.get("_bucket", "unknown"),
            "year": extract_year(rec),
            "title_guess": meta.get("title_guess", ""),
            "problem": meta.get("problem", ""),
            "method": meta.get("method", ""),
            "domain": meta.get("domain", ""),
            "level": meta.get("level", ""),
            "model_type": meta.get("model_type", ""),
            "data_type": meta.get("data_type", ""),
            "supports_environment_design": bool(rel.get("supports_environment_design", False)),
            "supports_rl_agent": bool(rel.get("supports_rl_agent", False)),
            "supports_problem_motivation": bool(rel.get("supports_problem_motivation", False)),
            "supports_calibration": bool(rel.get("supports_calibration", False)),
            "key_finding": rel.get("key_finding", ""),
            "gap_identified": rel.get("gap_identified", ""),
            "key_variables": keyvars,
            "gap_row": classify(combined_text, ROW_KEYWORDS, "Other PM Topic"),
            "gap_col": classify(combined_text, COL_KEYWORDS, "Other Method"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# CHART BUILDERS  (each returns a matplotlib Figure)
# --------------------------------------------------------------------------

def chart_publication_trend(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    known = df.dropna(subset=["year"]).copy()
    known["year"] = known["year"].astype(int)
    unknown_count = df["year"].isna().sum()

    if known.empty:
        ax.text(0.5, 0.5, "No parsable years found", ha="center", va="center")
        return fig

    year_counts = known.groupby("year").size()
    full_range = range(int(year_counts.index.min()), int(year_counts.index.max()) + 1)
    year_counts = year_counts.reindex(full_range, fill_value=0)

    ax.bar(year_counts.index, year_counts.values, color=COLOR_MAIN, width=0.65, label="Papers per year")
    ax2 = ax.twinx()
    ax2.plot(year_counts.index, year_counts.cumsum().values, color=COLOR_ACCENT,
             marker="o", linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative count", color=COLOR_ACCENT)
    ax2.tick_params(axis="y", colors=COLOR_ACCENT)
    ax2.spines["top"].set_visible(False)

    ax.set_xlabel("Publication year")
    ax.set_ylabel("Number of papers")
    ax.set_title(f"Publication Trend Toward the Research Topic\n(n={len(known)} dated; "
                 f"{unknown_count} undated not shown)")
    ax.set_xticks(list(full_range))
    ax.set_xticklabels(list(full_range), rotation=45)
    fig.tight_layout()
    return fig


def chart_category_bar(df: pd.DataFrame, column: str, title: str, top_n: int = 10, horizontal=True):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = df[column].replace("", "Unspecified").fillna("Unspecified").value_counts().head(top_n)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]
    if horizontal:
        ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
        ax.set_xlabel("Number of papers")
    else:
        ax.bar(counts.index, counts.values, color=colors)
        ax.set_ylabel("Number of papers")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def chart_pie(df: pd.DataFrame, column: str, title: str):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    counts = df[column].replace("", "Unspecified").fillna("Unspecified").value_counts()
    ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%", startangle=90,
           colors=[PALETTE[i % len(PALETTE)] for i in range(len(counts))],
           wedgeprops={"edgecolor": "white", "linewidth": 1})
    ax.set_title(title)
    fig.tight_layout()
    return fig


def chart_relevance_support(df: pd.DataFrame):
    dims = ["supports_problem_motivation", "supports_environment_design",
            "supports_rl_agent", "supports_calibration"]
    labels = ["Problem\nMotivation", "Environment\nDesign", "RL Agent\nDesign", "Calibration\nParameters"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    buckets = sorted(df["bucket"].unique())
    bottom = np.zeros(len(dims))
    for i, b in enumerate(buckets):
        sub = df[df["bucket"] == b]
        vals = [sub[d].sum() for d in dims]
        ax.bar(labels, vals, bottom=bottom, label=b.replace("_", " ").title(),
               color=PALETTE[i % len(PALETTE)])
        bottom += np.array(vals)

    ax.set_ylabel("Number of papers")
    ax.set_title("How the Archive Supports Each Part of the Thesis")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def chart_gap_matrix(df: pd.DataFrame):
    rows = [r for r in ROW_KEYWORDS.keys()]
    cols = [c for c in COL_KEYWORDS.keys()]
    matrix = np.zeros((len(rows), len(cols)), dtype=int)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            matrix[i, j] = ((df["gap_row"] == r) & (df["gap_col"] == c)).sum()

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)

    for i in range(len(rows)):
        for j in range(len(cols)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "#222222"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10)

    # highlight the cell representing the user's own contribution
    if YOUR_ROW in rows and YOUR_COL in cols:
        ri, ci = rows.index(YOUR_ROW), cols.index(YOUR_COL)
        rect = plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1, fill=False,
                              edgecolor=COLOR_ACCENT, linewidth=3)
        ax.add_patch(rect)
        ax.text(ci, ri + 0.62, "your\ncontribution", ha="center", va="top",
                color=COLOR_ACCENT, fontsize=8, fontweight="bold")

    ax.set_title("Research Gap Matrix: Problem Domain × Methodological Approach\n"
                  "(highlighted cell = where this thesis sits)")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04, label="Papers in archive")
    fig.tight_layout()
    return fig


def chart_key_variables(df: pd.DataFrame, top_n: int = 15):
    counter = Counter()
    for vars_list in df["key_variables"]:
        counter.update(vars_list)
    top = counter.most_common(top_n)
    if not top:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.text(0.5, 0.5, "No key variables found", ha="center", va="center")
        return fig

    labels, values = zip(*top)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(labels[::-1], values[::-1], color=COLOR_MAIN)
    ax.set_xlabel("Frequency across papers")
    ax.set_title(f"Most Frequent Key Variables / Concepts (top {len(top)})")
    fig.tight_layout()
    return fig


def chart_source_split(df: pd.DataFrame):
    return chart_pie(df, "bucket", "Archive Composition: Highly Relevant vs. Valuable")


def build_combined_dashboard(df: pd.DataFrame, out_path: Path):
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35)

    panels = [
        (chart_publication_trend, (0, slice(0, 2))),
        (chart_source_split, (0, 2)),
        (lambda d: chart_category_bar(d, "domain", "Domain Distribution"), (1, 0)),
        (lambda d: chart_category_bar(d, "model_type", "Model Type Distribution"), (1, 1)),
        (lambda d: chart_category_bar(d, "level", "Academic Level"), (1, 2)),
        (chart_gap_matrix, (2, slice(0, 2))),
        (chart_key_variables, (2, 2)),
    ]

    # Combined dashboard rebuilt panel-by-panel using image placement
    # (simplest robust approach: render each chart separately then compose)
    fig.suptitle("Literature Archive Dashboard — RL Agent for Project Portfolio\n"
                 "Budgeting under Cashflow Uncertainties", fontsize=16, fontweight="bold")

    import io
    from matplotlib import image as mpimg

    for builder, pos in panels:
        sub_fig = builder(df)
        buf = io.BytesIO()
        sub_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(sub_fig)
        buf.seek(0)
        img = mpimg.imread(buf)
        ax = fig.add_subplot(gs[pos])
        ax.imshow(img)
        ax.axis("off")

    fig.savefig(out_path, dpi=FIGSIZE_DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# TABLES / REPORT
# --------------------------------------------------------------------------

def write_summary_table(df: pd.DataFrame, out_csv: Path):
    cols = ["filename", "bucket", "year", "title_guess", "domain", "problem", "method",
            "model_type", "level", "data_type", "gap_row", "gap_col",
            "supports_problem_motivation", "supports_environment_design",
            "supports_rl_agent", "supports_calibration", "key_finding", "gap_identified"]
    df[cols].sort_values(["year", "bucket"], na_position="last").to_csv(out_csv, index=False)


def write_gap_report(df: pd.DataFrame, out_md: Path):
    lines = ["# Gap Analysis Report", ""]
    lines.append(f"Total papers processed: **{len(df)}**")
    lines.append(f"- Highly relevant: {(df['bucket']=='highly_relevant').sum()}")
    lines.append(f"- Valuable: {(df['bucket']=='valuable').sum()}")
    lines.append("")

    your_cell = df[(df["gap_row"] == YOUR_ROW) & (df["gap_col"] == YOUR_COL)]
    lines.append(f"## Your positioning: {YOUR_ROW} × {YOUR_COL}")
    lines.append(f"Papers directly in this cell: **{len(your_cell)}**")
    if len(your_cell) == 0:
        lines.append("No paper in the archive combines these two dimensions — "
                      "this is the gap your thesis fills.")
    lines.append("")

    lines.append("## All identified gaps (verbatim from extraction)")
    for _, row in df.iterrows():
        if row["gap_identified"]:
            lines.append(f"- **{row['filename']}** ({row['bucket']}): {row['gap_identified']}")
    lines.append("")

    lines.append("## All key findings (verbatim from extraction)")
    for _, row in df.iterrows():
        if row["key_finding"]:
            lines.append(f"- **{row['filename']}** ({row['bucket']}): {row['key_finding']}")

    out_md.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build literature review dashboard from structured JSON archive.")
    parser.add_argument("--base-dir", type=str, default="lit_archive_passed",
                     help="Folder that contains the 'highly_relevant' and 'valuable' subfolders.")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="Output folder (default: <base-dir>/lit_review_dashboard_output)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else base_dir / "lit_review_dashboard_output"
    fig_dir = out_dir / "figures"
    tbl_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading records from {base_dir} ...")
    records = load_records(base_dir)
    print(f"  loaded {len(records)} JSON records")
    if not records:
        print("No records found — check --base-dir and folder structure. Exiting.")
        return

    df = records_to_dataframe(records)

    chart_specs = [
        ("01_publication_trend.png", lambda: chart_publication_trend(df)),
        ("02_domain_distribution.png", lambda: chart_category_bar(df, "domain", "Domain Distribution")),
        ("03_method_distribution.png", lambda: chart_category_bar(df, "method", "Method Distribution", top_n=12)),
        ("04_model_type_distribution.png", lambda: chart_category_bar(df, "model_type", "Model Type Distribution")),
        ("05_academic_level.png", lambda: chart_category_bar(df, "level", "Academic Level Distribution")),
        ("06_data_type_pie.png", lambda: chart_pie(df, "data_type", "Data Type Used in Papers")),
        ("07_relevance_support.png", lambda: chart_relevance_support(df)),
        ("08_research_gap_matrix.png", lambda: chart_gap_matrix(df)),
        ("09_key_variables_frequency.png", lambda: chart_key_variables(df)),
        ("10_source_folder_split.png", lambda: chart_source_split(df)),
    ]

    for fname, builder in chart_specs:
        fig = builder()
        fig.savefig(fig_dir / fname, dpi=FIGSIZE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")

    print("  building combined dashboard panel...")
    build_combined_dashboard(df, fig_dir / "00_combined_dashboard.png")

    write_summary_table(df, tbl_dir / "papers_summary.csv")
    write_gap_report(df, tbl_dir / "gap_analysis_report.md")

    print(f"\nDone. Output written to: {out_dir}")


if __name__ == "__main__":
    main()