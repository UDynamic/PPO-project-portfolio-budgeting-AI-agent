r"""
test_report.py  —  Unified report generator for the MILP baseline test suite.
                   Merges the table renderer and visualizer into one script.

Usage (standalone):
    python tests/milp/test_report.py
    python tests/milp/test_report.py --cases SP-1 MP-K
    python tests/milp/test_report.py --output path/to/report.tex

Called by test_milp.py via generate_report() after the pytest run.

Output:
    tests/milp/report.tex  (default)

The .tex file is self-contained and designed to be \input{} into a parent
document that already loads:
    booktabs, longtable, pgfplots, tikz, mdframed, enumitem, xcolor
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any

# ── paths ────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
CASES_FILE = HERE / "cases.json"
DEFAULT_OUT = HERE / "report.tex"

# ── group titles ─────────────────────────────────────────────────────────────
GROUP_TITLES = {
    "G1": "Group 1: Progress and Milestone Certification",
    "G2": "Group 2: Discounting and Timing Incentives",
    "G3": "Group 3: Advance Payments and Recovery",
    "G4": "Group 4: Retention",
    "G5": "Group 5: Counter-Trajectory Cases (Active)",
    "G6": "Group 6: Forced and Starvation Termination",
    "G7": "Group 7: Multi-Project Mixed Outcomes",
    "G8": "Group 8: Multi-Project Portfolio Optimisation",
    "G9": "Group 9: Staggered-Start Multi-Project",
}

# ── canonical case order for the report ──────────────────────────────────────
CASE_ORDER = [
    "SP-1", "SP-3", "SP-J",
    "SP-4", "MP-K", "MP-Kp", "SP-L", "SP-Lp", "SP-M", "SP-Mp",
    "SP-2", "SP-2a", "SP-2b", "SP-2c", "SP-2d", "SP-A", "SP-B",
    "SP-C", "SP-D",
    "SP-E", "SP-F", "SP-G",
    "SP-5", "SP-H", "SP-I",
    "MP-3", "MP-A", "MP-B",
    "MP-2", "MP-4", "MP-C", "MP-D",
    "MP-1", "MP-E",
]


# ════════════════════════════════════════════════════════════════════════════
# 1. HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _v(cell: Any) -> float:
    """Unwrap a toleranced cell or plain number."""
    if isinstance(cell, dict) and "v" in cell:
        return float(cell["v"])
    if cell is None:
        return 0.0
    return float(cell)


def _scalar(raw: Any, default: float = 0.0) -> float:
    return _v(raw) if raw is not None else default


def _fmt(x: float, dp: int = 3) -> str:
    """Format a float, suppressing tiny values to 0."""
    if abs(x) < 1e-9:
        return "0"
    if x == int(x):
        return str(int(x))
    fmt = f"{x:.{dp}f}".rstrip("0").rstrip(".")
    return fmt


def _tex_coords(pairs: list[tuple[float, float]]) -> str:
    """Return pgfplots coordinate string."""
    return " ".join(f"({_fmt(x)},{_fmt(y)})" for x, y in pairs)


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Generate n human-friendly tick values spanning [lo, hi]."""
    span = hi - lo
    if span < 1e-9:
        return [lo]
    raw_step = span / max(n - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    nice_steps = [1, 2, 2.5, 5, 10]
    step = magnitude * min(nice_steps, key=lambda s: abs(s * magnitude - raw_step))
    start = math.ceil(lo / step) * step
    ticks = []
    val = start
    while val <= hi + 1e-9:
        ticks.append(round(val, 10))
        val += step
    if not ticks:
        ticks = [lo, hi]
    return ticks


def _axis_range(vals: list[float], pad_frac: float = 0.12) -> tuple[float, float]:
    """Return axis [lo, hi] with padding."""
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, abs(hi) * 0.05, 1.0)
    return lo - span * pad_frac, hi + span * pad_frac


def _esc(s: str) -> str:
    """Minimal LaTeX escaping for case IDs (handle hyphens)."""
    return s.replace("_", r"\_")


# ════════════════════════════════════════════════════════════════════════════
# 2. FIGURE GENERATORS
# ════════════════════════════════════════════════════════════════════════════

# Panel geometry (cm) — shared across all figures
_W  = "5.5cm"   # panel width
_H  = "4.0cm"   # panel height
_H2 = "3.6cm"   # shorter height for multi-project rows
_DX = "5.8cm"   # horizontal offset between panels

def _scope(xshift: str = "") -> str:
    attr = f"[xshift={xshift}]" if xshift else ""
    return f"\\begin{{scope}}{attr}"


def _end_scope() -> str:
    return "\\end{scope}"


# ── Project figure (3-panel row per project) ──────────────────────────────

def _project_row(case_id: str, proj_idx: int, proj_entry: dict,
                 height: str = _H, is_multi: bool = False) -> str:
    """
    Generate one tikzpicture row (3 panels) for a single project.
    Panels: EVM signals | Milestone profile | SPI/CPI
    """
    ps      = proj_entry["params"]
    tl      = proj_entry["timeline"]
    BAC     = float(ps["BAC"])
    fi      = int(ps["fi"])
    si      = int(ps["si"])
    H_proj  = fi   # x-axis goes to fi+1
    xmax    = H_proj + 1
    label_suffix = f"{case_id.lower().replace('-','')}_p{proj_idx+1}"

    # ── collect timeline data ────────────────────────────────────────────
    t_vals   = [_v(r["t"])   for r in tl]
    P_vals   = [_v(r["P"])   for r in tl]
    SPI_vals = [_v(r["SPI"]) for r in tl]
    CPI_vals = [_v(r["CPI"]) for r in tl]
    x_vals   = [_v(r["x"])   for r in tl]
    BCWS_vals= [_v(r["BCWS"]) for r in tl]
    ACWP_vals= [_v(r["ACWP"]) for r in tl]  # fraction of BAC

    # Milestone thresholds
    ms_theta = list(ps["ms_theta"])
    ms_e     = list(ps["ms_e"])
    M        = len(ms_theta)

    # termination period (if any)
    term_t = ps.get("term_at")

    # inflows per period: R_net + R_ret
    inflows_by_t: dict[int, float] = {}
    advance_A = float(ps.get("A", 0) or 0)
    alpha     = float(ps.get("alpha", 0) or 0)
    if advance_A == 0 and alpha > 0:
        CP = float(ps["CP"])
        advance_A = alpha * CP

    for row in tl:
        t  = int(_v(row["t"]))
        rn = _v(row["R_net"])
        rr = _v(row["R_ret"])
        rt = _v(row["R_term"])
        inflows_by_t[t] = rn + rr + (rt if rt > 0 else 0)

    # ── EVM panel ────────────────────────────────────────────────────────
    font_size = "\\tiny" if is_multi else "\\scriptsize"
    lbl_font  = font_size

    # BCWS: linear from 0 to 1 over si..fi
    D_plan = int(ps["D_plan"])
    bcws_coords = [(0.0, 0.0)]
    for i, row in enumerate(tl):
        t_r   = _v(row["t"])
        bcws_r = _v(row["BCWS"])
        bcws_coords.append((t_r, bcws_r))
    # extend flat to xmax
    if bcws_coords:
        bcws_coords.append((xmax, bcws_coords[-1][1]))

    # BCWP (actual P)
    bcwp_coords = [(0.0, 0.0)]
    for row in tl:
        bcwp_coords.append((_v(row["t"]), _v(row["P"])))

    # ACWP (cumulative spend / BAC)
    acwp_coords = [(0.0, 0.0)]
    cum = 0.0
    for row in tl:
        cum += _v(row["x"]) / BAC if BAC > 0 else 0
        acwp_coords.append((_v(row["t"]), round(cum, 6)))

    # xtick list
    xtick_list = list(range(0, xmax + 1))
    xtick_str  = ",".join(str(x) for x in xtick_list)

    title_evm = f"Project {proj_idx+1} EVM" if is_multi else "EVM signals"
    title_ms  = f"P{proj_idx+1} $x$\\ /\\ inflows" if is_multi else "Milestone profile"
    title_spi = f"SPI / CPI" if True else "SPI / CPI"

    lines = []
    lines.append(f"% === Project {proj_idx+1} row for {case_id} ===")
    lines.append("\\begin{tikzpicture}[font=\\footnotesize]")

    # ── Panel 1: EVM ────────────────────────────────────────────────────
    lines.append(_scope())
    lines.append(f"\\begin{{axis}}[")
    lines.append(f"    name=evm{label_suffix},")
    lines.append(f"    width={_W}, height={height},")
    lines.append(f"    title={{\\footnotesize {title_evm}}},")
    lines.append(f"    xlabel={{Period $t$}}, ylabel={{Fraction of $\\BAC$}},")
    lines.append(f"    xmin=0, xmax={xmax}, ymin=0, ymax=1.30,")
    lines.append(f"    xtick={{{xtick_str}}},")
    lines.append(f"    ytick={{0,0.5,1}},")
    lines.append(f"    grid=major, grid style={{dashed,gray!25}},")
    lines.append(f"    tick label style={{font={lbl_font}}},")
    lines.append(f"    label style={{font={lbl_font}}}, title style={{font={lbl_font}}},")
    lines.append(f"    clip=false,")
    lines.append(f"  ]")
    # BCWS
    lines.append(f"  \\addplot[thick,blue!70!black] coordinates {{{_tex_coords(bcws_coords)}}};")
    # BCWP
    lines.append(f"  \\addplot[thick,green!55!black] coordinates {{{_tex_coords(bcwp_coords)}}};")
    # ACWP
    lines.append(f"  \\addplot[thick,orange!85!black] coordinates {{{_tex_coords(acwp_coords)}}};")
    # milestone thresholds
    for j, theta_j in enumerate(ms_theta):
        tname = f"$\\theta_{{{j+1}}}$"
        lines.append(
            f"  \\draw[green!60!black,dashed,thin] "
            f"(axis cs:0,{_fmt(theta_j)})--(axis cs:{xmax},{_fmt(theta_j)})"
            f" node[right,green!60!black,font=\\tiny] {{{tname}}};"
        )
    # fi label
    lines.append(f"  \\node[font=\\tiny,blue!60] at (axis cs:{fi},1.08) {{$f_i{{=}}{fi}$}};")
    # termination line
    if term_t is not None:
        lines.append(
            f"  \\draw[red!80!black,dashed,thick] "
            f"(axis cs:{term_t},0)--(axis cs:{term_t},1.30)"
            f" node[above,font=\\tiny,red!80!black] {{Term.\\,$t={term_t}$}};"
        )
    lines.append(f"\\end{{axis}}")
    lines.append(_end_scope())
    lines.append("")

    # ── Panel 2: Milestone profile (x outflows down, inflows up) ─────────
    # y range: symmetric ±BAC
    ms_ymax = BAC
    ms_ymin = -BAC
    # ticks at ±BAC, ±BAC/2, 0
    half = BAC / 2
    ytick_ms = f"{_fmt(-ms_ymax)},{_fmt(-half)},0,{_fmt(half)},{_fmt(ms_ymax)}"
    ytlbl_ms = (f"$-\\BAC$,${_fmt(-half)}$,$0$,${_fmt(half)}$,$\\BAC$")

    xmax_ms = xmax - 1  # exclude the trailing flat period

    # build spend bars (negative) and inflow bars (positive)
    spend_coords  = []
    inflow_coords = []
    # advance bar at t=si-1 (or t=0 for si=1)
    adv_t = si - 1
    if advance_A > 0:
        inflow_coords.append((adv_t, advance_A))

    for row in tl:
        t_r = int(_v(row["t"]))
        x_r = _v(row["x"])
        in_r = inflows_by_t.get(t_r, 0.0)
        # termination settlement shown as red bar going down
        rt_r = _v(row["R_term"])

        if x_r > 1e-6:
            spend_coords.append((t_r, -x_r))
        if in_r > 1e-6:
            inflow_coords.append((t_r, in_r))
        if rt_r < -1e-6:          # negative settlement = outflow
            spend_coords.append((t_r, rt_r))

    lines.append(_scope(f"{_DX}"))
    lines.append(f"\\begin{{axis}}[")
    lines.append(f"    name=ms{label_suffix},")
    lines.append(f"    width={_W}, height={height},")
    lines.append(f"    title={{\\footnotesize {title_ms}}},")
    lines.append(f"    xlabel={{Period $t$}}, ylabel={{Amount}},")
    lines.append(f"    xmin=-0.5, xmax={xmax_ms}.5, ymin={_fmt(-ms_ymax)}, ymax={_fmt(ms_ymax)},")
    lines.append(f"    xtick={{{','.join(str(x) for x in range(0, xmax_ms+1))}}},")
    lines.append(f"    ytick={{{ytick_ms}}},")
    lines.append(f"    yticklabels={{{ytlbl_ms}}},")
    lines.append(f"    grid=major, grid style={{dashed,gray!25}},")
    lines.append(f"    tick label style={{font={lbl_font}}},")
    lines.append(f"    label style={{font={lbl_font}}}, title style={{font={lbl_font}}},")
    lines.append(f"    axis x line=center, axis y line=left,")
    lines.append(f"    clip=false,")
    lines.append(f"  ]")

    if spend_coords:
        lines.append(
            f"  \\addplot[ybar,bar width=0.22cm,fill=red!60,draw=red!80]"
            f" coordinates {{{_tex_coords(spend_coords)}}};"
        )
    if inflow_coords:
        lines.append(
            f"  \\addplot[ybar,bar width=0.22cm,fill=msgreen!70,draw=msgreen]"
            f" coordinates {{{_tex_coords(inflow_coords)}}};"
        )

    # milestone labels
    certified_at: dict[int, int] = {}
    for row in tl:
        t_r = int(_v(row["t"]))
        for ms_num in row.get("ms_certified", []):
            certified_at[ms_num] = t_r

    for ms_num, t_cert in sorted(certified_at.items()):
        # find inflow at that period
        in_at = inflows_by_t.get(t_cert, 0.0)
        y_lbl = in_at + ms_ymax * 0.05 if in_at > 0 else ms_ymax * 0.08
        lines.append(
            f"  \\node[font=\\tiny,msgreen] at (axis cs:{t_cert},{_fmt(y_lbl)}) "
            f"{{MS{ms_num}}};"
        )
    # advance label
    if advance_A > 0:
        y_adv = advance_A + ms_ymax * 0.05
        lines.append(
            f"  \\node[font=\\tiny,msgreen] at (axis cs:{adv_t},{_fmt(y_adv)}) {{$A_i$}};"
        )
    lines.append(f"\\end{{axis}}")
    lines.append(_end_scope())
    lines.append("")

    # ── Panel 3: SPI / CPI ───────────────────────────────────────────────
    spi_cpi_vals = [v for v in SPI_vals + CPI_vals
                    if not math.isinf(v) and not math.isnan(v)]
    spi_max = max(spi_cpi_vals) if spi_cpi_vals else 2.0
    spi_max = max(spi_max * 1.15, 2.0)
    spi_max = math.ceil(spi_max * 2) / 2     # round to nearest 0.5

    spi_ticks = _nice_ticks(0, spi_max, 5)
    spi_tick_str = ",".join(_fmt(v) for v in spi_ticks)

    spi_pts = []
    cpi_pts = []
    for row in tl:
        t_r   = _v(row["t"])
        spi_r = _v(row["SPI"])
        cpi_r = _v(row["CPI"])
        if not math.isinf(spi_r) and not math.isnan(spi_r):
            spi_pts.append((t_r, min(spi_r, spi_max * 0.98)))
        if not math.isinf(cpi_r) and not math.isnan(cpi_r):
            cpi_pts.append((t_r, min(cpi_r, spi_max * 0.98)))

    lines.append(_scope(f"11.6cm"))
    lines.append(f"\\begin{{axis}}[")
    lines.append(f"    name=spicpi{label_suffix},")
    lines.append(f"    width={_W}, height={height},")
    lines.append(f"    title={{\\footnotesize SPI / CPI}},")
    lines.append(f"    xlabel={{Period $t$}}, ylabel={{Index}},")
    lines.append(f"    xmin=0, xmax={xmax}, ymin=0, ymax={_fmt(spi_max)},")
    lines.append(f"    xtick={{{xtick_str}}},")
    lines.append(f"    ytick={{{spi_tick_str}}},")
    lines.append(f"    grid=major, grid style={{dashed,gray!25}},")
    lines.append(f"    tick label style={{font={lbl_font}}},")
    lines.append(f"    label style={{font={lbl_font}}}, title style={{font={lbl_font}}},")
    lines.append(f"    clip=false,")
    lines.append(f"  ]")
    # reference line at 1
    lines.append(
        f"  \\draw[red!40,dashed,thin] (axis cs:0,1)--(axis cs:{xmax},1);"
    )
    if spi_pts:
        lines.append(
            f"  \\addplot[thick,green!60!black,mark=*,mark size=1.5pt]"
            f" coordinates {{{_tex_coords(spi_pts)}}};"
        )
    if cpi_pts:
        lines.append(
            f"  \\addplot[thick,orange!80!black,mark=square*,mark size=1.5pt]"
            f" coordinates {{{_tex_coords(cpi_pts)}}};"
        )
    lines.append(f"\\end{{axis}}")
    lines.append(_end_scope())
    lines.append("")

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


def figure_project(case_id: str, case: dict) -> str:
    """
    Full project figure block: one tikzpicture row per project.
    Multi-project cases get stacked rows with \\vspace between them.
    """
    n = len(case["projects"])
    is_multi = n > 1
    height = _H2 if is_multi else _H

    rows = []
    for pi, proj_entry in enumerate(case["projects"]):
        rows.append(_project_row(case_id, pi, proj_entry, height, is_multi))
        if pi < n - 1:
            rows.append("\\vspace{0.3cm}")
        rows.append("")

    label = f"fig:{case_id.lower().replace('-','')}_project"
    n_str = f"{n}-project" if is_multi else "single-project"
    caption = (
        f"{_esc(case_id)} project view ({n_str}). "
        f"Outcome: \\texttt{{{case['meta']['outcome']}}}."
    )
    Zstar = case["meta"].get("Zstar")
    if Zstar is not None:
        caption += f" $Z^*={_fmt(Zstar, 3)}$."

    out = ["\\begin{figure}[htbp]", "\\centering"]
    out.extend(rows)
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\end{figure}")
    return "\n".join(out)


# ── Portfolio figure (3-panel) ────────────────────────────────────────────

def figure_portfolio(case_id: str, case: dict) -> str:
    """
    Portfolio figure: Cash balance | Period cash flows | Cumul. disc. NCF
    All panels have dynamic axes derived from the actual data.
    """
    params   = case["params"]
    portfolio = case["portfolio"]
    H_raw    = params.get("H", params.get("horizon", 8))
    H        = int(_scalar(H_raw, 8))
    B0_raw   = params.get("B0", params.get("B1", 300))
    B1       = _scalar(B0_raw, 300.0)

    # Detect advance at t=0 so we can show it
    total_advance = 0.0
    for proj_entry in case["projects"]:
        ps = proj_entry["params"]
        adv = float(ps.get("A", 0) or 0)
        if adv == 0:
            alpha = float(ps.get("alpha", 0) or 0)
            CP    = float(ps.get("CP", ps.get("BAC", 0)) or 0)
            adv   = alpha * CP
        total_advance += adv

    # ── Cash balance data ─────────────────────────────────────────────────
    B_coords: list[tuple[float, float]] = []
    for row in portfolio:
        t_r = _v(row["t"])
        if t_r > H:
            break
        Bt_r = _v(row["B_t"])
        # Show opening and closing value within period
        B_coords.append((t_r, Bt_r))

    # Include starting cash (B1 + advance) at t=1 as a step entry
    B_all = [B1 + total_advance] + [_v(r["B_t"]) for r in portfolio if _v(r["t"]) <= H]
    B_lo, B_hi = _axis_range(B_all, pad_frac=0.10)
    B_ticks = _nice_ticks(B_lo, B_hi, 5)
    B_tick_str = ",".join(_fmt(v, 1) for v in B_ticks)
    xtick_port = ",".join(str(int(t)) for t in range(1, H + 1))

    # Build step-plot: show B before and after events within each period
    B_step: list[tuple[float, float]] = []
    B_cur = B1 + total_advance
    for row in portfolio:
        t_r = _v(row["t"])
        if t_r > H:
            break
        B_new = _v(row["B_t"])
        B_step.append((t_r, B_cur))     # opening
        B_step.append((t_r, B_new))     # closing (after events)
        B_cur = B_new

    # ── Period cash flows (inflows + outflows + net line) ─────────────────
    inflow_coords  = []
    outflow_coords = []
    net_coords     = [(0.0, 0.0)]   # start at zero

    # advance at t=0 as inflow
    if total_advance > 0:
        inflow_coords.append((0.0, total_advance))
        net_coords = [(0.0, total_advance)]

    for row in portfolio:
        t_r   = _v(row["t"])
        if t_r > H:
            break
        in_r  = _v(row["sum_inflow"])
        out_r = _v(row["sum_outflow"])
        net_r = _v(row["sum_net"])

        if in_r > 1e-6:
            inflow_coords.append((t_r, in_r))
        if out_r > 1e-6:
            outflow_coords.append((t_r, -out_r))
        net_coords.append((t_r, net_r))

    # Dynamic y range for flow panel
    flow_vals = (
        [y for _, y in inflow_coords]
        + [y for _, y in outflow_coords]
        + [0.0]
    )
    f_lo, f_hi = _axis_range(flow_vals, pad_frac=0.15)
    f_lo = min(f_lo, -1.0)
    f_hi = max(f_hi, 1.0)
    f_ticks = _nice_ticks(f_lo, f_hi, 6)
    f_tick_str = ",".join(_fmt(v) for v in f_ticks)

    # ── Cumulative discounted NCF ─────────────────────────────────────────
    cum_coords: list[tuple[float, float]] = []
    for row in portfolio:
        t_r = _v(row["t"])
        if t_r > H:
            break
        cum_coords.append((t_r, _v(row["cum_Z"])))

    if total_advance > 0 and cum_coords:
        # Prepend the advance contribution at t=1
        first_cum = _v(portfolio[0]["cum_Z"])
        cum_coords = [(1.0, first_cum)] + cum_coords[1:]

    cum_vals = [y for _, y in cum_coords] + [0.0]
    c_lo, c_hi = _axis_range(cum_vals, pad_frac=0.15)
    c_ticks = _nice_ticks(c_lo, c_hi, 5)
    c_tick_str = ",".join(_fmt(v) for v in c_ticks)

    label   = f"fig:{case_id.lower().replace('-','')}_portfolio"
    Zstar   = case["meta"].get("Zstar")
    caption = f"{_esc(case_id)} portfolio view."
    if Zstar is not None:
        caption += f" $Z^*={_fmt(Zstar, 3)}$."

    lbl_font = "\\scriptsize"
    xmax_port = H + 0.5
    xmin_flow = -0.5 if (total_advance > 0 or any(t < 1 for t, _ in net_coords)) else -0.5

    out = ["\\begin{figure}[htbp]", "\\centering",
           "\\begin{tikzpicture}[font=\\footnotesize]"]

    # ── Panel 1: Cash balance ─────────────────────────────────────────────
    out.append(_scope())
    out.append("\\begin{axis}[")
    out.append(f"    width={_W}, height={_H},")
    out.append(f"    title={{\\footnotesize Cash balance $B_t$}},")
    out.append(f"    xlabel={{Period $t$}}, ylabel={{Amount}},")
    out.append(f"    xmin=0.5, xmax={_fmt(xmax_port)},")
    out.append(f"    ymin={_fmt(B_lo, 1)}, ymax={_fmt(B_hi, 1)},")
    out.append(f"    xtick={{{xtick_port}}},")
    out.append(f"    ytick={{{B_tick_str}}},")
    out.append(f"    grid=major, grid style={{dashed,gray!25}},")
    out.append(f"    tick label style={{font={lbl_font}}},")
    out.append(f"    label style={{font={lbl_font}}}, title style={{font={lbl_font}}},")
    out.append(f"  ]")
    if B_step:
        out.append(
            f"  \\addplot[thick,cyan!60!black,mark=*,mark size=1.5pt]"
            f" coordinates {{{_tex_coords(B_step)}}};"
        )
    out.append(
        f"  \\draw[dashed,gray] (axis cs:0.5,{_fmt(B1)})--(axis cs:{_fmt(xmax_port)},{_fmt(B1)})"
        f" node[right,font=\\tiny,gray] {{$B_1$}};"
    )
    out.append("\\end{axis}")
    out.append(_end_scope())
    out.append("")

    # ── Panel 2: Period cash flows ────────────────────────────────────────
    xmax_flow = H + 0.5
    xtick_flow = ",".join(str(x) for x in range(0, H + 1))
    out.append(_scope(_DX))
    out.append("\\begin{axis}[")
    out.append(f"    width={_W}, height={_H},")
    out.append(f"    title={{\\footnotesize Period cash flows}},")
    out.append(f"    xlabel={{Period $t$}}, ylabel={{Net amount}},")
    out.append(f"    xmin={xmin_flow}, xmax={_fmt(xmax_flow)},")
    out.append(f"    ymin={_fmt(f_lo)}, ymax={_fmt(f_hi)},")
    out.append(f"    xtick={{{xtick_flow}}},")
    out.append(f"    ytick={{{f_tick_str}}},")
    out.append(f"    grid=major, grid style={{dashed,gray!25}},")
    out.append(f"    tick label style={{font={lbl_font}}},")
    out.append(f"    label style={{font={lbl_font}}}, title style={{font={lbl_font}}},")
    out.append(f"    axis x line=center, axis y line=left,")
    out.append(f"    clip=false,")
    out.append(f"  ]")
    if outflow_coords:
        out.append(
            f"  \\addplot[ybar,bar width=0.22cm,fill=red!60,draw=red!80]"
            f" coordinates {{{_tex_coords(outflow_coords)}}};"
        )
    if inflow_coords:
        out.append(
            f"  \\addplot[ybar,bar width=0.22cm,fill=msgreen!70,draw=msgreen]"
            f" coordinates {{{_tex_coords(inflow_coords)}}};"
        )
    if net_coords:
        out.append(
            f"  \\addplot[thick,blue!70!black,mark=o,mark size=1.5pt]"
            f" coordinates {{{_tex_coords(net_coords)}}};"
        )
    out.append("\\end{axis}")
    out.append(_end_scope())
    out.append("")

    # ── Panel 3: Cumulative discounted NCF ────────────────────────────────
    out.append(_scope("11.6cm"))
    out.append("\\begin{axis}[")
    out.append(f"    width={_W}, height={_H},")
    out.append(f"    title={{\\footnotesize Cumul.\\ discounted NCF}},")
    out.append(f"    xlabel={{Period $t$}}, ylabel={{Amount}},")
    out.append(f"    xmin=0.5, xmax={_fmt(xmax_port)},")
    out.append(f"    ymin={_fmt(c_lo)}, ymax={_fmt(c_hi)},")
    out.append(f"    xtick={{{xtick_port}}},")
    out.append(f"    ytick={{{c_tick_str}}},")
    out.append(f"    grid=major, grid style={{dashed,gray!25}},")
    out.append(f"    tick label style={{font={lbl_font}}},")
    out.append(f"    label style={{font={lbl_font}}}, title style={{font={lbl_font}}},")
    out.append(f"  ]")
    if cum_coords:
        out.append(
            f"  \\addplot[thick,blue!70!black,mark=*,mark size=1.5pt]"
            f" coordinates {{{_tex_coords(cum_coords)}}};"
        )
    out.append("\\end{axis}")
    out.append(_end_scope())
    out.append("")

    out.append("\\end{tikzpicture}")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\end{figure}")
    return "\n".join(out)


# ════════════════════════════════════════════════════════════════════════════
# 3. DATA TABLE GENERATOR
# ════════════════════════════════════════════════════════════════════════════

def _data_table_project(proj_idx: int, proj_entry: dict, BAC: float) -> str:
    """Per-project timeline table (one row per period)."""
    tl = proj_entry["timeline"]

    header = (
        r"\begin{center}""\n"
        r"\begin{tabular}{rrrrrrrrrrl}""\n"
        r"\toprule""\n"
        r"$t$ & $x$ & $P$ & BCWS & BCWP & ACWP & SPI & CPI & $\tau$ & MS & $R_{\mathrm{net}}$\\""\n"
        r"\midrule"
    )
    rows = [header]
    for row in tl:
        t     = int(_v(row["t"]))
        x     = _v(row["x"])
        P     = _v(row["P"])
        bcws  = _v(row["BCWS"])
        bcwp  = _v(row["BCWP"])
        acwp  = _v(row["ACWP"])
        spi   = _v(row["SPI"])
        cpi   = _v(row["CPI"])
        tau   = int(_v(row["tau_rem"]))
        ms    = ",".join(str(m) for m in row.get("ms_certified", [])) or "---"
        rnet  = _v(row["R_net"])
        rret  = _v(row["R_ret"])
        rterm = _v(row["R_term"])

        total_in = rnet + rret + (rterm if rterm != 0 else 0)
        in_str   = _fmt(total_in) if abs(total_in) > 1e-6 else "0"

        rows.append(
            f"{t} & {_fmt(x)} & {_fmt(P,3)} & {_fmt(bcws,3)} & {_fmt(bcwp,3)} & "
            f"{_fmt(acwp,3)} & {_fmt(spi,3)} & {_fmt(cpi,3)} & {tau} & "
            f"\\texttt{{{ms}}} & {in_str} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{center}")
    return "\n".join(rows)


def _data_table_portfolio(case: dict) -> str:
    """Portfolio strip table."""
    port = case["portfolio"]
    params = case["params"]
    H_raw  = params.get("H", params.get("horizon", 8))
    H      = int(_scalar(H_raw, 8))

    header = (
        r"\begin{center}""\n"
        r"\begin{tabular}{rrrrrrr}""\n"
        r"\toprule""\n"
        r"$t$ & $B_t$ & $\Sigma\mathrm{In}$ & $\Sigma\mathrm{Out}$ & "
        r"$\Sigma\mathrm{Net}$ & $\gamma^{t-1}\mathrm{Net}$ & $Z_t^{\mathrm{cum}}$\\""\n"
        r"\midrule"
    )
    rows = [header]
    for row in port:
        t_r = _v(row["t"])
        if t_r > H:
            break
        rows.append(
            f"{int(t_r)} & {_fmt(_v(row['B_t']),1)} & "
            f"{_fmt(_v(row['sum_inflow']))} & {_fmt(_v(row['sum_outflow']))} & "
            f"{_fmt(_v(row['sum_net']))} & {_fmt(_v(row['disc_net']))} & "
            f"{_fmt(_v(row['cum_Z']))} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{center}")
    return "\n".join(rows)


# ════════════════════════════════════════════════════════════════════════════
# 4. VERIFICATION TABLE
# ════════════════════════════════════════════════════════════════════════════

# Test result record: pass/fail per category per case
# Keys match the test method names from test_milp.py
TEST_CATS = [
    ("budget",    "Budget alloc."),
    ("progress",  "Progress"),
    ("evm",       "EVM signals"),
    ("termctr",   "Term.~counter"),
    ("mscert",    "MS cert."),
    ("payment",   "Payment id."),
    ("cashbal",   "Cash bal."),
    ("zstar",     "$Z^*$"),
    ("outcome",   "Outcome"),
    ("monotone",  "P monotone"),
    ("advance",   "Advance"),
]

# placeholder symbol constants
_PASS  = r"{\color{green!60!black}\checkmark}"
_FAIL  = r"{\color{red!80!black}\ding{55}}"
_SKIP  = r"{\color{gray}$\circ$}"
_NA    = r"{\color{gray}---}"


def verification_table(
    results: dict[str, dict[str, str]] | None,
    cases: dict,
) -> str:
    """
    Generate the full verification table as a longtable.

    results: dict mapping case_id → {cat_key → "pass"/"fail"/"skip"/"na"}
             If None, renders the table with placeholder "---" entries
             (for standalone generation without a live test run).
    """
    ordered = [cid for cid in CASE_ORDER if cid in cases]

    cols = "l" + "c" * len(TEST_CATS)
    cat_headers = " & ".join(f"\\rotatebox{{70}}{{\\scriptsize {label}}}"
                              for _, label in TEST_CATS)

    out = []
    out.append(r"\begin{longtable}{" + cols + "}")
    out.append(r"\toprule")
    out.append(f"\\textbf{{Case}} & {cat_headers} \\\\")
    out.append(r"\midrule")
    out.append(r"\endfirsthead")
    out.append(r"\toprule")
    out.append(f"\\textbf{{Case}} & {cat_headers} \\\\")
    out.append(r"\midrule")
    out.append(r"\endhead")
    out.append(r"\midrule \multicolumn{" + str(len(TEST_CATS)+1) +
               r"}{r}{\small\itshape continued \ldots} \\")
    out.append(r"\endfoot")
    out.append(r"\bottomrule")
    out.append(r"\endlastfoot")

    prev_group = None
    for cid in ordered:
        case  = cases[cid]
        group = case["meta"]["group"]

        if group != prev_group:
            out.append(
                r"\multicolumn{" + str(len(TEST_CATS)+1) + r"}{l}{"
                r"\small\textit{" + GROUP_TITLES.get(group, group) + r"}} \\"
            )
            prev_group = group

        if results is not None:
            case_res = results.get(cid, {})
        else:
            case_res = {}

        Zstar = case["meta"].get("Zstar")
        cells = []
        for key, _ in TEST_CATS:
            if key == "zstar" and Zstar is None:
                cells.append(_SKIP)
                continue
            status = case_res.get(key, "na")
            if   status == "pass": cells.append(_PASS)
            elif status == "fail": cells.append(_FAIL)
            elif status == "skip": cells.append(_SKIP)
            else:                  cells.append(_NA)

        row = f"\\texttt{{{_esc(cid)}}} & " + " & ".join(cells) + " \\\\"
        out.append(row)

    out.append(r"\end{longtable}")
    return "\n".join(out)


# ════════════════════════════════════════════════════════════════════════════
# 5. PER-CASE BLOCK
# ════════════════════════════════════════════════════════════════════════════

def case_block(case_id: str, case: dict) -> str:
    """
    Full .tex block for one case:
    subsection heading + parameter table + data tables + figures.
    """
    meta   = case["meta"]
    params = case["params"]
    n      = len(case["projects"])
    Zstar  = meta.get("Zstar")
    outcome = meta.get("outcome", "")

    # ── subsection heading ────────────────────────────────────────────────
    heading = f"\\subsection*{{{_esc(case_id)}  (Group~{meta['group']})}}"

    # ── parameter summary table ───────────────────────────────────────────
    def _ps(k: str, default: str = "---") -> str:
        v = params.get(k)
        if v is None:
            return default
        return str(int(_scalar(v))) if isinstance(_scalar(v), float) and _scalar(v) == int(_scalar(v)) else _fmt(_scalar(v))

    # Collect across projects
    all_BAC = [float(p["params"]["BAC"]) for p in case["projects"]]
    all_CP  = [float(p["params"]["CP"])  for p in case["projects"]]
    H_raw   = params.get("H", params.get("horizon", 8))
    H       = int(_scalar(H_raw, 8))

    param_rows = []
    param_rows.append(f"$n$,\\;$H$ & ${n}$,\\;${H}$")
    if n == 1:
        ps0 = case["projects"][0]["params"]
        param_rows.append(f"$\\BAC,\\;CP$ & ${_fmt(all_BAC[0])},\\;{_fmt(all_CP[0])}$")
        param_rows.append(f"$s_i,\\;f_i,\\;D^{{\\mathrm{{plan}}}}$ & "
                          f"${ps0['si']},\\;{ps0['fi']},\\;{ps0['D_plan']}$")
        eta_s = ps0.get("eta")
        if eta_s is not None:
            param_rows.append(f"$\\bar\\eta$ & ${_fmt(float(eta_s))}$")
        alpha_s = ps0.get("alpha", 0)
        rho_s   = ps0.get("rho", 0)
        A_s     = ps0.get("A", 0)
        if float(alpha_s or 0) > 0:
            param_rows.append(f"$\\alpha,\\;A$ & ${_fmt(float(alpha_s))},\\;{_fmt(float(A_s or 0))}$")
        if float(rho_s or 0) > 0:
            param_rows.append(f"$\\rho$ & ${_fmt(float(rho_s))}$")
        ms_theta = ps0.get("ms_theta", [])
        ms_phi   = ps0.get("ms_phi", [])
        ms_e     = ps0.get("ms_e", [])
        M = len(ms_theta)
        if M:
            theta_str = ",".join(_fmt(v) for v in ms_theta)
            phi_str   = ",".join(_fmt(v) for v in ms_phi)
            e_str     = ",".join(str(v) for v in ms_e)
            param_rows.append(f"$M,\\;\\theta$ & ${M},\\;({theta_str})$")
            param_rows.append(f"$\\phi,\\;e_{{i,j}}$ & $({phi_str}),\\;({e_str})$")
        mu_s   = ps0.get("mu", 0.30)
        Omega_s = ps0.get("Omega", 1)
        tau_s  = ps0.get("tau_tol", 2)
        param_rows.append(f"$\\mu,\\;\\Omega,\\;\\tau^{{\\mathrm{{tol}}}}$ & "
                          f"${_fmt(float(mu_s))},\\;{Omega_s},\\;{tau_s}$")
    else:
        bacs = ", ".join(_fmt(b) for b in all_BAC)
        cps  = ", ".join(_fmt(c) for c in all_CP)
        param_rows.append(f"$\\BAC_i$ & $({bacs})$")
        param_rows.append(f"$CP_i$   & $({cps})$")

    B0_raw = params.get("B0", params.get("B1", 300))
    B0     = _scalar(B0_raw, 300.0)
    param_rows.append(f"$B_1,\\;\\gamma$ & ${_fmt(B0)},\\;0.95$")
    if Zstar is not None:
        param_rows.append(f"$Z^*$ & ${_fmt(Zstar, 3)}$")
    param_rows.append(f"Outcome & \\texttt{{{outcome}}}")

    param_body = " \\\\\n".join(param_rows) + " \\\\"
    param_table = (
        "\\begin{center}\n"
        "\\begin{tabular}{ll}\n"
        "\\toprule\n"
        + param_body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{center}"
    )

    # ── data tables ───────────────────────────────────────────────────────
    data_tables = []
    for pi, proj_entry in enumerate(case["projects"]):
        BAC = float(proj_entry["params"]["BAC"])
        if n > 1:
            data_tables.append(f"\\paragraph{{Project {pi+1} timeline.}}")
        data_tables.append(_data_table_project(pi, proj_entry, BAC))

    data_tables.append("\\paragraph{Portfolio strip.}")
    data_tables.append(_data_table_portfolio(case))

    # ── figures ───────────────────────────────────────────────────────────
    fig_proj = figure_project(case_id, case)
    fig_port = figure_portfolio(case_id, case)

    parts = [
        heading,
        param_table,
        "",
        "\n".join(data_tables),
        "",
        fig_proj,
        "",
        fig_port,
        "",
        "\\newpage",
    ]
    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# 6. FULL REPORT ASSEMBLER
# ════════════════════════════════════════════════════════════════════════════

_PREAMBLE_COLORS = r"""
\definecolor{noteblu}{RGB}{60,60,180}
\definecolor{msgreen}{RGB}{0,140,60}
\definecolor{allocblue}{RGB}{100,160,220}
"""

_TIKZ_STYLES = r"""
\tikzset{
  every node/.style={font=\scriptsize},
  inflow/.style={-{Latex[length=2.5mm]}, thick, green!60!black},
  outflow/.style={-{Latex[length=2.5mm]}, thick, red!70!black},
}
"""

_LEGEND_TABLE = r"""
\section*{Figure Legends}
All figures share the following visual language.
\medskip

\begin{tabular}{@{}p{4.5cm} p{9cm}@{}}
\toprule
\textbf{Panel} & \textbf{Meaning} \\
\midrule
\textbf{EVM (left)} &
  \begin{tikzpicture}[baseline=(current bounding box.center)]
    \draw[thick,blue!70!black]   (0,0)    -- (1,0)    node[right,font=\tiny]{BCWS};
    \draw[thick,green!55!black]  (0,-0.4) -- (1,-0.4) node[right,font=\tiny]{BCWP};
    \draw[thick,orange!85!black] (0,-0.8) -- (1,-0.8) node[right,font=\tiny]{ACWP};
    \draw[green!60!black,dashed,thin] (0,-1.2) -- (1,-1.2) node[right,font=\tiny]{$\theta_j$ thresholds};
  \end{tikzpicture} \\
\cmidrule{2-2}
& Red dashed vertical = termination period. \\
\midrule
\textbf{Profile (centre)} &
  \begin{tikzpicture}[baseline=(current bounding box.center)]
    \draw[fill=red!60,draw=red!80] (0,0) rectangle (0.3,-0.4);
    \node[font=\tiny,anchor=west] at (0.4,-0.2) {$x_{i,t}$ spend (down)};
    \draw[fill=msgreen!70,draw=msgreen] (0,-0.7) rectangle (0.3,-1.1);
    \node[font=\tiny,anchor=west] at (0.4,-0.9) {Inflows: $R^{\mathrm{net}}$, $A_i$, $R^{\mathrm{ret}}$ (up)};
  \end{tikzpicture} \\
\midrule
\textbf{SPI/CPI (right)} &
  $\bullet$ = SPI (green),\quad $\blacksquare$ = CPI (orange),\quad
  red dashed = reference 1. \\
\midrule
\textbf{Portfolio left} &
  Cyan step-line = $B_t$; grey dashed = $B_1$ reference. \\
\midrule
\textbf{Portfolio centre} &
  Green bars = inflow, red bars = outflow,
  blue line with $\circ$ = net per period. \\
\midrule
\textbf{Portfolio right} &
  Blue line with $\bullet$ = cumulative discounted NCF. \\
\bottomrule
\end{tabular}

\bigskip
X-axes scale to each case's horizon $H$.
Middle project panel uses symmetric $\pm\BAC_i$ range.
Portfolio panels use data-driven ranges.
"""

_CONVENTIONS = r"""
\section*{Conventions}
\begin{itemize}
  \item $\gamma = 0.95$ throughout.
  \item Advance $A_i$ arrives at $t = s_i - 1$ (period~$0$ when $s_i=1$).
  \item Payment identity: $A_i + \sum_j R_{i,j}^{\mathrm{net}} + R_i^{\mathrm{ret}} = CP_i$
        checked for completed projects only.
  \item Retention released as a lump at the period the final milestone certifies.
  \item Advance recovery deductions: $\delta^{\mathrm{rec}} \phi_j CP_i$
        (pre-retention base, FIDIC convention).
\end{itemize}
"""


def build_report(
    cases: dict,
    case_ids: list[str] | None,
    results: dict | None = None,
) -> str:
    r"""
    Assemble the complete .tex report body (no \documentclass — designed
    to be \input{} from a parent document).

    cases:    full cases dict from cases.json
    case_ids: ordered list of case IDs to include (None = all in CASE_ORDER)
    results:  test results from the pytest run (None = placeholder table)
    """
    ordered = [
        cid for cid in (case_ids or CASE_ORDER) if cid in cases
    ]

    lines = []

    # ── colour / style macros ─────────────────────────────────────────────
    lines.append(_PREAMBLE_COLORS.strip())
    lines.append("")
    lines.append(_TIKZ_STYLES.strip())
    lines.append("")

    lines.append(r"\newcommand{\BAC}{\mathrm{BAC}}")
    lines.append(r"\newcommand{\EAC}{\mathrm{EAC}}")
    lines.append("")

    # ── verification table ────────────────────────────────────────────────
    lines.append(r"\section*{Test Verification Summary}")
    lines.append(
        r"The table below shows pass (\checkmark), fail (\ding{55}), "
        r"skip ($\circ$), or not-applicable (---) for each of the "
        r"11 test categories across all cases. "
        r"Results are produced by \texttt{test\_milp.py} calling "
        r"\texttt{baselines/milp.py}."
    )
    lines.append("")
    lines.append(verification_table(results, cases))
    lines.append("")
    lines.append(r"\newpage")
    lines.append("")

    # ── legends and conventions ───────────────────────────────────────────
    lines.append(_LEGEND_TABLE.strip())
    lines.append("")
    lines.append(r"\newpage")
    lines.append("")
    lines.append(_CONVENTIONS.strip())
    lines.append("")
    lines.append(r"\newpage")
    lines.append("")

    # ── per-case blocks, grouped ──────────────────────────────────────────
    lines.append(r"\section*{Cases}")
    prev_group = None
    for cid in ordered:
        case  = cases[cid]
        group = case["meta"]["group"]
        if group != prev_group:
            title = GROUP_TITLES.get(group, group)
            lines.append(f"\n\\subsection*{{{title}}}")
            lines.append("")
            prev_group = group
        lines.append(case_block(cid, case))
        lines.append("")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 7. PUBLIC API  (called by test_milp.py)
# ════════════════════════════════════════════════════════════════════════════

def generate_report(
    results: dict | None = None,
    case_ids: list[str] | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """
    Load cases.json, assemble the .tex report, write it to disk.

    Parameters
    ----------
    results :
        dict mapping case_id → {cat_key → "pass"/"fail"/"skip"/"na"}.
        Pass None to generate the report with placeholder table cells.
    case_ids :
        Ordered list of case IDs to include.  None = all in CASE_ORDER.
    output_path :
        Destination .tex path.  None → tests/milp/report.tex.

    Returns
    -------
    Path to the written file.
    """
    with open(CASES_FILE) as f:
        cases = json.load(f)

    out_path = Path(output_path) if output_path else DEFAULT_OUT
    body     = build_report(cases, case_ids, results)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# 8. CLI
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate report.tex from cases.json"
    )
    parser.add_argument(
        "--cases", nargs="*", metavar="ID",
        help="Case IDs to include (default: all)",
    )
    parser.add_argument(
        "--output", metavar="PATH", default=None,
        help=f"Output .tex path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    out = generate_report(
        results=None,
        case_ids=args.cases or None,
        output_path=args.output,
    )
    print(f"Written: {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()