# ui.py
#
# Textual TUI for the Portfolio Budgeting Environment.
# Run with:  python ui.py

from __future__ import annotations

import importlib
import sqlite3
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button, DataTable, Footer, Header, Input,
    Label, RichLog, Static, TabbedContent, TabPane,
)
from rich.text import Text
from rich.table import Table
from rich.console import Console
from rich import box as rich_box

from db_init import init_db
from env import PortfolioEnv


# ─────────────────────────────────────────────────────────────
# CONFIG REGISTRY
# ─────────────────────────────────────────────────────────────

CONFIG_REGISTRY = {
    "1": ("config_seed_sp", "seed_single_project", "Single project  — CFG-SINGLE-001"),
    "2": ("config_seed_dp", "seed_dual_project",   "Dual project    — CFG-DUAL-001"),
}


# ─────────────────────────────────────────────────────────────
# FORMAT HELPERS
# ─────────────────────────────────────────────────────────────

def _fmt_f4(v) -> str:
    return f"{v:.4f}" if v is not None else "—"

def _fmt_f2(v) -> str:
    return f"{v:.2f}" if v is not None else "—"

def _fmt_pct(v) -> str:
    return f"{v*100:.1f}%" if v is not None else "—"

def _fmt_int(v) -> str:
    return str(int(v)) if v is not None else "—"

def _fmt_mu(v) -> str:
    return f"{v:,.2f}" if v is not None else "—"

def _fmt_bool_breach(v) -> str:
    return "✘" if v else "✔"

def _fmt_status(v) -> str:
    return (v or "—").upper()


# ─────────────────────────────────────────────────────────────
# COLUMN DEFINITIONS — grouped
#
# Each entry: (col_id, header, width, fmt_fn, group)
# Groups:
#   "id"      — identity
#   "alloc"   — allocation & decision  (periodic + cumulative)
#   "evm"     — EVM metrics
# ─────────────────────────────────────────────────────────────

PROJECT_COLUMNS: list[tuple[str, str, int, callable, str]] = [

    # ── identity ─────────────────────────────────────────────
    ("t",                    "t",           4,  _fmt_int,          "id"),
    ("status",               "Status",      12, _fmt_status,       "id"),

    # ── Allocation & Decision ─────────────────────────────────
    # periodic
    ("inflow_period",        "Inflow",      11, _fmt_mu,           "alloc"),
    ("outflow_period",       "Outflow",     11, _fmt_mu,           "alloc"),
    ("net_cf_period",        "Net CF",      11, _fmt_mu,           "alloc"),
    ("cash_deficit_period",  "Deficit",     10, _fmt_mu,           "alloc"),
    ("allocation",           "Alloc",       11, _fmt_mu,           "alloc"),
    ("interest_cost",        "Interest",    11, _fmt_mu,           "alloc"),
    ("budget_draw",          "BudgetDraw",  11, _fmt_mu,           "alloc"),
    # cumulative
    ("inflow_cum",           "∑Inflow",     11, _fmt_mu,           "alloc"),
    ("outflow_cum",          "∑Outflow",    11, _fmt_mu,           "alloc"),
    ("net_cf_cum",           "∑NetCF",      11, _fmt_mu,           "alloc"),
    ("cash_deficit_cum",     "∑Deficit",    10, _fmt_mu,           "alloc"),
    ("alloc_cum",            "∑Alloc",      11, _fmt_mu,           "alloc"),
    ("interest_cum",         "∑Interest",   11, _fmt_mu,           "alloc"),
    ("budget_draw_cum",      "∑BudgetDraw", 12, _fmt_mu,           "alloc"),

    # ── EVM metrics ───────────────────────────────────────────
    ("progress",             "Progress",    10, _fmt_f4,           "evm"),
    ("progress_plan",        "Plan",        10, _fmt_f4,           "evm"),
    ("plan_deviation",       "Deviation",   10, _fmt_f4,           "evm"),
    ("spi",                  "SPI(t)",       9, _fmt_f4,           "evm"),
    ("cpi",                  "CPI",          9, _fmt_f4,           "evm"),
    ("tcpi",                 "TCPI",         9, _fmt_f4,           "evm"),
    ("eac",                  "EAC",         12, _fmt_mu,           "evm"),
    ("eac_bac",              "EAC/BAC",     10, _fmt_f4,           "evm"),
    ("schedule_slip",        "SchedSlip",   10, _fmt_f2,           "evm"),
    ("forecast_finish",      "FcstFinish",  11, _fmt_f2,           "evm"),
]

COLUMN_IDS  = [c[0] for c in PROJECT_COLUMNS]
COLUMN_HDRS = [c[1] for c in PROJECT_COLUMNS]
COLUMN_FMTS = {c[0]: c[3] for c in PROJECT_COLUMNS}
COLUMN_GRP  = {c[0]: c[4] for c in PROJECT_COLUMNS}

_GRP_ORDER  = ["id", "alloc", "evm"]
_GRP_LABELS = {
    "id":    "Identity",
    "alloc": "Allocation & Decision",
    "evm":   "EVM Metrics",
}
_GRP_COLORS = {
    "id":    "bold white",
    "alloc": "bold cyan",
    "evm":   "bold yellow",
}

# Termination fields rendered in the dedicated status box, not the DataTable
TERM_FIELDS = [
    ("cure_remaining",    "Cure Remaining",   _fmt_int,         "bold white"),
    ("breach_idle",       "Idle Breach",      _fmt_bool_breach, None),
    ("breach_deviation",  "Deviation Breach", _fmt_bool_breach, None),
    ("breach_schedule",   "Schedule Breach",  _fmt_bool_breach, None),
    ("breach_cost",       "Cost Breach",      _fmt_bool_breach, None),
    ("breach_deadline",   "Deadline Breach",  _fmt_bool_breach, None),
    ("any_breach",        "Any Breach",       _fmt_bool_breach, None),
]


# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────

CSS = """
Screen {
    background: $surface;
}

/* ── selector screen ── */
#selector-container {
    align: center middle;
    height: 1fr;
}
#selector-box {
    width: 60;
    border: double $primary;
    padding: 2 4;
    background: $panel;
}
#selector-box Label {
    margin-bottom: 1;
}
#selector-box Button {
    width: 100%;
    margin-bottom: 1;
}

/* ── top bar ── */
#top-bar {
    height: 3;
    background: $panel;
    border-bottom: solid $primary;
    padding: 0 2;
    align: left middle;
}

/* ── tabs ── */
TabbedContent {
    height: 1fr;
}
TabPane {
    padding: 0 1;
}

/* ── portfolio tab ── */
#stats-panel {
    height: auto;
    border: solid $primary-darken-2;
    padding: 0 1;
    background: $panel;
}
#cashflow-log {
    border: solid $primary-darken-2;
    background: $panel;
    height: 1fr;
    min-height: 8;
}
#alloc-panel {
    height: auto;
    border: solid $accent;
    padding: 1;
    background: $panel;
}
#submit-btn {
    margin-top: 1;
    width: 20;
}

/* ── project tab ── */
.proj-tab-container {
    height: 1fr;
    layout: vertical;
    overflow-y: auto;
}

/* top two panels side by side */
.proj-top-row {
    height: auto;
    layout: horizontal;
    margin-bottom: 1;
}

/* static params table — left column */
.proj-static-panel {
    width: 1fr;
    border: solid $primary-darken-2;
    background: $panel;
    padding: 0 1;
}

/* right column stacks payment profile + termination box */
.proj-right-col {
    width: 2fr;
    layout: vertical;
}

/* milestone / payment profile table */
.proj-milestone-panel {
    height: auto;
    border: solid $primary-darken-2;
    background: $panel;
    padding: 0 1;
    margin-bottom: 1;
}

/* termination / boundary status box */
.proj-term-panel {
    height: auto;
    border: solid $warning-darken-1;
    background: $panel;
    padding: 0 1;
}

/* section title inside a panel */
.section-title {
    text-style: bold;
    color: $text-muted;
    padding: 0 0 1 0;
}

/* scrollable static widget for Rich tables */
.rich-static {
    height: auto;
    overflow: auto;
}

/* group header row */
.group-header {
    height: 1;
    background: $panel-darken-1;
    padding: 0 1;
}

/* the main timestep DataTable */
.proj-datatable-container {
    height: auto;
    min-height: 20;
    border: solid $primary-darken-2;
}

DataTable {
    height: auto;
    min-height: 20;
}

/* ── shared helpers ── */
.panel-title {
    color: $text-muted;
    text-style: bold;
    margin-bottom: 1;
}
.bad  { color: $error; }
.dim  { color: $text-muted; }
"""


# ─────────────────────────────────────────────────────────────
# CELL STYLING
# ─────────────────────────────────────────────────────────────

def _color(val: float, lo_good: float, hi_warn: float, invert: bool = False) -> str:
    if not invert:
        if val >= lo_good:  return "green"
        if val >= hi_warn:  return "yellow"
        return "red"
    else:
        if val <= lo_good:  return "green"
        if val <= hi_warn:  return "yellow"
        return "red"


def _cell_style(col_id: str, value) -> str:
    if value is None:
        return "dim"

    if col_id == "status":
        return {
            "ACTIVE":      "bold green",
            "COMPLETED":   "bold cyan",
            "TERMINATED":  "bold red",
            "NOT_STARTED": "dim",
        }.get(str(value).upper(), "bold yellow")

    if col_id == "spi":
        return f"bold {_color(value, 0.95, 0.80)}"
    if col_id == "cpi":
        return f"bold {_color(value, 0.95, 0.80)}"
    if col_id == "tcpi":
        return f"bold {_color(value, 1.05, 1.10, invert=True)}"
    if col_id == "eac_bac":
        return f"bold {_color(value, 1.05, 1.20, invert=True)}"
    if col_id == "schedule_slip":
        return f"bold {_color(value, 0.0, 2.0, invert=True)}"
    if col_id == "forecast_finish":
        return "dim"
    if col_id == "plan_deviation":
        return "bold red" if value > 0.10 else ("bold yellow" if value > 0 else "bold green")

    if col_id in ("inflow_period", "inflow_cum"):
        return "green" if (value or 0) > 0 else "dim"
    if col_id in ("outflow_period", "outflow_cum"):
        return "red" if (value or 0) > 0 else "dim"
    if col_id in ("net_cf_period", "net_cf_cum"):
        return "bold green" if (value or 0) >= 0 else "bold red"
    if col_id in ("cash_deficit_period", "cash_deficit_cum"):
        return "bold red" if (value or 0) > 0 else "dim"
    if col_id in ("allocation", "alloc_cum"):
        return "cyan" if (value or 0) > 0 else "dim"
    if col_id in ("interest_cost", "interest_cum"):
        return "yellow" if (value or 0) > 0 else "dim"
    if col_id in ("budget_draw", "budget_draw_cum"):
        return "magenta" if (value or 0) > 0 else "dim"

    return ""


def _make_cell(col_id: str, value) -> Text:
    fmt_fn = COLUMN_FMTS[col_id]
    text   = fmt_fn(value)
    style  = _cell_style(col_id, value)
    return Text(text, style=style)


# ─────────────────────────────────────────────────────────────
# RICH TABLE BUILDERS
# ─────────────────────────────────────────────────────────────

def _build_static_params_table(proj: dict) -> Table:
    """Return a Rich Table with per-project static parameters."""
    tbl = Table(
        title="Project Parameters",
        box=rich_box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
        title_style="bold white",
        expand=False,
    )
    tbl.add_column("Parameter",  style="dim",       no_wrap=True)
    tbl.add_column("Value",      style="bold white", no_wrap=True)

    rows = [
        ("Budget (BAC)",           f"{proj['budget']:,.2f}"),
        ("Contract Price",         f"{proj['price']:,.2f}"),
        ("Margin",                 f"{proj['margin']*100:.1f}%"),
        ("Start",                  str(proj["start"])),
        ("Finish (planned)",       str(proj["finish"])),
        ("Duration",               str(proj["duration"])),
        ("S-curve α",              f"{proj['scurve_a']:.3f}"),
        ("S-curve β",              f"{proj['scurve_b']:.3f}"),
        ("Advance %",              f"{proj['advance_percent']*100:.1f}%"),
        ("Advance Trigger",        f"{proj['advance_trigger']*100:.1f}%"),
        ("Advance Recovery Rate",  f"{proj['advance_recovery']*100:.1f}%"),
        ("Retention Rate",         f"{proj['retention_rate']*100:.1f}%"),
        ("Schedule Cap (periods)", str(proj["schedule_cap"])),
        ("Cost Cap (×BAC)",        f"{proj['cost_cap']:.2f}×"),
        ("Cure Length",            str(proj["cure_length"])),
    ]
    for param, val in rows:
        tbl.add_row(param, val)
    return tbl


def _build_milestone_table(
    proj: dict,
    milestones: list,
    proj_state: dict,
) -> Table:
    tbl = Table(
        title="Payment Profile",
        box=rich_box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold yellow",
        title_style="bold white",
        expand=True,
    )

    tbl.add_column("Milestone",      style="bold white", no_wrap=True)
    tbl.add_column("Threshold",      style="cyan",       no_wrap=True, justify="right")
    tbl.add_column("Weight",         style="white",      no_wrap=True, justify="right")
    tbl.add_column("Gross",          style="white",      no_wrap=True, justify="right")
    tbl.add_column("Adv. Recovery",  style="yellow",     no_wrap=True, justify="right")
    tbl.add_column("Cum. Recovered", style="yellow",     no_wrap=True, justify="right")
    tbl.add_column("Retention",      style="yellow",     no_wrap=True, justify="right")
    tbl.add_column("Net Payment",    style="bold green", no_wrap=True, justify="right")
    tbl.add_column("Earliest t",     style="cyan",       no_wrap=True, justify="center")
    tbl.add_column("Status",         style="white",      no_wrap=True, justify="center")

    price            = proj["price"]
    advance_pct      = proj["advance_percent"]
    adv_recovery_rt  = proj["advance_recovery"]
    retention_rt     = proj["retention_rate"]
    advance_received = proj_state.get("advance_received", 0.0)

    # ── Advance Payment row ───────────────────────────────────
    advance_gross = advance_pct * price
    adv_status    = (
        "[bold green]PAID[/]"
        if proj_state.get("advance_received", 0) > 0
        else "[dim]PENDING[/]"
    )
    tbl.add_row(
        "Advance",
        "0%",
        f"{advance_pct*100:.1f}%",
        f"{advance_gross:,.2f}",
        "—", "—", "—",
        f"{advance_gross:,.2f}",
        f"t={proj['start']}",
        adv_status,
    )

    # ── Interim milestone rows ────────────────────────────────
    cum_recovered = 0.0
    for ms in milestones:
        is_final         = ms["threshold"] >= 1.0
        gross            = ms["payment_weight"] * price
        desired_recovery = gross * adv_recovery_rt
        remaining_cap    = max(0.0, advance_received - cum_recovered)
        recovery         = min(desired_recovery, remaining_cap)
        cum_recovered   += recovery
        retention        = 0.0 if is_final else gross * retention_rt
        net              = gross - recovery - retention

        if ms["certified"]:
            cert_t   = ms.get("certified_t")
            ms_style = f"[bold cyan]CERTIFIED t={cert_t}[/]"
        else:
            progress = proj_state.get("progress", 0.0)
            if progress >= ms["threshold"]:
                ms_style = "[bold yellow]ELIGIBLE[/]"
            else:
                ms_style = "[dim]PENDING[/]"

        label = "Final MS / Completion" if is_final else f"MS {ms['j']+1}"

        tbl.add_row(
            label,
            f"{ms['threshold']*100:.0f}%",
            f"{ms['payment_weight']*100:.1f}%",
            f"{gross:,.2f}",
            f"{recovery:,.2f}",
            f"{cum_recovered:,.2f}",
            f"{retention:,.2f}",
            f"{net:,.2f}",
            f"t={ms['earliest_t']}",
            ms_style,
        )

    # ── Retention Release row ─────────────────────────────────
    expected_retention = sum(
        ms["payment_weight"] * price * retention_rt
        for ms in milestones
        if ms["threshold"] < 1.0
    )
    ret_held     = proj_state.get("retention_held", 0.0)
    ret_released = proj_state.get("retention_released", False)
    display_ret  = ret_held if ret_held > 0 else expected_retention
    ret_status   = "[bold green]RELEASED[/]" if ret_released else (
        f"[yellow]HELD {ret_held:,.2f}[/]" if ret_held > 0 else "[dim]PENDING[/]"
    )
    tbl.add_row(
        "Retention Release",
        "100%", "—",
        f"{display_ret:,.2f}",
        "—", "—", "—",
        f"{display_ret:,.2f}",
        f"t={proj['finish']}",
        ret_status,
    )

    return tbl


# ─────────────────────────────────────────────────────────────
# GROUP HEADER ROW
# ─────────────────────────────────────────────────────────────

def _group_header_cells() -> list[Text]:
    grp_col: dict[str, list] = {}
    for col_id in COLUMN_IDS:
        g = COLUMN_GRP[col_id]
        grp_col.setdefault(g, []).append(col_id)

    first_of_group = {cols[0] for cols in grp_col.values()}

    cells = []
    for col_id in COLUMN_IDS:
        g = COLUMN_GRP[col_id]
        if col_id in first_of_group:
            label = _GRP_LABELS.get(g, g)
            cells.append(Text(label, style=_GRP_COLORS.get(g, "bold white")))
        else:
            cells.append(Text("", style="dim"))
    return cells


# ─────────────────────────────────────────────────────────────
# SELECTOR SCREEN
# ─────────────────────────────────────────────────────────────

class SelectorScreen(Screen):
    def __init__(self, conn: sqlite3.Connection):
        super().__init__()
        self.conn      = conn
        self.seed_fn   = None
        self.config_id: Optional[str] = None
        self.method: str = "manual"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Container(id="selector-container"):
            with Vertical(id="selector-box"):
                yield Label("Select configuration", classes="panel-title")
                for key, (_, _, desc) in CONFIG_REGISTRY.items():
                    yield Button(f"[{key}]  {desc}", id=f"cfg-{key}")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id.startswith("cfg-"):
            key = btn_id.split("-")[1]
            module_name, func_name, _ = CONFIG_REGISTRY[key]
            mod = importlib.import_module(module_name)
            self.seed_fn   = getattr(mod, func_name)
            self.config_id = self.seed_fn(self.conn)
            self._show_method_selector()
        elif btn_id == "method-manual":
            self.method = "manual"
            self._launch()

    def _show_method_selector(self) -> None:
        box = self.query_one("#selector-box", Vertical)
        box.remove_children()
        box.mount(Label("Select method", classes="panel-title"))
        box.mount(Button("Manual allocation", id="method-manual"))

    def _launch(self) -> None:
        self.app.switch_screen(
            MainScreen(self.conn, self.config_id, self.method)
        )


# ─────────────────────────────────────────────────────────────
# PORTFOLIO TAB  (Tab 0)
# ─────────────────────────────────────────────────────────────

class PortfolioTab(TabPane):
    def __init__(self, n_projects: int):
        super().__init__("Portfolio", id="tab-portfolio")
        self.n_projects = n_projects

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="stats-panel"):
                yield Static("", id="stat-period")
                yield Static("", id="stat-budget")
                yield Static("", id="stat-horizon")
                yield Static("", id="stat-cumreward")

            yield RichLog(id="cashflow-log", highlight=True, markup=False)

            with Vertical(id="alloc-panel"):
                yield Label("Allocations", classes="panel-title")
                yield Static("", id="alloc-hint")
                for i in range(self.n_projects):
                    yield Input(
                        placeholder=f"Project {i} allocation",
                        id=f"alloc-input-{i}",
                        type="number",
                    )
                yield Static("", id="alloc-error", classes="bad")
                yield Button("Submit", id="submit-btn", variant="primary")

    def update_stats(self, state: dict, total_reward: float) -> None:
        t      = state["t_episode"]
        budget = state["budget"]
        horiz  = state["horizon"]

        self.query_one("#stat-period",    Static).update(
            Text.assemble(("Period  ", "dim"), (str(t), "bold green")))
        self.query_one("#stat-budget",    Static).update(
            Text.assemble(("Budget  ", "dim"), (f"{budget:,.2f}", "bold green")))
        self.query_one("#stat-horizon",   Static).update(
            Text.assemble(("Horizon  ", "dim"), (str(horiz), "bold")))
        self.query_one("#stat-cumreward", Static).update(
            Text.assemble(
                ("Cumulative reward  ", "dim"),
                (f"{total_reward:,.4f}",
                 "bold green" if total_reward >= 0 else "bold red"),
            ))

        active = [p for p in state["projects"] if p["status"] == "active"]
        hint   = "  ".join(f"P{p['i']} (BAC {p['budget']:,.0f})" for p in active)
        self.query_one("#alloc-hint", Static).update(
            Text.assemble(("Active: ", "dim"), (hint, "bold cyan"))
        )
        for i, p in enumerate(state["projects"]):
            try:
                inp = self.query_one(f"#alloc-input-{i}", Input)
                inp.disabled = (p["status"] != "active")
                if p["status"] != "active":
                    inp.value = "0"
            except NoMatches:
                pass

    def log_cashflow(self, t: int, reward: float, info: dict, budget_after: float) -> None:
        log = self.query_one("#cashflow-log", RichLog)
        log.write(Text(f"─── Period {t} complete ───────────────────", style="bold pink1"))
        for cf in info.get("cashflow", []):
            i     = cf["i"]
            alloc = cf["allocation"]
            adv   = cf["advance"]
            ms    = cf["milestone_net"]
            ret   = cf["retention_release"]
            sett  = cf["settlement"]
            if not (alloc > 0 or adv > 0 or ms != 0 or ret > 0 or sett != 0):
                continue
            log.write(Text(f"  Project {i}", style="bold cyan"))
            if alloc > 0:
                log.write(Text(f"    ↳ Allocation      -{alloc:,.2f}", style="red"))
            if adv > 0:
                log.write(Text(f"    ↳ Advance         +{adv:,.2f}", style="green"))
            if ms > 0:
                log.write(Text(f"    ↳ Milestone net   +{ms:,.2f}", style="green"))
            if ret > 0:
                log.write(Text(f"    ↳ Retention rel.  +{ret:,.2f}", style="cyan"))
            if sett and sett < 0:
                log.write(Text(f"    ↳ Settlement      {sett:,.2f}", style="red"))
        rwd_style = "bold green" if reward >= 0 else "bold red"
        log.write(Text(
            f"  Reward: {reward:+.4f}    Budget after: {budget_after:,.2f}",
            style=rwd_style,
        ))
        log.write("")


# ─────────────────────────────────────────────────────────────
# PROJECT TAB  (Tab 1+)
#
# Layout:
#   ┌──────────────────────┬──────────────────────────────────┐
#   │  Project Identity    │  Payment Profile (top)           │
#   │  (static params)     │  Boundary & Termination (bottom) │
#   └──────────────────────┴──────────────────────────────────┘
#   ┌──────────────────────────────────────────────────────────┐
#   │  Group header band                                       │
#   │  DataTable  (rows = periods)                             │
#   └──────────────────────────────────────────────────────────┘
# ─────────────────────────────────────────────────────────────

class ProjectTab(TabPane):
    def __init__(self, proj_index: int):
        super().__init__(f"Project {proj_index}", id=f"tab-proj-{proj_index}")
        self.proj_index = proj_index
        self._row_count = 0
        self._cum: dict = {
            "inflow": 0.0, "outflow": 0.0, "net_cf": 0.0,
            "deficit": 0.0, "alloc": 0.0, "interest": 0.0,
            "budget_draw": 0.0,
        }

    def compose(self) -> ComposeResult:
        i = self.proj_index
        with ScrollableContainer(classes="proj-tab-container"):
            # ── top row: identity (left)  +  right column ────
            with Horizontal(classes="proj-top-row"):
                # left: project identity / static params
                with Vertical(classes="proj-static-panel"):
                    yield Label("Project Identity", classes="section-title")
                    yield Static("", id=f"proj-static-{i}", classes="rich-static")

                # right: payment profile (top) + termination status (bottom)
                with Vertical(classes="proj-right-col"):
                    with Vertical(classes="proj-milestone-panel"):
                        yield Label("Payment Profile", classes="section-title")
                        yield Static("", id=f"proj-milestone-{i}", classes="rich-static")
                    with Vertical(classes="proj-term-panel"):
                        yield Label("Boundary & Termination Status", classes="section-title")
                        yield Static("", id=f"proj-term-{i}")

            # ── group header band ─────────────────────────────
            yield Static("", id=f"proj-grp-hdr-{i}", classes="group-header")

            # ── timestep DataTable ────────────────────────────
            with Container(classes="proj-datatable-container"):
                yield DataTable(
                    id=f"proj-table-{i}",
                    zebra_stripes=True,
                    cursor_type="row",
                )

    # ── render static params ──────────────────────────────────
    def render_static_params(self, proj: dict, proj_state: dict) -> None:
        i   = self.proj_index
        tbl = _build_static_params_table(proj)
        try:
            self.query_one(f"#proj-static-{i}", Static).update(tbl)
        except NoMatches:
            pass

    # ── render payment profile ────────────────────────────────
    def render_milestone_table(
        self,
        proj: dict,
        milestones: list,
        proj_state: dict,
    ) -> None:
        i   = self.proj_index
        tbl = _build_milestone_table(proj, milestones, proj_state)
        try:
            self.query_one(f"#proj-milestone-{i}", Static).update(tbl)
        except NoMatches:
            pass

    # ── render termination / boundary status box ─────────────
    def render_termination_box(self, term_values: dict) -> None:
        i   = self.proj_index
        tbl = Table(
            box=rich_box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        tbl.add_column("Condition", style="dim",       no_wrap=True)
        tbl.add_column("Status",    style="bold white", no_wrap=True, justify="center")

        bool_fields = [
            ("breach_idle",      "Idle Breach"),
            ("breach_deviation", "Deviation Breach"),
            ("breach_schedule",  "Schedule Breach"),
            ("breach_cost",      "Cost Breach"),
            ("breach_deadline",  "Deadline Breach"),
            ("any_breach",       "⚠ Any Breach"),
        ]
        for key, label in bool_fields:
            val    = term_values.get(key, False)
            status = Text("✘  BREACH", style="bold red") if val else Text("✔  OK", style="bold green")
            tbl.add_row(label, status)

        tbl.add_section()
        cure       = term_values.get("cure_remaining")
        cure_text  = _fmt_int(cure)
        cure_style = (
            "bold red"    if (cure is not None and cure <= 1) else
            "bold yellow" if (cure is not None and cure <= 2) else
            "bold green"
        )
        tbl.add_row("Cure Periods Remaining", Text(cure_text, style=cure_style))

        try:
            self.query_one(f"#proj-term-{i}", Static).update(tbl)
        except NoMatches:
            pass

    # ── render group header band ──────────────────────────────
    def render_group_header(self) -> None:
        i        = self.proj_index
        grp_cols: dict[str, list] = {}
        for col_id in COLUMN_IDS:
            g = COLUMN_GRP[col_id]
            grp_cols.setdefault(g, []).append(col_id)

        parts = []
        for g in _GRP_ORDER:
            if g not in grp_cols:
                continue
            label = _GRP_LABELS.get(g, g)
            color = _GRP_COLORS.get(g, "bold white")
            parts.append(Text(f"  ◆ {label}  ", style=color))

        try:
            self.query_one(f"#proj-grp-hdr-{i}", Static).update(Text.assemble(*parts))
        except NoMatches:
            pass

    # ── ensure DataTable columns exist (called once) ──────────
    def _ensure_columns(self) -> None:
        i     = self.proj_index
        table = self.query_one(f"#proj-table-{i}", DataTable)
        if table.ordered_columns:
            return
        for col_id, hdr, width, _, grp in PROJECT_COLUMNS:
            color = _GRP_COLORS.get(grp, "white")
            table.add_column(Text(hdr, style=color), key=col_id, width=width)
        table.add_row(*_group_header_cells(), key="__grp__")

    # ── append one period row ─────────────────────────────────
    def append_row(
        self,
        t: int,
        proj_state: dict,
        state_snapshot: dict,
        proj_params: dict,
        cfg: dict,
        cashflow: dict,
        milestones: list,
        period_reward: float,
    ) -> None:
        i     = self.proj_index
        table = self.query_one(f"#proj-table-{i}", DataTable)
        self._ensure_columns()

        ps = proj_state

        eac     = ps.get("eac", proj_params["budget"])
        eac_bac = eac / proj_params["budget"] if proj_params["budget"] > 0 else 1.0
        alloc   = cashflow.get("allocation",   0.0)
        interest= cashflow.get("interest_cost", 0.0)
        draw    = cashflow.get("treasury_draw", 0.0)

        adv_in   = cashflow.get("advance",           0.0)
        ms_in    = cashflow.get("milestone_net",      0.0)
        ret_in   = cashflow.get("retention_release",  0.0)
        sett_in  = cashflow.get("settlement",         0.0)
        inflow_p  = adv_in + ms_in + ret_in + max(0.0, sett_in)
        outflow_p = alloc + interest
        net_cf_p  = inflow_p - outflow_p
        deficit_p = max(0.0, -net_cf_p)

        self._cum["inflow"]      += inflow_p
        self._cum["outflow"]     += outflow_p
        self._cum["net_cf"]      += net_cf_p
        self._cum["deficit"]     += deficit_p
        self._cum["alloc"]       += alloc
        self._cum["interest"]    += interest
        self._cum["budget_draw"] += draw

        dev     = ps.get("plan_deviation", 0.0)
        slip    = ps.get("schedule_slip",  0.0)
        pdt     = cfg.get("plan_deviation_threshold", 0.10)
        b_idle  = alloc < 1e-9 and ps.get("status") == "active"
        b_dev   = dev  > pdt
        b_sched = slip > proj_params["schedule_cap"]
        b_cost  = eac_bac > proj_params["cost_cap"]
        b_dl    = (t >= proj_params["finish"] + proj_params["schedule_cap"]
                   and ps.get("status") == "active")
        b_any   = b_idle or b_dev or (b_sched and b_cost) or b_dl

        row: dict = {
            "t":                   t,
            "status":              ps.get("status") or "not_started",
            "inflow_period":       inflow_p,
            "outflow_period":      outflow_p,
            "net_cf_period":       net_cf_p,
            "cash_deficit_period": deficit_p,
            "allocation":          alloc,
            "interest_cost":       interest,
            "budget_draw":         draw,
            "inflow_cum":          self._cum["inflow"],
            "outflow_cum":         self._cum["outflow"],
            "net_cf_cum":          self._cum["net_cf"],
            "cash_deficit_cum":    self._cum["deficit"],
            "alloc_cum":           self._cum["alloc"],
            "interest_cum":        self._cum["interest"],
            "budget_draw_cum":     self._cum["budget_draw"],
            "progress":            ps.get("progress",       0.0),
            "progress_plan":       ps.get("progress_plan",  0.0),
            "plan_deviation":      ps.get("plan_deviation", 0.0),
            "spi":                 ps.get("spi",            1.0),
            "cpi":                 ps.get("cpi",            1.0),
            "tcpi":                ps.get("tcpi",           1.0),
            "eac":                 eac,
            "eac_bac":             eac_bac,
            "schedule_slip":       ps.get("schedule_slip",  0.0),
            "forecast_finish":     ps.get("forecast_finish", float(proj_params["finish"])),
            # termination — used for the status box only, not the DataTable
            "cure_remaining":      ps.get("cure_remaining", proj_params.get("cure_length", 0)),
            "breach_idle":         b_idle,
            "breach_deviation":    b_dev,
            "breach_schedule":     b_sched,
            "breach_cost":         b_cost,
            "breach_deadline":     b_dl,
            "any_breach":          b_any,
        }

        cells = [_make_cell(col_id, row[col_id]) for col_id in COLUMN_IDS]
        table.add_row(*cells, key=f"t{t}")
        self._row_count += 1
        table.move_cursor(row=self._row_count, animate=False)

        self.render_milestone_table(proj_params, milestones, ps)
        self.render_termination_box({
            "cure_remaining":   row["cure_remaining"],
            "breach_idle":      row["breach_idle"],
            "breach_deviation": row["breach_deviation"],
            "breach_schedule":  row["breach_schedule"],
            "breach_cost":      row["breach_cost"],
            "breach_deadline":  row["breach_deadline"],
            "any_breach":       row["any_breach"],
        })

    # ── initial snapshot row at t=0 ──────────────────────────
    def append_initial_row(
        self,
        proj_state: dict,
        state_snapshot: dict,
        proj_params: dict,
        cfg: dict,
        milestones: list,
    ) -> None:
        self.render_static_params(proj_params, proj_state)
        self.render_milestone_table(proj_params, milestones, proj_state)
        self.render_group_header()
        self.render_termination_box({
            "cure_remaining":   proj_params.get("cure_length", 0),
            "breach_idle":      False,
            "breach_deviation": False,
            "breach_schedule":  False,
            "breach_cost":      False,
            "breach_deadline":  False,
            "any_breach":       False,
        })
        self.append_row(
            t=0,
            proj_state=proj_state,
            state_snapshot=state_snapshot,
            proj_params=proj_params,
            cfg=cfg,
            cashflow={},
            milestones=milestones,
            period_reward=0.0,
        )


# ─────────────────────────────────────────────────────────────
# MAIN SCREEN
# ─────────────────────────────────────────────────────────────

class MainScreen(Screen):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+p", "command_palette", "Command Palette"),
    ]

    def __init__(self, conn: sqlite3.Connection, config_id: str, method: str):
        super().__init__()
        self.conn         = conn
        self.config_id    = config_id
        self.method       = method
        self.env: Optional[PortfolioEnv] = None
        self.state: dict  = {}
        self.total_reward = 0.0
        self.done         = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Initialising…", id="top-bar")
        with TabbedContent(id="main-tabs"):
            pass
        yield Footer()

    def on_mount(self) -> None:
        self.env   = PortfolioEnv(self.conn, self.config_id, method=self.method)
        self.state = self.env.reset()
        self._rebuild_tabs()
        # Defer population until after Textual finishes mounting all panes
        self.call_after_refresh(self._after_mount)

    def _after_mount(self) -> None:
        self._refresh_portfolio()
        self._append_initial_rows()

    # ── rebuild tabs ──────────────────────────────────────────
    def _rebuild_tabs(self) -> None:
        tabs = self.query_one("#main-tabs", TabbedContent)
        tabs.clear_panes()
        n = len(self.env.projects)
        tabs.add_pane(PortfolioTab(n))
        for i in range(n):
            tabs.add_pane(ProjectTab(i))

    # ── write t=0 snapshot into every project tab ─────────────
    def _append_initial_rows(self) -> None:
        for i, (proj, ps) in enumerate(zip(self.env.projects, self.env.proj_state)):
            snap = self.state["projects"][i]
            try:
                tab = self.query_one(f"ProjectTab#tab-proj-{i}")
                tab.append_initial_row(
                    proj_state=dict(ps),
                    state_snapshot=snap,
                    proj_params=proj,
                    cfg=dict(self.env.cfg),
                    milestones=self.env.milestones[i],
                )
            except NoMatches:
                pass

    # ── refresh top bar and portfolio tab ─────────────────────
    def _refresh_portfolio(self) -> None:
        state  = self.state
        budget = state["budget"]
        t      = state["t_episode"]

        self.query_one("#top-bar", Static).update(
            Text.assemble(
                ("Portfolio Budgeting  ", "bold"),
                ("Period ", "dim"), (str(t), "bold green"),
                ("  │  Budget ", "dim"), (f"{budget:,.2f}", "bold green"),
                ("  │  Reward ", "dim"),
                (f"{self.total_reward:,.4f}",
                 "bold green" if self.total_reward >= 0 else "bold red"),
            )
        )
        try:
            port = self.query_one(PortfolioTab)
            port.update_stats(state, self.total_reward)
        except NoMatches:
            pass

    # ── handle submit ─────────────────────────────────────────
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "submit-btn" or self.done:
            return

        n        = len(self.env.projects)
        allocs   = []
        error_wg = self.query_one("#alloc-error", Static)

        for i in range(n):
            try:
                inp = self.query_one(f"#alloc-input-{i}", Input)
                val = float(inp.value or "0")
                if val < 0:
                    error_wg.update("All values must be ≥ 0.")
                    return
                allocs.append(val)
            except (NoMatches, ValueError):
                allocs.append(0.0)

        budget = self.state["budget"]
        if sum(allocs) > budget + 1e-6:
            error_wg.update(
                f"Total {sum(allocs):,.2f} exceeds budget {budget:,.2f}."
            )
            return
        error_wg.update("")

        t_executed = self.state["t_episode"]

        new_state, reward, self.done, info = self.env.step(allocs)
        self.total_reward += reward

        cf_by_proj = {cf["i"]: cf for cf in info.get("cashflow", [])}

        for i, (proj, ps) in enumerate(zip(self.env.projects, self.env.proj_state)):
            snap = new_state["projects"][i]
            try:
                tab = self.query_one(f"ProjectTab#tab-proj-{i}")
                tab.append_row(
                    t=t_executed + 1,
                    proj_state=dict(ps),
                    state_snapshot=snap,
                    proj_params=proj,
                    cfg=dict(self.env.cfg),
                    cashflow=cf_by_proj.get(i, {}),
                    milestones=self.env.milestones[i],
                    period_reward=reward / n,
                )
            except NoMatches:
                pass

        try:
            port = self.query_one(PortfolioTab)
            port.log_cashflow(t_executed, reward, info, new_state["budget"])
        except NoMatches:
            pass

        self.state = new_state
        self._refresh_portfolio()

        if self.done:
            self._show_done()

    # ── episode-end summary ───────────────────────────────────
    def _show_done(self) -> None:
        try:
            port = self.query_one(PortfolioTab)
            log  = port.query_one("#cashflow-log", RichLog)
            log.write(Text("═══ EPISODE COMPLETE ═══", style="bold green"))
            log.write(Text(
                f"Total reward: {self.total_reward:,.4f}",
                style="bold green" if self.total_reward >= 0 else "bold red",
            ))
            for proj, ps in zip(self.env.projects, self.env.proj_state):
                status = ps.get("status") or "not_started"
                col    = "green" if status == "completed" else "red"
                log.write(Text(
                    f"  Project {proj['i']}  {status.upper()}"
                    f"  progress {ps['progress']:.4f}"
                    f"  CPI {ps['cpi']:.4f}  SPI {ps['spi']:.4f}",
                    style=col,
                ))
            log.write(Text("Press Q to quit.", style="dim"))
        except NoMatches:
            pass

    def action_quit(self) -> None:
        self.conn.close()
        self.app.exit()


# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────

class PortfolioApp(App):
    CSS                    = CSS
    TITLE                  = "Portfolio Budget Allocator"
    BINDINGS               = [Binding("q", "quit", "Quit")]
    DEFAULT_CSS            = ""
    ENABLE_COMMAND_PALETTE = True

    def on_ready(self) -> None:
        self.theme = "ansi-light"

    def __init__(self):
        super().__init__()
        self.conn = init_db()

    def on_mount(self) -> None:
        self.push_screen(SelectorScreen(self.conn))

    def action_quit(self) -> None:
        self.conn.close()
        self.exit()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

app = PortfolioApp()

if __name__ == "__main__":
    app.run()