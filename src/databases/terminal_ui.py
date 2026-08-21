# terminal_ui.py
#
# Textual TUI for the Portfolio Budgeting Environment.
# Run with:  python terminal_ui.py
#
# Tab 0  — Portfolio  : period, budget, reward, cash-flow log, allocation input
# Tab 1+ — Project i  : scrollable DataTable (rows = periods, columns = all signals)

from __future__ import annotations

import importlib
import sqlite3
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button, DataTable, Footer, Header, Input,
    Label, RichLog, Static, TabbedContent, TabPane,
)
from rich.text import Text

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
# COLUMN DEFINITIONS
# Each entry: (key_in_state, label, width, format_fn)
# ─────────────────────────────────────────────────────────────

def _fmt_f4(v) -> str:
    return f"{v:.4f}" if v is not None else "—"

def _fmt_f2(v) -> str:
    return f"{v:.2f}" if v is not None else "—"

def _fmt_pct(v) -> str:
    return f"{v*100:.2f}%" if v is not None else "—"

def _fmt_int(v) -> str:
    return str(int(v)) if v is not None else "—"

def _fmt_mu(v) -> str:
    return f"{v:,.2f}" if v is not None else "—"

def _fmt_bool_breach(v) -> str:
    return "✘" if v else "✔"

def _fmt_status(v) -> str:
    return (v or "—").upper()

# (column_id, header_label, width, format_fn)
PROJECT_COLUMNS: list[tuple[str, str, int, callable]] = [
    # ── identity ──────────────────────────────────────────────
    ("t",                       "t",             4,  _fmt_int),
    ("status",                  "Status",        12, _fmt_status),
    # ── progress ──────────────────────────────────────────────
    ("progress",                "Progress",      10, _fmt_f4),
    ("progress_plan",           "Plan",          10, _fmt_f4),
    ("plan_deviation",          "Deviation",     10, _fmt_f4),
    # ── EVM ───────────────────────────────────────────────────
    ("spi",                     "SPI(t)",        9,  _fmt_f4),
    ("cpi",                     "CPI",           9,  _fmt_f4),
    ("tcpi",                    "TCPI",          9,  _fmt_f4),
    ("eac",                     "EAC",           12, _fmt_mu),
    ("eac_bac",                 "EAC/BAC",       10, _fmt_f4),
    ("schedule_slip",           "SchedSlip",     10, _fmt_f2),
    ("cure_remaining",          "CureLeft",      9,  _fmt_int),
    # ── breach flags ──────────────────────────────────────────
    ("breach_deviation",        "B:Dev",         7,  _fmt_bool_breach),
    ("breach_schedule",         "B:Sched",       9,  _fmt_bool_breach),
    ("breach_cost",             "B:Cost",        8,  _fmt_bool_breach),
    ("breach_both",             "B:Both",        8,  _fmt_bool_breach),
    # ── cash-flow (for the period just executed) ─────────────
    ("allocation",              "Alloc",         10, _fmt_mu),
    ("advance",                 "Advance",       10, _fmt_mu),
    ("milestone_net",           "MS Net",        10, _fmt_mu),
    ("retention_release",       "Ret.Rel.",      10, _fmt_mu),
    ("settlement",              "Settle",        10, _fmt_mu),
    # ── next milestone ────────────────────────────────────────
    ("next_ms_threshold_gap",   "NextGap",       10, _fmt_f4),
    ("next_ms_net_payment",     "NextPay",       10, _fmt_mu),
    ("next_ms_earliest_t",      "NextEarliest",  13, _fmt_int),
    ("next_ms_is_final",        "IsFinal",       9,
     lambda v: "★ yes" if v else "no"),
    # ── period reward ─────────────────────────────────────────
    ("period_reward",           "Reward",        10, _fmt_f4),
]

COLUMN_IDS  = [c[0] for c in PROJECT_COLUMNS]
COLUMN_HDRS = [c[1] for c in PROJECT_COLUMNS]
COLUMN_FMTS = {c[0]: c[3] for c in PROJECT_COLUMNS}


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
    padding: 1 2;
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

/* ── project tab — full-height DataTable ── */
.proj-table-pane {
    height: 1fr;
    padding: 0;
}

DataTable {
    height: 1fr;
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
# HELPERS
# ─────────────────────────────────────────────────────────────

def _color(val: float, lo_good: float, hi_warn: float, invert: bool = False) -> str:
    if not invert:
        if val >= lo_good:   return "green"
        if val >= hi_warn:   return "yellow"
        return "red"
    else:
        if val <= lo_good:   return "green"
        if val <= hi_warn:   return "yellow"
        return "red"


def _cell_style(col_id: str, value) -> str:
    """Return a Rich style string for a data cell based on column semantics."""
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
    if col_id == "cure_remaining":
        return f"bold {_color(value, 2.0, 1.0)}"
    if col_id == "plan_deviation":
        return "bold red" if value > 0.10 else ("bold yellow" if value > 0 else "bold green")

    if col_id in ("breach_deviation", "breach_schedule", "breach_cost", "breach_both"):
        return "bold red" if value else "bold green"

    if col_id in ("allocation", "advance", "milestone_net", "retention_release"):
        return "green" if (value or 0) > 0 else "dim"
    if col_id == "settlement":
        return "red" if (value or 0) < 0 else "dim"

    if col_id == "next_ms_threshold_gap":
        if value is None:   return "dim"
        if value <= 0.0:    return "bold green"
        if value <= 0.10:   return "bold yellow"
        return "bold red"

    if col_id == "period_reward":
        if value is None:   return "dim"
        return "bold green" if value >= 0 else "bold red"

    return ""   # default


def _make_cell(col_id: str, value) -> Text:
    fmt_fn = COLUMN_FMTS[col_id]
    text   = fmt_fn(value)
    style  = _cell_style(col_id, value)
    return Text(text, style=style)


# ─────────────────────────────────────────────────────────────
# SELECTOR SCREEN
# ─────────────────────────────────────────────────────────────

class SelectorScreen(Screen):
    def __init__(self, conn: sqlite3.Connection):
        super().__init__()
        self.conn = conn
        self.seed_fn = None
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
            self.seed_fn = getattr(mod, func_name)
            self.config_id = self.seed_fn(self.conn)
            self._show_method_selector()

        elif btn_id == "method-manual":
            self.method = "manual"
            self._launch()

    def _show_method_selector(self) -> None:
        box_widget = self.query_one("#selector-box", Vertical)
        box_widget.remove_children()
        box_widget.mount(Label("Select method", classes="panel-title"))
        box_widget.mount(Button("Manual allocation", id="method-manual"))

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

        self.query_one("#stat-period", Static).update(
            Text.assemble(("Period  ", "dim"), (str(t), "bold green")))
        self.query_one("#stat-budget", Static).update(
            Text.assemble(("Budget  ", "dim"), (f"{budget:,.2f}", "bold green")))
        self.query_one("#stat-horizon", Static).update(
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
# PROJECT TAB  (Tab 1+)  — DataTable, rows=periods
# ─────────────────────────────────────────────────────────────

class ProjectTab(TabPane):
    """One tab per project.  Columns = all signals.  Rows = periods (appended)."""

    def __init__(self, proj_index: int):
        super().__init__(f"Project {proj_index}", id=f"tab-proj-{proj_index}")
        self.proj_index = proj_index
        self._row_count = 0

    def compose(self) -> ComposeResult:
        i = self.proj_index
        yield Label(
            f"Project {i} — time-series signals  "
            "| ← → scroll columns  | ↑ ↓ scroll rows",
            classes="panel-title",
        )
        yield DataTable(id=f"proj-table-{i}", zebra_stripes=True, cursor_type="row")

    def on_mount(self) -> None:
        """Add all column headers once."""
        table = self.query_one(f"#proj-table-{self.proj_index}", DataTable)
        for col_id, hdr, width, _ in PROJECT_COLUMNS:
            table.add_column(hdr, key=col_id, width=width)

    # ── called once per period after env.step() ──────────────
    def append_row(
        self,
        t: int,
        proj_state: dict,       # env.proj_state[i]  (internal state object)
        state_snapshot: dict,   # state["projects"][i]  (obs snapshot)
        proj_params: dict,      # env.projects[i]
        cfg: dict,
        cashflow: dict,         # single project's cashflow entry from info
        period_reward: float,
    ) -> None:
        table = self.query_one(f"#proj-table-{self.proj_index}", DataTable)

        ps = proj_state
        snap = state_snapshot

        # ── compute derived breach flags ──────────────────────
        eac     = ps.get("eac", proj_params["budget"])
        eac_bac = eac / proj_params["budget"] if proj_params["budget"] > 0 else 1.0
        dev     = ps.get("plan_deviation", 0.0)
        slip    = ps.get("schedule_slip",  0.0)
        pdt     = cfg.get("plan_deviation_threshold", 0.10)

        b_dev   = dev  > pdt
        b_sched = slip > proj_params["schedule_cap"]
        b_cost  = eac_bac > proj_params["cost_cap"]
        b_both  = b_sched and b_cost

        # ── build row dict ────────────────────────────────────
        row: dict = {
            "t":                    t,
            "status":               ps.get("status") or "not_started",
            "progress":             ps.get("progress",        0.0),
            "progress_plan":        ps.get("progress_plan",   0.0),
            "plan_deviation":       ps.get("plan_deviation",  0.0),
            "spi":                  ps.get("spi",             1.0),
            "cpi":                  ps.get("cpi",             1.0),
            "tcpi":                 ps.get("tcpi",            1.0),
            "eac":                  eac,
            "eac_bac":              eac_bac,
            "schedule_slip":        ps.get("schedule_slip",   0.0),
            "cure_remaining":       ps.get("cure_remaining",  proj_params.get("cure_length", 0)),
            "breach_deviation":     b_dev,
            "breach_schedule":      b_sched,
            "breach_cost":          b_cost,
            "breach_both":          b_both,
            # cashflow — zero-fill if project not in this period's cf
            "allocation":           cashflow.get("allocation",       0.0),
            "advance":              cashflow.get("advance",          0.0),
            "milestone_net":        cashflow.get("milestone_net",    0.0),
            "retention_release":    cashflow.get("retention_release",0.0),
            "settlement":           cashflow.get("settlement",       0.0),
            # next milestone (from snapshot)
            "next_ms_threshold_gap":  snap.get("next_ms_threshold_gap",  0.0),
            "next_ms_net_payment":    snap.get("next_ms_net_payment",    0.0),
            "next_ms_earliest_t":     snap.get("next_ms_earliest_t"),
            "next_ms_is_final":       snap.get("next_ms_is_final",       False),
            "period_reward":          period_reward,
        }

        cells = [_make_cell(col_id, row[col_id]) for col_id in COLUMN_IDS]
        table.add_row(*cells, key=f"t{t}")
        self._row_count += 1
        # scroll to the newest row
        table.move_cursor(row=self._row_count - 1, animate=False)

    # ── initial row for t=0 (before first step) ──────────────
    def append_initial_row(
        self,
        proj_state: dict,
        state_snapshot: dict,
        proj_params: dict,
        cfg: dict,
    ) -> None:
        """Insert the t=0 / pre-action row (no cashflow yet)."""
        self.append_row(
            t=0,
            proj_state=proj_state,
            state_snapshot=state_snapshot,
            proj_params=proj_params,
            cfg=cfg,
            cashflow={},
            period_reward=0.0,
        )


# ─────────────────────────────────────────────────────────────
# MAIN SCREEN
# ─────────────────────────────────────────────────────────────

class MainScreen(Screen):
    BINDINGS = [Binding("q", "quit", "Quit")]

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

    # ── write the t=0 snapshot into every project table ───────
    def _append_initial_rows(self) -> None:
        for i, (proj, ps) in enumerate(zip(self.env.projects, self.env.proj_state)):
            snap = self.state["projects"][i]
            try:
                tab = self.query_one(f"#tab-proj-{i}", ProjectTab)
                tab.append_initial_row(
                    proj_state=dict(ps),
                    state_snapshot=snap,
                    proj_params=proj,
                    cfg=dict(self.env.cfg),
                )
            except NoMatches:
                pass

    # ── refresh portfolio stats bar ───────────────────────────
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
            port = self.query_one("#tab-portfolio", PortfolioTab)
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

        # ── step the environment ──────────────────────────────
        new_state, reward, self.done, info = self.env.step(allocs)
        self.total_reward += reward

        # ── cashflow lookup: index by project i ───────────────
        cf_by_proj = {cf["i"]: cf for cf in info.get("cashflow", [])}

        # ── append one row to each project table ──────────────
        for i, (proj, ps) in enumerate(zip(self.env.projects, self.env.proj_state)):
            snap = new_state["projects"][i]
            try:
                tab = self.query_one(f"#tab-proj-{i}", ProjectTab)
                tab.append_row(
                    t=t_executed + 1,
                    proj_state=dict(ps),
                    state_snapshot=snap,
                    proj_params=proj,
                    cfg=dict(self.env.cfg),
                    cashflow=cf_by_proj.get(i, {}),
                    period_reward=reward / n,   # approximate per-project share
                )
            except NoMatches:
                pass

        # ── update portfolio tab log + stats ──────────────────
        try:
            port = self.query_one("#tab-portfolio", PortfolioTab)
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
            port = self.query_one("#tab-portfolio", PortfolioTab)
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
    ENABLE_COMMAND_PALETTE = False

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

if __name__ == "__main__":
    PortfolioApp().run()