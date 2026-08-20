# terminal_ui.py
#
# Textual TUI for the Portfolio Budgeting Environment.
# Run with:  python terminal_ui.py
#
# Tab 0  — Portfolio  : period, budget, reward, cash-flow log, allocation input
# Tab 1+ — Project i  : EVM signals, breach conditions, milestone history, next payment

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
    Button, Footer, Header, Input,
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

/* ── project tab — 2×2 CSS grid ── */
/*
   Each ProjectTab renders a Container with class proj-grid.
   CSS grid places children in reading order: evm (0,0), ms (0,1),
   breach (1,0), next (1,1). We override order below.
*/
.proj-grid {
    layout: grid;
    grid-size: 2;
    grid-rows: 1fr 1fr;
    grid-gutter: 1;
    height: 1fr;
}

/* panel base — shared */
.evm-panel,
.breach-panel,
.ms-panel,
.next-panel {
    padding: 1;
    background: $panel;
    height: 1fr;
}

/* individual borders */
.evm-panel    { border: solid $primary-darken-2; }
.ms-panel     { border: solid $primary-darken-2; }
.breach-panel { border: solid $warning-darken-1; }
.next-panel   { border: solid $accent; }


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
    """Return a Rich color string based on threshold rules."""
    if not invert:
        if val >= lo_good:
            return "green"
        if val >= hi_warn:
            return "yellow"
        return "red"
    else:
        if val <= lo_good:
            return "green"
        if val <= hi_warn:
            return "yellow"
        return "red"


def _tag(breached: bool) -> Text:
    """Small ✔/✘ status tag."""
    if breached:
        return Text(" ✘ BREACH", style="bold red on red")
    return Text(" ✔ clear", style="bold green on green")


# ─────────────────────────────────────────────────────────────
# SELECTOR SCREEN
# ─────────────────────────────────────────────────────────────

class SelectorScreen(Screen):
    """Config + method selection before the main TUI."""

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
            # ── inline stats bar ──────────────────────────────
            with Horizontal(id="stats-panel"):
                yield Static("", id="stat-period")
                yield Static("", id="stat-budget")
                yield Static("", id="stat-horizon")
                yield Static("", id="stat-cumreward")

            # ── scrollable cash-flow log ──────────────────────
            yield RichLog(id="cashflow-log", highlight=True, markup=False)

            # ── allocation input area ─────────────────────────
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

    # ── update top stats bar ──────────────────────────────────
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

        # hint line: active projects
        active = [p for p in state["projects"] if p["status"] == "active"]
        hint   = "  ".join(f"P{p['i']} (BAC {p['budget']:,.0f})" for p in active)
        self.query_one("#alloc-hint", Static).update(
            Text.assemble(("Active: ", "dim"), (hint, "bold cyan"))
        )

        # enable/disable inputs
        for i, p in enumerate(state["projects"]):
            try:
                inp = self.query_one(f"#alloc-input-{i}", Input)
                inp.disabled = (p["status"] != "active")
                if p["status"] != "active":
                    inp.value = "0"
            except NoMatches:
                pass

    # ── append one period block to the cash-flow log ──────────
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
# ─────────────────────────────────────────────────────────────

class ProjectTab(TabPane):
    def __init__(self, proj_index: int):
        super().__init__(f"Project {proj_index}", id=f"tab-proj-{proj_index}")
        self.proj_index = proj_index

    def compose(self) -> ComposeResult:
        i = self.proj_index
        # proj-grid is a 2-column CSS grid; children placed in DOM order:
        #   slot 0 (col 0, row 0) → EVM
        #   slot 1 (col 1, row 0) → Milestone history
        #   slot 2 (col 0, row 1) → Breach conditions
        #   slot 3 (col 1, row 1) → Next payment
        with Container(classes="proj-grid"):
            with ScrollableContainer(classes="evm-panel"):
                yield Label("EVM SIGNALS", classes="panel-title")
                yield Static("", id=f"evm-content-{i}")

            with ScrollableContainer(classes="ms-panel"):
                yield Label("MILESTONE HISTORY", classes="panel-title")
                yield Static("", id=f"ms-content-{i}")

            with ScrollableContainer(classes="breach-panel"):
                yield Label("BREACH CONDITIONS", classes="panel-title")
                yield Static("", id=f"breach-content-{i}")

            with ScrollableContainer(classes="next-panel"):
                yield Label("NEXT PAYMENT", classes="panel-title")
                yield Static("", id=f"next-content-{i}")

    # ── master update entry point ─────────────────────────────
    def update(self, proj_state: dict, proj_params: dict, cfg: dict) -> None:
        self._update_evm(proj_state, proj_params)
        self._update_breach(proj_state, proj_params, cfg)
        self._update_ms(proj_state)
        self._update_next(proj_state)

    # ── EVM SIGNALS ───────────────────────────────────────────
    def _update_evm(self, ps: dict, proj: dict) -> None:
        i       = self.proj_index
        status  = ps.get("status") or "not_started"
        spi     = ps.get("spi",             1.0)
        cpi     = ps.get("cpi",             1.0)
        tcpi    = ps.get("tcpi",            1.0)
        eac     = ps.get("eac",             proj["budget"])
        slip    = ps.get("schedule_slip",   0.0)
        cure    = ps.get("cure_remaining",  proj["cure_length"])
        prog    = ps.get("progress",        0.0)
        plan    = ps.get("progress_plan",   0.0)
        dev     = ps.get("plan_deviation",  0.0)
        eac_bac = eac / proj["budget"] if proj["budget"] > 0 else 1.0

        status_style = {
            "active":      "bold green",
            "completed":   "bold cyan",
            "terminated":  "bold red",
            "not_started": "dim",
        }.get(status, "bold yellow")

        spi_s   = _color(spi,     0.95, 0.80)
        cpi_s   = _color(cpi,     0.95, 0.80)
        tcpi_s  = _color(tcpi,    1.05, 1.10, invert=True)
        eac_s   = _color(eac_bac, 1.05, 1.20, invert=True)   # reused for EAC/BAC row
        slip_s  = _color(slip,    0.0,  2.0,  invert=True)
        cure_s  = _color(cure,    2.0,  1.0)

        t = Text()

        # Status
        t.append("Status        ", style="dim")
        t.append(f"{status.upper()}\n", style=status_style)

        # Progress
        t.append("Progress      ", style="dim")
        t.append(f"{prog:.4f}  ", style="bold")
        t.append(f"plan {plan:.4f}  dev {dev:+.4f}\n", style="dim")

        # SPI(t)
        t.append("SPI(t)        ", style="dim")
        t.append(f"{spi:.4f}\n", style=f"bold {spi_s}")

        # CPI
        t.append("CPI           ", style="dim")
        t.append(f"{cpi:.4f}\n", style=f"bold {cpi_s}")

        # TCPI
        t.append("TCPI          ", style="dim")
        t.append(f"{tcpi:.4f}\n", style=f"bold {tcpi_s}")

        # EAC  (value + ratio dimmed inline)
        t.append("EAC           ", style="dim")
        t.append(f"{eac:,.2f}  ", style="bold")
        t.append(f"({eac_bac:.4f}×)\n", style="dim")

        # EAC/BAC  — own row, inverted thresholds, cap dimmed
        t.append("EAC/BAC       ", style="dim")
        t.append(f"{eac_bac:.4f}×  ", style=f"bold {eac_s}")
        t.append(f"(cap {proj['cost_cap']}×)\n", style="dim")

        # Schedule slip  — inverted, cap dimmed
        t.append("Schedule slip ", style="dim")
        t.append(f"{slip:+.2f} p  ", style=f"bold {slip_s}")
        t.append(f"(cap {proj['schedule_cap']} p)\n", style="dim")

        # Cure left
        t.append("Cure left     ", style="dim")
        t.append(f"{cure} / {proj['cure_length']}", style=f"bold {cure_s}")

        try:
            self.query_one(f"#evm-content-{i}", Static).update(t)
        except NoMatches:
            pass

    # ── BREACH CONDITIONS ─────────────────────────────────────
    def _update_breach(self, ps: dict, proj: dict, cfg: dict) -> None:
        i         = self.proj_index
        dev       = ps.get("plan_deviation",  0.0)
        slip      = ps.get("schedule_slip",   0.0)
        eac       = ps.get("eac",             proj["budget"])
        eac_bac   = eac / proj["budget"] if proj["budget"] > 0 else 1.0

        pdt        = cfg.get("plan_deviation_threshold", 0.10)
        sched_cap  = proj["schedule_cap"]
        cost_cap   = proj["cost_cap"]

        dev_breach   = dev > pdt
        sched_breach = slip > sched_cap
        cost_breach  = eac_bac > cost_cap
        both_breach  = sched_breach and cost_breach

        # column widths
        C0, C1, C2 = 22, 16, 14

        t = Text()

        # header — 4 columns, all dimmed
        t.append(
            f"{'Condition':<{C0}}{'Current':<{C1}}{'Limit':<{C2}}Status\n",
            style="dim",
        )

        # row 1: Idle (zero alloc) — unknowable per-period
        t.append(f"{'Idle (zero alloc)':<{C0}}", style="dim")
        t.append(f"{'—':<{C1}}", style="dim")
        t.append(f"{'alloc = 0':<{C2}}", style="dim")
        t.append("(see portfolio tab)\n", style="dim")

        # row 2: Deviation
        t.append(f"{'Deviation':<{C0}}", style="dim")
        t.append(f"{dev:<{C1}.4f}")
        t.append(f"{'> ' + str(pdt):<{C2}}", style="dim")
        t.append_text(_tag(dev_breach))
        t.append("\n")

        # row 3: Schedule slip
        t.append(f"{'Schedule slip':<{C0}}", style="dim")
        t.append(f"{slip:<{C1}.2f}")
        t.append(f"{'cap ' + str(sched_cap):<{C2}}", style="dim")
        t.append_text(_tag(sched_breach))
        t.append("\n")

        # row 4: Cost (EAC/BAC)
        t.append(f"{'Cost (EAC/BAC)':<{C0}}", style="dim")
        t.append(f"{eac_bac:<{C1}.4f}")
        t.append(f"{'cap ' + str(cost_cap) + '×':<{C2}}", style="dim")
        t.append_text(_tag(cost_breach))
        t.append("\n")

        # row 5: Sched AND Cost
        t.append(f"{'Sched AND Cost':<{C0}}", style="dim")
        t.append(
            f"{'both' if both_breach else 'not both':<{C1}}",
            style="bold red" if both_breach else "bold green",
        )
        t.append(f"{'need both':<{C2}}", style="dim")
        t.append_text(_tag(both_breach))

        try:
            self.query_one(f"#breach-content-{i}", Static).update(t)
        except NoMatches:
            pass

    # ── MILESTONE HISTORY ─────────────────────────────────────
    def _update_ms(self, ps: dict) -> None:
        i = self.proj_index

        t = Text()
        # header
        t.append(
            f"{'MS':<5}{'Thresh':<10}{'Status':<14}{'Net MU':<12}{'t≥':<6}Certified\n",
            style="dim",
        )

        for ms in ps.get("ms_history", []):
            certified  = ms.get("certified", False)
            is_final   = ms.get("is_final",  False)
            thresh_pct = f"{ms['threshold'] * 100:.0f}%"
            if is_final:
                thresh_pct += " ★"
            status_str   = "CERTIFIED" if certified else "pending"
            status_style = "bold green" if certified else "dim"
            cert_t       = f"t={ms['certified_t']}" if certified else "—"

            t.append(f"{ms['j']:<5}", style="dim")
            t.append(f"{thresh_pct:<10}", style="bold" if is_final else "")
            t.append(f"{status_str:<14}", style=status_style)
            t.append(f"{ms['net']:<12.2f}")
            t.append(f"{ms['earliest_t']:<6}", style="dim")
            t.append(f"{cert_t}\n", style="dim")

        try:
            self.query_one(f"#ms-content-{i}", Static).update(t)
        except NoMatches:
            pass

    # ── NEXT PAYMENT ──────────────────────────────────────────
    def _update_next(self, ps: dict) -> None:
        i        = self.proj_index
        gap      = ps.get("next_ms_threshold_gap",  0.0)
        net_pay  = ps.get("next_ms_net_payment",    0.0)
        earliest = ps.get("next_ms_earliest_t")
        is_fin   = ps.get("next_ms_is_final",       False)
        t_ep     = ps.get("t_episode",              ps.get("t_project", 0) or 0)
        next_t   = (t_ep or 0) + 1
        plan     = ps.get("progress_plan",          0.0)
        dev      = ps.get("plan_deviation",         0.0)

        # threshold gap coloring
        gap_reached = gap <= 0.0
        if gap_reached:
            gap_str   = "reached ✔"
            gap_style = "bold green"
        elif gap <= 0.10:
            gap_str   = f"{gap:.4f}"
            gap_style = "bold yellow"
        else:
            gap_str   = f"{gap:.4f}"
            gap_style = "bold red"

        # period lock
        if earliest is None:
            lock_str, lock_style = "—", "dim"
        elif next_t >= earliest:
            lock_str  = f"t={earliest}  unlocked ✔"
            lock_style = "bold green"
        else:
            diff      = earliest - next_t
            lock_str  = f"t={earliest}  ({diff} period{'s' if diff != 1 else ''} away)"
            lock_style = "bold yellow"

        t = Text()

        if is_fin:
            t.append("★ FINAL MILESTONE\n", style="bold yellow")

        t.append("Threshold gap   ", style="dim")
        t.append(f"{gap_str}\n", style=gap_style)

        t.append("Net payment     ", style="dim")
        t.append(
            f"{net_pay:,.2f} MU\n",
            style="bold green" if net_pay > 0 else "dim",
        )

        t.append("Period lock     ", style="dim")
        t.append(f"{lock_str}\n", style=lock_style)

        t.append("Plan target nxt ", style="dim")
        t.append(f"{plan:.4f}\n", style="bold")

        t.append("Δ vs plan       ", style="dim")
        t.append(f"{dev:+.4f}", style="bold green" if dev <= 0 else "bold red")

        try:
            self.query_one(f"#next-content-{i}", Static).update(t)
        except NoMatches:
            pass


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

    # ── compose ───────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Initialising…", id="top-bar")
        with TabbedContent(id="main-tabs"):
            pass
        yield Footer()

    # ── on mount ──────────────────────────────────────────────
    def on_mount(self) -> None:
        self.env   = PortfolioEnv(self.conn, self.config_id, method=self.method)
        self.state = self.env.reset()
        self._rebuild_tabs()
        self._refresh_all()

    # ── rebuild tabs after env reset ──────────────────────────
    def _rebuild_tabs(self) -> None:
        tabs = self.query_one("#main-tabs", TabbedContent)
        tabs.clear_panes()
        n = len(self.env.projects)
        tabs.add_pane(PortfolioTab(n))
        for i in range(n):
            tabs.add_pane(ProjectTab(i))

    # ── refresh every panel ───────────────────────────────────
    def _refresh_all(self) -> None:
        state = self.state
        env   = self.env
        t     = state["t_episode"]
        budget = state["budget"]

        # top-bar summary
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

        # portfolio tab
        try:
            port = self.query_one("#tab-portfolio", PortfolioTab)
            port.update_stats(state, self.total_reward)
        except NoMatches:
            pass

        # project tabs
        for i, (proj, ps_dict) in enumerate(zip(env.projects, env.proj_state)):
            combined = dict(ps_dict)
            proj_state_snapshot = state["projects"][i]
            combined["ms_history"]             = proj_state_snapshot.get("ms_history", [])
            combined["next_ms_threshold_gap"]  = proj_state_snapshot.get("next_ms_threshold_gap", 0.0)
            combined["next_ms_net_payment"]    = proj_state_snapshot.get("next_ms_net_payment", 0.0)
            combined["next_ms_earliest_t"]     = proj_state_snapshot.get("next_ms_earliest_t")
            combined["next_ms_is_final"]       = proj_state_snapshot.get("next_ms_is_final", False)
            combined["t_episode"]              = state["t_episode"]

            try:
                proj_tab = self.query_one(f"#tab-proj-{i}", ProjectTab)
                proj_tab.update(combined, proj, dict(env.cfg))
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

        t_executed         = self.state["t_episode"]
        self.state, reward, self.done, info = self.env.step(allocs)
        self.total_reward += reward

        try:
            port = self.query_one("#tab-portfolio", PortfolioTab)
            port.log_cashflow(t_executed, reward, info, self.state["budget"])
        except NoMatches:
            pass

        self._refresh_all()

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
    CSS                   = CSS
    TITLE                 = "Portfolio Budget Allocator"
    BINDINGS              = [Binding("q", "quit", "Quit")]
    DEFAULT_CSS           = ""
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