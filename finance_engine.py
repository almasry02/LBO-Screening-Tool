"""
LBO Finance Engine – Phase 1 MVP
Deterministic calculations: FCF, Debt Capacity, DSCR, IRR, MOIC
"""

import numpy as np
import pandas as pd

def _irr(cashflows: list) -> float:
    """Newton-Raphson IRR (replaces numpy_financial)"""
    cf = np.array(cashflows, dtype=float)
    rate = 0.10
    for _ in range(1000):
        t = np.arange(len(cf), dtype=float)
        npv = np.sum(cf / (1 + rate) ** t)
        dnpv = np.sum(-t * cf / (1 + rate) ** (t + 1))
        if abs(dnpv) < 1e-12:
            break
        rate -= npv / dnpv
        if rate <= -1:
            return float("nan")
    return rate
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class CompanyInputs:
    """Standardized company financials (post-mapping)"""
    # P&L
    revenue: float                  # €M
    ebitda: float                   # €M
    ebit: float                     # €M
    depreciation: float             # €M
    interest_expense: float         # €M
    tax_rate: float                 # e.g. 0.25

    # Balance Sheet
    total_debt: float               # €M
    cash: float                     # €M
    net_working_capital: float      # €M (optional, can be 0)

    # Capex
    capex: float                    # €M (maintenance capex)

    # Meta
    company_name: str = "Target"
    fiscal_year: int = 2024


@dataclass
class LBOAssumptions:
    """Deal & structure assumptions"""
    # Entry
    entry_ev_multiple: float = 6.0       # EV/EBITDA
    equity_contribution_pct: float = 0.40 # 40% equity / 60% debt

    # Debt Structure
    senior_debt_rate: float = 0.065      # 6.5% p.a.
    debt_amortization_years: int = 7     # years to repay

    # Exit
    exit_multiple: float = 7.0           # EV/EBITDA at exit
    holding_period: int = 5              # years

    # Growth
    revenue_cagr: float = 0.04           # 4% p.a.
    ebitda_margin_improvement: float = 0.005  # +50bps p.a.

    # Covenants
    max_leverage_covenant: float = 5.5   # Net Debt / EBITDA
    min_dscr_covenant: float = 1.20      # DSCR floor


@dataclass
class LBOResults:
    """All computed outputs"""
    # Entry
    entry_ev: float = 0.0
    entry_equity: float = 0.0
    entry_debt: float = 0.0
    entry_leverage: float = 0.0          # Net Debt / EBITDA

    # FCF
    fcf_series: list = field(default_factory=list)
    fcf_yield: float = 0.0               # FCF / EV

    # Debt
    debt_capacity: float = 0.0
    debt_schedule: pd.DataFrame = field(default_factory=pd.DataFrame)

    # DSCR
    dscr_base: float = 0.0
    dscr_series: list = field(default_factory=list)
    covenant_breach_year: Optional[int] = None

    # Returns
    exit_ev: float = 0.0
    exit_equity: float = 0.0
    irr: float = 0.0
    moic: float = 0.0

    # Flags
    red_flags: list = field(default_factory=list)
    is_lbo_viable: bool = False


# ─────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────

class LBOEngine:

    def __init__(self, inputs: CompanyInputs, assumptions: LBOAssumptions):
        self.c = inputs
        self.a = assumptions

    def run(self) -> LBOResults:
        res = LBOResults()

        self._calc_entry(res)
        self._calc_fcf_and_projections(res)
        self._calc_debt_schedule(res)
        self._calc_dscr(res)
        self._calc_returns(res)
        self._evaluate_viability(res)

        return res

    # ── Entry ──────────────────────────────────

    def _calc_entry(self, res: LBOResults):
        res.entry_ev = self.c.ebitda * self.a.entry_ev_multiple
        res.entry_equity = res.entry_ev * self.a.equity_contribution_pct
        res.entry_debt = res.entry_ev * (1 - self.a.equity_contribution_pct)
        res.entry_leverage = (res.entry_debt - self.c.cash) / self.c.ebitda

    # ── Free Cash Flow ──────────────────────────

    def _calc_fcf_and_projections(self, res: LBOResults):
        fcf_list = []
        rev = self.c.revenue
        margin = self.c.ebitda / self.c.revenue

        for year in range(1, self.a.holding_period + 1):
            rev *= (1 + self.a.revenue_cagr)
            margin += self.a.ebitda_margin_improvement
            ebitda_proj = rev * margin
            ebit_proj = ebitda_proj - self.c.depreciation
            nopat = ebit_proj * (1 - self.c.tax_rate)
            fcf = nopat + self.c.depreciation - self.c.capex - (0.01 * rev)  # ~1% NWC build
            fcf_list.append(round(fcf, 2))

        res.fcf_series = fcf_list
        res.fcf_yield = fcf_list[0] / res.entry_ev if res.entry_ev > 0 else 0

    # ── Debt Schedule ───────────────────────────

    def _calc_debt_schedule(self, res: LBOResults):
        """Straight-line amortization over holding period"""
        annual_amort = res.entry_debt / self.a.debt_amortization_years
        rows = []
        debt_bal = res.entry_debt

        for year in range(1, self.a.holding_period + 1):
            interest = debt_bal * self.a.senior_debt_rate
            amort = min(annual_amort, debt_bal)
            debt_bal = max(0, debt_bal - amort)
            rows.append({
                "Year": year,
                "Opening Debt (€M)": round(debt_bal + amort, 2),
                "Interest (€M)": round(interest, 2),
                "Amortization (€M)": round(amort, 2),
                "Closing Debt (€M)": round(debt_bal, 2),
            })

        res.debt_schedule = pd.DataFrame(rows).set_index("Year")
        res.debt_capacity = self.c.ebitda * 5.0  # rough 5x EBITDA max capacity

    # ── DSCR ────────────────────────────────────

    def _calc_dscr(self, res: LBOResults):
        dscr_list = []
        annual_amort = res.entry_debt / self.a.debt_amortization_years
        debt_bal = res.entry_debt

        for i, fcf in enumerate(res.fcf_series):
            interest = debt_bal * self.a.senior_debt_rate
            debt_service = interest + min(annual_amort, debt_bal)
            dscr = fcf / debt_service if debt_service > 0 else 99
            dscr_list.append(round(dscr, 3))
            debt_bal = max(0, debt_bal - annual_amort)

            # Covenant breach check
            if dscr < self.a.min_dscr_covenant and res.covenant_breach_year is None:
                res.covenant_breach_year = i + 1

        res.dscr_series = dscr_list
        res.dscr_base = dscr_list[0] if dscr_list else 0

    # ── Returns ─────────────────────────────────

    def _calc_returns(self, res: LBOResults):
        # Project EBITDA at exit
        rev = self.c.revenue
        margin = self.c.ebitda / self.c.revenue
        for _ in range(self.a.holding_period):
            rev *= (1 + self.a.revenue_cagr)
            margin += self.a.ebitda_margin_improvement
        exit_ebitda = rev * margin

        res.exit_ev = exit_ebitda * self.a.exit_multiple
        exit_debt = res.debt_schedule["Closing Debt (€M)"].iloc[-1]
        res.exit_equity = max(0, res.exit_ev - exit_debt + self.c.cash)

        # IRR & MOIC
        cash_flows = [-res.entry_equity] + [0] * (self.a.holding_period - 1) + [res.exit_equity]
        res.moic = res.exit_equity / res.entry_equity if res.entry_equity > 0 else 0
        try:
            res.irr = _irr(cash_flows)
            if np.isnan(res.irr):
                res.irr = 0.0
        except Exception:
            res.irr = 0.0

    # ── Viability & Red Flags ───────────────────

    def _evaluate_viability(self, res: LBOResults):
        flags = []

        if res.entry_leverage > 6.0:
            flags.append(f"⚠️ Entry Leverage {res.entry_leverage:.1f}x exceeds 6.0x – aggressive structure")
        if res.dscr_base < 1.20:
            flags.append(f"🔴 DSCR Year 1 = {res.dscr_base:.2f}x – below 1.20x covenant floor")
        if res.irr < 0.15:
            flags.append(f"🔴 IRR {res.irr:.1%} – below typical PE hurdle of 15%")
        if res.moic < 2.0:
            flags.append(f"⚠️ MOIC {res.moic:.1f}x – below typical 2.0x threshold")
        if res.covenant_breach_year:
            flags.append(f"🔴 Covenant breach projected in Year {res.covenant_breach_year}")
        if self.c.ebitda / self.c.revenue < 0.08:
            flags.append(f"⚠️ EBITDA margin {self.c.ebitda/self.c.revenue:.1%} – thin margin, limited debt service buffer")
        if res.fcf_yield < 0.04:
            flags.append(f"⚠️ FCF yield {res.fcf_yield:.1%} – low free cash generation relative to entry EV")

        res.red_flags = flags
        res.is_lbo_viable = (
            res.irr >= 0.15 and
            res.moic >= 2.0 and
            res.dscr_base >= 1.10 and
            res.entry_leverage <= 7.0
        )


# ─────────────────────────────────────────────
# SENSITIVITY ENGINE
# ─────────────────────────────────────────────

class SensitivityEngine:
    """Generates heatmap data for IRR sensitivity"""

    def __init__(self, inputs: CompanyInputs, base_assumptions: LBOAssumptions):
        self.inputs = inputs
        self.base = base_assumptions

    def irr_heatmap(
        self,
        exit_multiples: list,
        ebitda_cagrs: list,
    ) -> pd.DataFrame:
        """IRR heatmap: rows = exit multiples, cols = EBITDA CAGR"""
        rows = {}
        for em in exit_multiples:
            row = {}
            for cagr in ebitda_cagrs:
                a = LBOAssumptions(
                    entry_ev_multiple=self.base.entry_ev_multiple,
                    equity_contribution_pct=self.base.equity_contribution_pct,
                    senior_debt_rate=self.base.senior_debt_rate,
                    debt_amortization_years=self.base.debt_amortization_years,
                    exit_multiple=em,
                    holding_period=self.base.holding_period,
                    revenue_cagr=cagr,
                    ebitda_margin_improvement=self.base.ebitda_margin_improvement,
                    max_leverage_covenant=self.base.max_leverage_covenant,
                    min_dscr_covenant=self.base.min_dscr_covenant,
                )
                engine = LBOEngine(self.inputs, a)
                res = engine.run()
                row[f"{cagr:.0%}"] = round(res.irr * 100, 1)
            rows[f"{em:.1f}x"] = row

        return pd.DataFrame(rows).T

    def dscr_heatmap(
        self,
        interest_rates: list,
        leverage_multiples: list,
    ) -> pd.DataFrame:
        """DSCR heatmap: rows = leverage, cols = interest rate"""
        rows = {}
        for lev in leverage_multiples:
            row = {}
            for rate in interest_rates:
                equity_pct = 1 - (lev / (self.base.entry_ev_multiple))
                equity_pct = max(0.20, min(0.80, equity_pct))
                a = LBOAssumptions(
                    entry_ev_multiple=self.base.entry_ev_multiple,
                    equity_contribution_pct=equity_pct,
                    senior_debt_rate=rate,
                    debt_amortization_years=self.base.debt_amortization_years,
                    exit_multiple=self.base.exit_multiple,
                    holding_period=self.base.holding_period,
                    revenue_cagr=self.base.revenue_cagr,
                    ebitda_margin_improvement=self.base.ebitda_margin_improvement,
                )
                engine = LBOEngine(self.inputs, a)
                res = engine.run()
                row[f"{rate:.1%}"] = round(res.dscr_base, 2)
            rows[f"{lev:.1f}x"] = row

        return pd.DataFrame(rows).T
