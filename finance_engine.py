"""
LBO Finance Engine v3
─────────────────────────────────────────────────────────────────────────────
Sources:
  [R&P]  Rosenbaum & Pearl – Investment Banking: Valuation, LBOs, M&A (2020)
  [MPE]  Mastering Private Equity / Private Equity at Work
  [McK]  McKinsey – Valuation: Measuring and Managing the Value of Companies
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# IRR (Newton-Raphson, no numpy_financial dep)
# ─────────────────────────────────────────────

def _irr(cashflows: list) -> float:
    """[R&P Ch.5] Newton-Raphson IRR – converges in <100 iterations for typical LBO CFs"""
    cf = np.array(cashflows, dtype=float)
    rate = 0.15
    for _ in range(2000):
        t    = np.arange(len(cf), dtype=float)
        npv  = np.sum(cf / (1 + rate) ** t)
        dnpv = np.sum(-t * cf / (1 + rate) ** (t + 1))
        if abs(dnpv) < 1e-14:
            break
        new_rate = rate - npv / dnpv
        if new_rate <= -1:
            return float("nan")
        if abs(new_rate - rate) < 1e-10:
            rate = new_rate
            break
        rate = new_rate
    return rate


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class HistoricalYear:
    """Single year of historical financials (in native currency units)"""
    year:               int
    revenue:            float
    ebitda:             float
    ebit:               float
    depreciation:       float
    interest_expense:   float
    net_income:         float
    total_debt:         float
    cash:               float
    net_working_capital: float
    capex:              float
    tax_rate:           float = 0.25
    
    @property
    def ebitda_margin(self) -> float:
        return self.ebitda / self.revenue if self.revenue else 0.0

    @property
    def ebit_margin(self) -> float:
        return self.ebit / self.revenue if self.revenue else 0.0

    @property
    def capex_intensity(self) -> float:
        return self.capex / self.revenue if self.revenue else 0.0

    @property
    def nwc_intensity(self) -> float:
        return self.net_working_capital / self.revenue if self.revenue else 0.0

    @property
    def net_debt(self) -> float:
        return self.total_debt - self.cash

    @property
    def interest_coverage(self) -> float:
        return self.ebitda / self.interest_expense if self.interest_expense > 0 else 99.0

    @property
    def fcf_conversion(self) -> float:
        """[McK] FCF / EBITDA – measures earnings quality"""
        nopat = self.ebit * (1 - self.tax_rate)
        fcf   = nopat + self.depreciation - self.capex
        return fcf / self.ebitda if self.ebitda else 0.0


@dataclass
class HistoricalMetrics:
    """[R&P + McK] Aggregated historical analytics across multiple years"""
    years_used:         list
    revenue_cagr:       float           # compound annual growth rate
    ebitda_margin_avg:  float           # simple average
    ebitda_margin_med:  float           # median (robust to outliers)
    ebitda_volatility:  float           # std dev of annual margins [MPE]
    capex_intensity_avg: float          # avg CapEx / Revenue
    nwc_intensity_avg:  float           # avg NWC / Revenue
    fcf_conversion_avg: float           # avg FCF / EBITDA [McK]
    interest_coverage_avg: float        # avg EBITDA / Interest
    revenue_series:     list
    ebitda_series:      list
    margin_series:      list
    # Normalized base for LBO entry (latest year adjusted)
    normalized_revenue: float
    normalized_ebitda:  float
    normalized_capex:   float
    normalized_nwc_delta: float         # annual NWC build = nwc_intensity × rev_growth


@dataclass
class CompanyInputs:
    """Normalized entry-point financials for LBO engine (post historical analysis)"""
    revenue:            float
    ebitda:             float
    ebit:               float
    depreciation:       float
    interest_expense:   float
    tax_rate:           float
    total_debt:         float
    cash:               float
    net_working_capital: float
    capex:              float
    company_name:       str = "Target"
    currency_display:   str = "USD (tsd)"
    # Derived from historical analysis
    revenue_cagr_hist:  float = 0.04
    ebitda_margin_avg:  float = 0.20
    capex_intensity:    float = 0.04
    nwc_intensity:      float = 0.10


@dataclass
class LBOAssumptions:
    """[R&P Ch.4] Deal structure assumptions"""
    entry_ev_multiple:      float = 6.5
    equity_contribution_pct: float = 0.40
    senior_debt_rate:       float = 0.065
    debt_amortization_years: int  = 7
    exit_multiple:          float = 7.0
    holding_period:         int   = 5
    revenue_cagr:           float = 0.04   # forward CAGR assumption
    ebitda_margin_entry:    float = 0.20   # normalized margin at entry
    ebitda_margin_exit:     float = 0.22   # slight improvement assumed
    max_leverage_covenant:  float = 5.5
    min_dscr_covenant:      float = 1.20


@dataclass
class LBOResults:
    """Full LBO output set"""
    # Entry
    entry_ev:           float = 0.0
    entry_equity:       float = 0.0
    entry_debt:         float = 0.0
    entry_leverage:     float = 0.0    # Net Debt / EBITDA [R&P]
    interest_coverage:  float = 0.0   # EBITDA / Interest

    # Projections (5-year)
    revenue_proj:       list = field(default_factory=list)
    ebitda_proj:        list = field(default_factory=list)
    fcf_series:         list = field(default_factory=list)
    fcf_yield:          float = 0.0

    # Debt
    debt_capacity:      float = 0.0
    debt_schedule:      pd.DataFrame = field(default_factory=pd.DataFrame)

    # DSCR [MPE]
    dscr_base:          float = 0.0
    dscr_series:        list  = field(default_factory=list)
    covenant_breach_year: Optional[int] = None

    # Returns [R&P Ch.5]
    exit_ebitda:        float = 0.0
    exit_ev:            float = 0.0
    exit_equity:        float = 0.0
    irr:                float = 0.0
    moic:               float = 0.0

    # Downside
    downside_irr:       float = 0.0
    downside_moic:      float = 0.0

    # Flags
    red_flags:          list  = field(default_factory=list)
    is_lbo_viable:      bool  = False


# ─────────────────────────────────────────────
# HISTORICAL ANALYTICS ENGINE
# ─────────────────────────────────────────────

class HistoricalAnalyzer:
    """
    [McK + R&P] Derives normalized operating metrics from 3-10 years of history.
    Follows PE analyst practice: use historical CAGR + average margins as
    forward base case, not a single year snapshot.
    """

    def __init__(self, years: list[HistoricalYear]):
        # Sort ascending (oldest first)
        self.years = sorted(years, key=lambda y: y.year)

    def compute(self) -> HistoricalMetrics:
        yrs  = self.years
        n    = len(yrs)
        first, last = yrs[0], yrs[-1]

        # ── Revenue CAGR [R&P p.204] ─────────────
        if first.revenue > 0 and n > 1:
            rev_cagr = (last.revenue / first.revenue) ** (1 / (n - 1)) - 1
        else:
            rev_cagr = 0.03

        # ── Margin metrics ────────────────────────
        margins   = [y.ebitda_margin for y in yrs]
        cap_intys = [y.capex_intensity for y in yrs]
        nwc_intys = [y.nwc_intensity for y in yrs]
        fcf_convs = [y.fcf_conversion for y in yrs]
        cov_ratios = [y.interest_coverage for y in yrs]

        margin_avg  = float(np.mean(margins))
        margin_med  = float(np.median(margins))
        margin_vol  = float(np.std(margins))        # [MPE] EBITDA volatility signal
        capex_avg   = float(np.mean(cap_intys))
        nwc_avg     = float(np.mean(nwc_intys))
        fcf_conv    = float(np.mean(fcf_convs))
        ic_avg      = float(np.mean(cov_ratios))

        # ── Normalized base [R&P: "scrubbed EBITDA"] ─
        # Use 3-year average margin applied to latest revenue for normalization
        norm_margin   = margin_avg
        norm_revenue  = last.revenue
        norm_ebitda   = norm_revenue * norm_margin
        norm_capex    = norm_revenue * capex_avg
        # Annual NWC build = nwc_intensity × revenue_growth_amount
        norm_nwc_delta = norm_revenue * rev_cagr * nwc_avg

        return HistoricalMetrics(
            years_used          = [y.year for y in yrs],
            revenue_cagr        = rev_cagr,
            ebitda_margin_avg   = margin_avg,
            ebitda_margin_med   = margin_med,
            ebitda_volatility   = margin_vol,
            capex_intensity_avg = capex_avg,
            nwc_intensity_avg   = nwc_avg,
            fcf_conversion_avg  = fcf_conv,
            interest_coverage_avg = ic_avg,
            revenue_series      = [y.revenue for y in yrs],
            ebitda_series       = [y.ebitda for y in yrs],
            margin_series       = margins,
            normalized_revenue  = norm_revenue,
            normalized_ebitda   = norm_ebitda,
            normalized_capex    = norm_capex,
            normalized_nwc_delta = norm_nwc_delta,
        )


# ─────────────────────────────────────────────
# CORE LBO ENGINE
# ─────────────────────────────────────────────

class LBOEngine:
    """
    [R&P Ch.4-5] Full deterministic LBO model.
    Forward projections based on normalized historical metrics.
    FCF = NOPAT + D&A – CapEx – ΔNWC  [McK p.163]
    """

    def __init__(self, inputs: CompanyInputs, assumptions: LBOAssumptions,
                 historical_metrics: Optional[HistoricalMetrics] = None):
        self.c    = inputs
        self.a    = assumptions
        self.hist = historical_metrics

    def run(self) -> LBOResults:
        res = LBOResults()
        self._calc_entry(res)
        self._calc_projections(res)
        self._calc_debt_schedule(res)
        self._calc_dscr(res)
        self._calc_returns(res)
        self._calc_downside(res)
        self._evaluate_flags(res)
        return res

    # ── Entry Structure [R&P p.152] ──────────────

    def _calc_entry(self, res: LBOResults):
        res.entry_ev      = self.c.ebitda * self.a.entry_ev_multiple
        res.entry_equity  = res.entry_ev * self.a.equity_contribution_pct
        res.entry_debt    = res.entry_ev * (1 - self.a.equity_contribution_pct)
        net_debt          = res.entry_debt - self.c.cash
        res.entry_leverage = net_debt / self.c.ebitda if self.c.ebitda > 0 else 0
        res.interest_coverage = self.c.ebitda / max(self.c.interest_expense, 0.001)
        res.debt_capacity  = self.c.ebitda * self.a.max_leverage_covenant

    # ── 5-Year Forward Projections ────────────────

    def _calc_projections(self, res: LBOResults):
        """
        [McK p.193] Simple forward case:
        Revenue_t = Revenue × (1 + CAGR)^t
        EBITDA_t  = Revenue_t × margin (linearly interpolated to exit margin)
        FCF_t     = NOPAT + D&A – CapEx – ΔNWC
        """
        rev     = self.c.revenue
        # D&A grows proportionally with revenue (capex-driven)
        dep     = self.c.depreciation
        capex_intensity = self.c.capex / max(self.c.revenue, 1)
        nwc_intensity   = self.c.net_working_capital / max(self.c.revenue, 1)

        rev_list, ebitda_list, fcf_list = [], [], []
        margin_entry = self.c.ebitda / max(self.c.revenue, 1)
        margin_exit  = self.a.ebitda_margin_exit
        hp           = self.a.holding_period

        for t in range(1, hp + 1):
            rev_t   = rev * (1 + self.a.revenue_cagr) ** t
            # Linear margin interpolation entry→exit [R&P: "operational improvement"]
            margin_t = margin_entry + (margin_exit - margin_entry) * (t / hp)
            ebitda_t = rev_t * margin_t
            dep_t    = dep * (1 + self.a.revenue_cagr) ** t   # D&A grows with asset base
            ebit_t   = ebitda_t - dep_t
            nopat_t  = ebit_t * (1 - self.c.tax_rate)
            capex_t  = rev_t * capex_intensity
            # ΔNWC = change in NWC = nwc_intensity × Δrevenue [McK p.167]
            rev_prev = rev * (1 + self.a.revenue_cagr) ** (t - 1)
            delta_nwc = nwc_intensity * (rev_t - rev_prev)
            fcf_t    = nopat_t + dep_t - capex_t - delta_nwc

            rev_list.append(round(rev_t, 1))
            ebitda_list.append(round(ebitda_t, 1))
            fcf_list.append(round(fcf_t, 1))

        res.revenue_proj = rev_list
        res.ebitda_proj  = ebitda_list
        res.fcf_series   = fcf_list
        res.fcf_yield    = fcf_list[0] / res.entry_ev if res.entry_ev > 0 else 0

    # ── Debt Schedule [R&P p.156] ─────────────────

    def _calc_debt_schedule(self, res: LBOResults):
        """Straight-line senior debt amortization + cash sweep"""
        annual_amort = res.entry_debt / self.a.debt_amortization_years
        rows, debt_bal = [], res.entry_debt

        for year in range(1, self.a.holding_period + 1):
            opening   = debt_bal
            interest  = debt_bal * self.a.senior_debt_rate
            # Cash sweep: repay more if FCF allows [R&P p.157]
            fcf_avail = res.fcf_series[year - 1] if year <= len(res.fcf_series) else 0
            sweep     = max(0, fcf_avail - interest - annual_amort)
            amort     = min(annual_amort + sweep, debt_bal)
            debt_bal  = max(0, debt_bal - amort)
            rows.append({
                "Year":         year,
                "Opening":      round(opening, 1),
                "Interest":     round(interest, 1),
                "Amortization": round(amort, 1),
                "Cash Sweep":   round(sweep, 1),
                "Closing":      round(debt_bal, 1),
                "Coverage":     round(res.ebitda_proj[year-1] / max(interest, 0.001), 2)
                                if year <= len(res.ebitda_proj) else 0,
            })
        res.debt_schedule = pd.DataFrame(rows).set_index("Year")

    # ── DSCR [MPE, Rosenbaum p.158] ──────────────

    def _calc_dscr(self, res: LBOResults):
        """DSCR = FCF / (Interest + Scheduled Amortization)"""
        annual_amort = res.entry_debt / self.a.debt_amortization_years
        dscr_list, debt_bal = [], res.entry_debt

        for i, fcf in enumerate(res.fcf_series):
            interest     = debt_bal * self.a.senior_debt_rate
            debt_service = interest + min(annual_amort, debt_bal)
            dscr         = fcf / debt_service if debt_service > 0 else 99.0
            dscr_list.append(round(dscr, 3))
            debt_bal = max(0, debt_bal - annual_amort)
            if dscr < self.a.min_dscr_covenant and res.covenant_breach_year is None:
                res.covenant_breach_year = i + 1

        res.dscr_series = dscr_list
        res.dscr_base   = dscr_list[0] if dscr_list else 0

    # ── Returns [R&P p.160] ───────────────────────

    def _calc_returns(self, res: LBOResults):
        """IRR & MOIC on sponsor equity"""
        # Exit EBITDA = last projected year
        res.exit_ebitda = res.ebitda_proj[-1] if res.ebitda_proj else self.c.ebitda
        res.exit_ev     = res.exit_ebitda * self.a.exit_multiple
        exit_debt       = res.debt_schedule["Closing"].iloc[-1]
        res.exit_equity = max(0, res.exit_ev - exit_debt + self.c.cash)
        res.moic        = res.exit_equity / res.entry_equity if res.entry_equity > 0 else 0

        # IRR: equity invested at t=0, received at t=HP
        cfs = ([-res.entry_equity]
               + [0] * (self.a.holding_period - 1)
               + [res.exit_equity])
        irr = _irr(cfs)
        res.irr = irr if not np.isnan(irr) else 0.0

    # ── Downside Case [MPE] ───────────────────────

    def _calc_downside(self, res: LBOResults):
        """
        [MPE] Stress test: Revenue -10% at entry, EBITDA margin -200bps,
        exit multiple -1.0x. Represents a mild recession scenario.
        """
        c_stress = CompanyInputs(
            revenue             = self.c.revenue * 0.90,
            ebitda              = self.c.ebitda  * 0.85,
            ebit                = self.c.ebit    * 0.80,
            depreciation        = self.c.depreciation,
            interest_expense    = self.c.interest_expense,
            tax_rate            = self.c.tax_rate,
            total_debt          = self.c.total_debt,
            cash                = self.c.cash,
            net_working_capital = self.c.net_working_capital,
            capex               = self.c.capex,
            company_name        = self.c.company_name,
        )
        a_stress = LBOAssumptions(
            entry_ev_multiple       = self.a.entry_ev_multiple,
            equity_contribution_pct = self.a.equity_contribution_pct,
            senior_debt_rate        = self.a.senior_debt_rate,
            debt_amortization_years = self.a.debt_amortization_years,
            exit_multiple           = self.a.exit_multiple - 1.0,
            holding_period          = self.a.holding_period,
            revenue_cagr            = max(0, self.a.revenue_cagr - 0.03),
            ebitda_margin_entry     = self.a.ebitda_margin_entry - 0.02,
            ebitda_margin_exit      = self.a.ebitda_margin_exit  - 0.02,
            max_leverage_covenant   = self.a.max_leverage_covenant,
            min_dscr_covenant       = self.a.min_dscr_covenant,
        )
        try:
            stress_eng       = LBOEngine(c_stress, a_stress)
            stress_res       = stress_eng.run()
            res.downside_irr  = stress_res.irr
            res.downside_moic = stress_res.moic
        except Exception:
            res.downside_irr  = 0.0
            res.downside_moic = 0.0

    # ── Red Flag Evaluation ───────────────────────

    def _evaluate_flags(self, res: LBOResults):
        """Initial flag evaluation – overridden in app.py with user thresholds"""
        flags = []
        c, a  = self.c, self.a

        if res.entry_leverage > 6.0:
            flags.append(f"🔴 Entry Leverage {res.entry_leverage:.1f}x > 6.0x")
        if res.dscr_base < 1.20:
            flags.append(f"🔴 DSCR Y1 {res.dscr_base:.2f}x < 1.20x Covenant Floor")
        if res.irr < 0.20:
            flags.append(f"🔴 IRR {res.irr:.1%} < 20% Hurdle")
        if res.moic < 2.0:
            flags.append(f"⚠️ MOIC {res.moic:.1f}x < 2.0x")
        if res.covenant_breach_year:
            flags.append(f"🔴 Covenant Breach in Jahr {res.covenant_breach_year}")
        if c.ebitda / max(c.revenue, 1) < 0.08:
            flags.append(f"⚠️ EBITDA-Marge {c.ebitda/c.revenue:.1%} < 8% – dünner Puffer")
        if res.fcf_yield < 0.04:
            flags.append(f"⚠️ FCF Yield {res.fcf_yield:.1%} < 4%")

        res.red_flags      = flags
        res.is_lbo_viable  = (
            res.irr  >= 0.20 and res.moic >= 2.0
            and res.dscr_base >= 1.10 and res.entry_leverage <= 7.0
        )


# ─────────────────────────────────────────────
# SENSITIVITY ENGINE
# ─────────────────────────────────────────────

class SensitivityEngine:
    """[R&P p.200] Generates heatmap data for IRR and DSCR sensitivities"""

    def __init__(self, inputs: CompanyInputs, base_assumptions: LBOAssumptions):
        self.inputs = inputs
        self.base   = base_assumptions

    def irr_heatmap(self, exit_multiples: list, revenue_cagrs: list) -> pd.DataFrame:
        """IRR vs Exit Multiple × Revenue CAGR"""
        rows = {}
        for em in exit_multiples:
            row = {}
            for cagr in revenue_cagrs:
                a = LBOAssumptions(
                    entry_ev_multiple       = self.base.entry_ev_multiple,
                    equity_contribution_pct = self.base.equity_contribution_pct,
                    senior_debt_rate        = self.base.senior_debt_rate,
                    debt_amortization_years = self.base.debt_amortization_years,
                    exit_multiple           = em,
                    holding_period          = self.base.holding_period,
                    revenue_cagr            = cagr,
                    ebitda_margin_entry     = self.base.ebitda_margin_entry,
                    ebitda_margin_exit      = self.base.ebitda_margin_exit,
                    max_leverage_covenant   = self.base.max_leverage_covenant,
                    min_dscr_covenant       = self.base.min_dscr_covenant,
                )
                res = LBOEngine(self.inputs, a).run()
                row[f"{cagr:.0%}"] = round(res.irr * 100, 1)
            rows[f"{em:.1f}x"] = row
        return pd.DataFrame(rows).T

    def dscr_heatmap(self, interest_rates: list, leverage_multiples: list) -> pd.DataFrame:
        """DSCR vs Leverage × Interest Rate [MPE]"""
        rows = {}
        for lev in leverage_multiples:
            row = {}
            for rate in interest_rates:
                eq_pct = max(0.20, min(0.80, 1 - (lev / self.base.entry_ev_multiple)))
                a = LBOAssumptions(
                    entry_ev_multiple       = self.base.entry_ev_multiple,
                    equity_contribution_pct = eq_pct,
                    senior_debt_rate        = rate,
                    debt_amortization_years = self.base.debt_amortization_years,
                    exit_multiple           = self.base.exit_multiple,
                    holding_period          = self.base.holding_period,
                    revenue_cagr            = self.base.revenue_cagr,
                    ebitda_margin_entry     = self.base.ebitda_margin_entry,
                    ebitda_margin_exit      = self.base.ebitda_margin_exit,
                )
                res = LBOEngine(self.inputs, a).run()
                row[f"{rate:.1%}"] = round(res.dscr_base, 2)
            rows[f"{lev:.1f}x"] = row
        return pd.DataFrame(rows).T

    def leverage_irr_heatmap(self, exit_multiples: list, equity_pcts: list) -> pd.DataFrame:
        """[R&P] IRR vs Exit Multiple × Equity Contribution (leverage sensitivity)"""
        rows = {}
        for em in exit_multiples:
            row = {}
            for eq in equity_pcts:
                a = LBOAssumptions(
                    entry_ev_multiple       = self.base.entry_ev_multiple,
                    equity_contribution_pct = eq,
                    senior_debt_rate        = self.base.senior_debt_rate,
                    debt_amortization_years = self.base.debt_amortization_years,
                    exit_multiple           = em,
                    holding_period          = self.base.holding_period,
                    revenue_cagr            = self.base.revenue_cagr,
                    ebitda_margin_entry     = self.base.ebitda_margin_entry,
                    ebitda_margin_exit      = self.base.ebitda_margin_exit,
                )
                res = LBOEngine(self.inputs, a).run()
                row[f"{eq:.0%} Equity"] = round(res.irr * 100, 1)
            rows[f"{em:.1f}x"] = row
        return pd.DataFrame(rows).T