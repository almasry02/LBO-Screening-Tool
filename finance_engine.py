"""
LBO Finance Engine v4
Sources: [R&P] Rosenbaum & Pearl 2020 | [MPE] Mastering Private Equity | [McK] McKinsey Valuation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ── IRR Newton-Raphson ────────────────────────────────────────────────────────
def _irr(cashflows: list) -> float:
    cf = np.array(cashflows, dtype=float)
    rate = 0.15
    for _ in range(2000):
        t    = np.arange(len(cf), dtype=float)
        npv  = np.sum(cf / (1+rate)**t)
        dnpv = np.sum(-t * cf / (1+rate)**(t+1))
        if abs(dnpv) < 1e-14: break
        nr = rate - npv/dnpv
        if nr <= -1: return float("nan")
        if abs(nr - rate) < 1e-10: rate = nr; break
        rate = nr
    return rate

# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class HistoricalYear:
    year: int; revenue: float; ebitda: float; ebit: float
    depreciation: float; interest_expense: float; net_income: float
    total_debt: float; cash: float; net_working_capital: float
    capex: float; tax_rate: float = 0.25

    @property
    def ebitda_margin(self): return self.ebitda/self.revenue if self.revenue else 0.0
    @property
    def ebit_margin(self): return self.ebit/self.revenue if self.revenue else 0.0
    @property
    def capex_intensity(self): return self.capex/self.revenue if self.revenue else 0.0
    @property
    def nwc_intensity(self): return self.net_working_capital/self.revenue if self.revenue else 0.0
    @property
    def net_debt(self): return self.total_debt - self.cash
    @property
    def interest_coverage(self): return self.ebitda/self.interest_expense if self.interest_expense>0 else 99.0
    @property
    def fcf_conversion(self):
        nopat = self.ebit*(1-self.tax_rate)
        return (nopat+self.depreciation-self.capex)/self.ebitda if self.ebitda else 0.0


@dataclass
class HistoricalMetrics:
    years_used: list; revenue_cagr: float; ebitda_margin_avg: float
    ebitda_margin_med: float; ebitda_volatility: float
    capex_intensity_avg: float; nwc_intensity_avg: float
    fcf_conversion_avg: float; interest_coverage_avg: float
    revenue_series: list; ebitda_series: list; margin_series: list
    revenue_volatility: float
    normalized_revenue: float; normalized_ebitda: float
    normalized_capex: float; normalized_nwc_delta: float


@dataclass
class CompanyInputs:
    revenue: float; ebitda: float; ebit: float; depreciation: float
    interest_expense: float; tax_rate: float; total_debt: float
    cash: float; net_working_capital: float; capex: float
    company_name: str = "Target"; currency_display: str = "EUR (tsd)"
    revenue_cagr_hist: float = 0.04; ebitda_margin_avg: float = 0.20
    capex_intensity: float = 0.04; nwc_intensity: float = 0.10


@dataclass
class LBOAssumptions:
    entry_ev_multiple: float = 6.5; equity_contribution_pct: float = 0.40
    senior_debt_rate: float = 0.065; debt_amortization_years: int = 7
    exit_multiple: float = 7.0; holding_period: int = 5
    revenue_cagr: float = 0.04; ebitda_margin_entry: float = 0.20
    ebitda_margin_exit: float = 0.22; max_leverage_covenant: float = 5.5
    min_dscr_covenant: float = 1.20


@dataclass
class LBOResults:
    # Entry
    entry_ev: float=0.0; entry_equity: float=0.0; entry_debt: float=0.0
    entry_leverage: float=0.0; interest_coverage: float=0.0
    # Projections
    revenue_proj: list=field(default_factory=list)
    ebitda_proj: list=field(default_factory=list)
    fcf_series: list=field(default_factory=list)
    fcf_yield: float=0.0
    # Debt
    debt_capacity: float=0.0; debt_capacity_dscr: float=0.0
    debt_schedule: pd.DataFrame=field(default_factory=pd.DataFrame)
    # DSCR
    dscr_base: float=0.0; dscr_series: list=field(default_factory=list)
    covenant_breach_year: Optional[int]=None
    # Returns
    exit_ebitda: float=0.0; exit_ev: float=0.0; exit_equity: float=0.0
    irr: float=0.0; moic: float=0.0
    # Downside
    downside_irr: float=0.0; downside_moic: float=0.0
    # Cash Conversion [McK]
    cash_conversion: float=0.0
    # Value Creation Bridge [R&P p.162]
    vc_ebitda_growth: float=0.0; vc_multiple_exp: float=0.0; vc_debt_paydown: float=0.0
    # LBO Score
    lbo_score: float=0.0; lbo_score_breakdown: dict=field(default_factory=dict)
    # Revenue Quality
    revenue_quality_flag: str=""
    # Flags
    red_flags: list=field(default_factory=list); is_lbo_viable: bool=False


# ── Historical Analyzer ───────────────────────────────────────────────────────

class HistoricalAnalyzer:
    def __init__(self, years: list):
        self.years = sorted(years, key=lambda y: y.year)

    def compute(self) -> HistoricalMetrics:
        yrs = self.years; n = len(yrs)
        first, last = yrs[0], yrs[-1]
        rev_cagr = (last.revenue/first.revenue)**(1/(n-1))-1 if first.revenue>0 and n>1 else 0.03
        margins   = [y.ebitda_margin for y in yrs]
        cap_intys = [y.capex_intensity for y in yrs]
        nwc_intys = [y.nwc_intensity for y in yrs]
        fcf_convs = [y.fcf_conversion for y in yrs]
        cov_r     = [y.interest_coverage for y in yrs]
        rev_series = [y.revenue for y in yrs]
        rev_mean  = float(np.mean(rev_series))
        rev_vol   = float(np.std(rev_series)/rev_mean) if rev_mean>0 else 0.0
        margin_avg = float(np.mean(margins))
        norm_rev  = last.revenue
        return HistoricalMetrics(
            years_used=[y.year for y in yrs], revenue_cagr=rev_cagr,
            ebitda_margin_avg=margin_avg, ebitda_margin_med=float(np.median(margins)),
            ebitda_volatility=float(np.std(margins)), capex_intensity_avg=float(np.mean(cap_intys)),
            nwc_intensity_avg=float(np.mean(nwc_intys)), fcf_conversion_avg=float(np.mean(fcf_convs)),
            interest_coverage_avg=float(np.mean(cov_r)),
            revenue_series=rev_series, ebitda_series=[y.ebitda for y in yrs],
            margin_series=margins, revenue_volatility=rev_vol,
            normalized_revenue=norm_rev, normalized_ebitda=norm_rev*margin_avg,
            normalized_capex=norm_rev*float(np.mean(cap_intys)),
            normalized_nwc_delta=norm_rev*rev_cagr*float(np.mean(nwc_intys)),
        )


# ── LBO Engine ────────────────────────────────────────────────────────────────

class LBOEngine:
    def __init__(self, inputs: CompanyInputs, assumptions: LBOAssumptions,
                 historical_metrics: Optional[HistoricalMetrics]=None):
        self.c=inputs; self.a=assumptions; self.hist=historical_metrics

    def run(self) -> LBOResults:
        res = LBOResults()
        self._entry(res); self._projections(res); self._debt_schedule(res)
        self._dscr(res); self._returns(res); self._downside(res)
        self._cash_conversion(res); self._value_bridge(res)
        self._debt_capacity_dscr(res); self._lbo_score(res); self._flags(res)
        return res

    def fast_run(self) -> LBOResults:
        """Minimal run for heatmaps — skips downside/score/bridge/DSCR-capacity (100x faster)."""
        res = LBOResults()
        self._entry(res); self._projections(res); self._debt_schedule(res)
        self._dscr(res); self._returns(res)
        return res

    def _entry(self, res):
        res.entry_ev      = self.c.ebitda * self.a.entry_ev_multiple
        res.entry_equity  = res.entry_ev * self.a.equity_contribution_pct
        res.entry_debt    = res.entry_ev * (1-self.a.equity_contribution_pct)
        res.entry_leverage = (res.entry_debt-self.c.cash)/self.c.ebitda if self.c.ebitda>0 else 0
        res.interest_coverage = self.c.ebitda/max(self.c.interest_expense,0.001)
        res.debt_capacity = self.c.ebitda * self.a.max_leverage_covenant

    def _projections(self, res):
        rev=self.c.revenue; dep=self.c.depreciation
        cap_i=self.c.capex/max(self.c.revenue,1); nwc_i=self.c.net_working_capital/max(self.c.revenue,1)
        m_entry=self.c.ebitda/max(self.c.revenue,1); m_exit=self.a.ebitda_margin_exit; hp=self.a.holding_period
        rl,el,fl=[],[],[]
        for t in range(1,hp+1):
            rt=rev*(1+self.a.revenue_cagr)**t; mt=m_entry+(m_exit-m_entry)*(t/hp)
            et=rt*mt; dt=dep*(1+self.a.revenue_cagr)**t; ebit_t=et-dt
            nopat=ebit_t*(1-self.c.tax_rate); capex_t=rt*cap_i
            rp=rev*(1+self.a.revenue_cagr)**(t-1); dnwc=nwc_i*(rt-rp)
            rl.append(round(rt,1)); el.append(round(et,1)); fl.append(round(nopat+dt-capex_t-dnwc,1))
        res.revenue_proj=rl; res.ebitda_proj=el; res.fcf_series=fl
        res.fcf_yield=fl[0]/res.entry_ev if res.entry_ev>0 else 0

    def _debt_schedule(self, res):
        amort=res.entry_debt/self.a.debt_amortization_years; rows=[]; db=res.entry_debt
        for yr in range(1,self.a.holding_period+1):
            op=db; int_=db*self.a.senior_debt_rate
            fcf_a=res.fcf_series[yr-1] if yr<=len(res.fcf_series) else 0
            sweep=max(0,fcf_a-int_-amort); am=min(amort+sweep,db); db=max(0,db-am)
            cov=res.ebitda_proj[yr-1]/max(int_,0.001) if yr<=len(res.ebitda_proj) else 0
            rows.append({"Year":yr,"Opening":round(op,1),"Interest":round(int_,1),
                "Amortization":round(am,1),"Cash Sweep":round(sweep,1),
                "Closing":round(db,1),"Coverage":round(cov,2)})
        res.debt_schedule=pd.DataFrame(rows).set_index("Year")

    def _dscr(self, res):
        amort=res.entry_debt/self.a.debt_amortization_years; dl=[]; db=res.entry_debt
        for i,fcf in enumerate(res.fcf_series):
            int_=db*self.a.senior_debt_rate; ds=int_+min(amort,db)
            d=fcf/ds if ds>0 else 99.0; dl.append(round(d,3)); db=max(0,db-amort)
            if d<self.a.min_dscr_covenant and res.covenant_breach_year is None:
                res.covenant_breach_year=i+1
        res.dscr_series=dl; res.dscr_base=dl[0] if dl else 0

    def _returns(self, res):
        res.exit_ebitda=res.ebitda_proj[-1] if res.ebitda_proj else self.c.ebitda
        res.exit_ev=res.exit_ebitda*self.a.exit_multiple
        exit_debt=res.debt_schedule["Closing"].iloc[-1]
        res.exit_equity=max(0,res.exit_ev-exit_debt+self.c.cash)
        res.moic=res.exit_equity/res.entry_equity if res.entry_equity>0 else 0
        cfs=[-res.entry_equity]+[0]*(self.a.holding_period-1)+[res.exit_equity]
        irr=_irr(cfs); res.irr=irr if not np.isnan(irr) else 0.0

    def _downside(self, res):
        """Stress test: Revenue -10%, EBITDA -15%, Exit Multiple -1x, CAGR -3pp.
        Uses fast_run() to avoid infinite recursion and extra compute."""
        cs = CompanyInputs(
            revenue=self.c.revenue*.90, ebitda=self.c.ebitda*.85,
            ebit=self.c.ebit*.80, depreciation=self.c.depreciation,
            interest_expense=self.c.interest_expense, tax_rate=self.c.tax_rate,
            total_debt=self.c.total_debt, cash=self.c.cash,
            net_working_capital=self.c.net_working_capital, capex=self.c.capex)
        as_ = LBOAssumptions(
            entry_ev_multiple=self.a.entry_ev_multiple,
            equity_contribution_pct=self.a.equity_contribution_pct,
            senior_debt_rate=self.a.senior_debt_rate,
            debt_amortization_years=self.a.debt_amortization_years,
            exit_multiple=self.a.exit_multiple - 1.0,
            holding_period=self.a.holding_period,
            revenue_cagr=max(0, self.a.revenue_cagr - 0.03),
            ebitda_margin_entry=self.a.ebitda_margin_entry - 0.02,
            ebitda_margin_exit=self.a.ebitda_margin_exit - 0.02)
        try:
            sr = LBOEngine(cs, as_).fast_run()
            res.downside_irr = sr.irr; res.downside_moic = sr.moic
        except Exception:
            res.downside_irr = 0.0; res.downside_moic = 0.0

    def _cash_conversion(self, res):
        """[McK] FCF Y1 / Normalized EBITDA – key PE screening metric"""
        if self.c.ebitda>0 and res.fcf_series:
            res.cash_conversion=res.fcf_series[0]/self.c.ebitda
        else: res.cash_conversion=0.0

    def _value_bridge(self, res):
        """[R&P p.162] Decompose IRR into EBITDA growth, multiple expansion, debt paydown"""
        if res.entry_equity <= 0: return
        hp = self.a.holding_period
        exit_debt = res.debt_schedule["Closing"].iloc[-1] if not res.debt_schedule.empty else res.entry_debt

        # 1) Pure EBITDA growth with same entry/exit multiple, same leverage
        equity_ebitda_only = max(0, (res.exit_ebitda - self.c.ebitda) * self.a.entry_ev_multiple
                                 * (1-self.a.equity_contribution_pct))
        # 2) Multiple expansion: exit @ exit_multiple vs entry multiple on same EBITDA
        equity_mult_exp = max(0, (self.a.exit_multiple - self.a.entry_ev_multiple)
                              * res.exit_ebitda)
        # 3) Debt paydown equity value created
        equity_debt_paydown = max(0, res.entry_debt - exit_debt)

        total = equity_ebitda_only + equity_mult_exp + equity_debt_paydown
        if total > 0:
            res.vc_ebitda_growth  = equity_ebitda_only / total
            res.vc_multiple_exp   = equity_mult_exp / total
            res.vc_debt_paydown   = equity_debt_paydown / total
        else:
            res.vc_ebitda_growth = res.vc_multiple_exp = res.vc_debt_paydown = 1/3

    def _debt_capacity_dscr(self, res):
        """[MPE] Realistic debt capacity = min(max_leverage × EBITDA, DSCR-constrained max debt)
        Binary search for max debt where DSCR Y1 >= 1.30"""
        lo, hi = 0.0, self.c.ebitda * 8.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if mid <= 0:
                break
            amort = mid / self.a.debt_amortization_years
            int_  = mid * self.a.senior_debt_rate
            fcf1  = res.fcf_series[0] if res.fcf_series else self.c.ebitda * 0.5
            ds    = int_ + amort
            dscr  = fcf1 / ds if ds > 0 else 0
            if dscr >= 1.30:
                lo = mid
            else:
                hi = mid
        res.debt_capacity_dscr = min(lo, res.debt_capacity)

    def _lbo_score(self, res):
        """LBO Attractiveness Score 0–100 [composite]"""
        def score_metric(val, thresholds, max_pts):
            # thresholds: [(value, pct_0_to_1), ...] ascending → interpolate → * max_pts
            if val <= thresholds[0][0]: return max_pts * thresholds[0][1]
            if val >= thresholds[-1][0]: return max_pts * thresholds[-1][1]
            for i in range(len(thresholds)-1):
                v0,s0 = thresholds[i]; v1,s1 = thresholds[i+1]
                if v0 <= val <= v1:
                    t = (val-v0)/(v1-v0)
                    return max_pts * (s0 + t*(s1-s0))
            return 0.0

        # IRR score (max 30 pts)
        s_irr  = score_metric(res.irr*100, [(0,0),(15,.20),(20,.50),(25,.80),(35,1.0)], 30)
        # DSCR score (max 20 pts)
        s_dscr = score_metric(res.dscr_base, [(0,0),(1.0,.10),(1.2,.40),(1.5,.80),(2.5,1.0)], 20)
        # Cash Conversion score (max 15 pts)
        s_cc   = score_metric(res.cash_conversion*100, [(0,0),(30,.10),(50,.50),(70,.80),(90,1.0)], 15)
        # Revenue Stability score (max 15 pts) – lower vol = better
        rev_vol = self.hist.revenue_volatility if self.hist else 0.15
        s_rev  = score_metric(100 - rev_vol*100, [(0,0),(60,.10),(80,.60),(90,.85),(100,1.0)], 15)
        # Leverage score (max 10 pts) – lower leverage = better
        s_lev  = score_metric(10 - res.entry_leverage, [(0,0),(3,.10),(5,.60),(7,.90),(10,1.0)], 10)
        # Margin score (max 10 pts)
        margin = self.c.ebitda / max(self.c.revenue, 1) * 100
        s_marg = score_metric(margin, [(0,0),(8,.10),(15,.50),(20,.80),(30,1.0)], 10)

        total = s_irr + s_dscr + s_cc + s_rev + s_lev + s_marg
        res.lbo_score = round(min(100, max(0, total)), 1)
        res.lbo_score_breakdown = {
            "IRR":          round(s_irr,  1),
            "DSCR":         round(s_dscr, 1),
            "Cash Conv.":   round(s_cc,   1),
            "Rev. Stability": round(s_rev, 1),
            "Leverage":     round(s_lev,  1),
            "Margins":      round(s_marg, 1),
        }

    def _flags(self, res):
        flags = []
        c=self.c; a=self.a
        if res.entry_leverage>6.0: flags.append(f"🔴 Entry Leverage {res.entry_leverage:.1f}x > 6.0x")
        if res.dscr_base<1.20: flags.append(f"🔴 DSCR Y1 {res.dscr_base:.2f}x < 1.20x Covenant Floor")
        if res.irr<0.20: flags.append(f"🔴 IRR {res.irr:.1%} < 20% Hurdle")
        if res.moic<2.0: flags.append(f"⚠️ MOIC {res.moic:.1f}x < 2.0x")
        if res.covenant_breach_year: flags.append(f"🔴 Covenant Breach Year {res.covenant_breach_year}")
        if c.ebitda/max(c.revenue,1)<0.08: flags.append(f"⚠️ EBITDA Margin {c.ebitda/c.revenue:.1%} < 8%")
        if res.fcf_yield<0.04: flags.append(f"⚠️ FCF Yield {res.fcf_yield:.1%} < 4%")
        if res.cash_conversion<0.40: flags.append(f"⚠️ Cash Conversion {res.cash_conversion:.0%} < 40% — limited FCF quality")
        if self.hist and self.hist.revenue_volatility>0.20:
            flags.append(f"⚠️ Revenue Volatility {self.hist.revenue_volatility:.0%} > 20% — cyclical profile")
        res.red_flags=flags
        res.is_lbo_viable=(res.irr>=0.20 and res.moic>=2.0 and res.dscr_base>=1.10 and res.entry_leverage<=7.0)


# ── Sensitivity Engine ────────────────────────────────────────────────────────

class SensitivityEngine:
    def __init__(self, inputs: CompanyInputs, base: LBOAssumptions):
        self.inputs=inputs; self.base=base

    def _run(self, **kwargs) -> LBOResults:
        """Fast run for sensitivity analysis — no downside/score/bridge."""
        a = LBOAssumptions(
            entry_ev_multiple=kwargs.get("entry_ev_multiple",self.base.entry_ev_multiple),
            equity_contribution_pct=kwargs.get("equity_contribution_pct",self.base.equity_contribution_pct),
            senior_debt_rate=kwargs.get("senior_debt_rate",self.base.senior_debt_rate),
            debt_amortization_years=kwargs.get("debt_amortization_years",self.base.debt_amortization_years),
            exit_multiple=kwargs.get("exit_multiple",self.base.exit_multiple),
            holding_period=kwargs.get("holding_period",self.base.holding_period),
            revenue_cagr=kwargs.get("revenue_cagr",self.base.revenue_cagr),
            ebitda_margin_entry=kwargs.get("ebitda_margin_entry",self.base.ebitda_margin_entry),
            ebitda_margin_exit=kwargs.get("ebitda_margin_exit",self.base.ebitda_margin_exit),
        )
        return LBOEngine(self.inputs, a).fast_run()

    def irr_heatmap(self, exit_multiples, revenue_cagrs):
        rows={}
        for em in exit_multiples:
            row={}
            for cagr in revenue_cagrs:
                r=self._run(exit_multiple=em,revenue_cagr=cagr)
                row[f"{cagr:.0%}"]=round(r.irr*100,1)
            rows[f"{em:.1f}x"]=row
        return pd.DataFrame(rows).T

    def dscr_heatmap(self, interest_rates, leverage_multiples):
        rows={}
        for lev in leverage_multiples:
            row={}
            for rate in interest_rates:
                eq=max(0.20,min(0.80,1-(lev/self.base.entry_ev_multiple)))
                r=self._run(senior_debt_rate=rate,equity_contribution_pct=eq)
                row[f"{rate:.1%}"]=round(r.dscr_base,2)
            rows[f"{lev:.1f}x"]=row
        return pd.DataFrame(rows).T

    def leverage_irr_heatmap(self, exit_multiples, equity_pcts):
        rows={}
        for em in exit_multiples:
            row={}
            for eq in equity_pcts:
                r=self._run(exit_multiple=em,equity_contribution_pct=eq)
                row[f"{eq:.0%} Equity"]=round(r.irr*100,1)
            rows[f"{em:.1f}x"]=row
        return pd.DataFrame(rows).T