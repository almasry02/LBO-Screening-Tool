"""
LBO Screening Tool v4.1
[R&P] Rosenbaum & Pearl - Investment Banking (2020)
[MPE] Mastering Private Equity / Private Equity at Work
[McK] McKinsey - Valuation (7th ed.)
Performance: @st.cache_data on all expensive computations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib; matplotlib.use("Agg")
from io import BytesIO
import sys, os, time

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from finance_engine import (
    CompanyInputs, LBOAssumptions, LBOEngine,
    SensitivityEngine, HistoricalAnalyzer, HistoricalYear,
)
from data_parser import (
    parse_file, REQUIRED_FIELDS, detect_currency_and_unit,
)

def _detect_ccy(unit_string):
    """Detect currency using current session language."""
    lang = st.session_state.get("lang", "en")
    return detect_currency_and_unit(unit_string, lang=lang)

# ══════════════════════════════════════════════════════
# CACHED ENGINE FUNCTIONS
# All heavy math is wrapped in @st.cache_data so a slider
# change only re-runs what actually changed.
# ══════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _build_demo_data():
    """Build demo historical data ONCE — cached forever."""
    demo_hist = [
        HistoricalYear(2015, 543286, 161502, 136379, 25122,  280, 105220,  72899,  64060, 170245, 19281, 0.245),
        HistoricalYear(2016, 751544, 263878, 236245, 27633,   23, 170915, 132090, 227575, 165635, 20725, 0.285),
        HistoricalYear(2017, 632580, 135677,  98116, 37561,   18,  81491, 107903,  63390, 265996, 28171, 0.196),
        HistoricalYear(2018, 564849, 117409,  73612, 43797,    0,  58721,      0,  42653, 193273, 32848, 0.226),
        HistoricalYear(2019, 543253, 112684,  68983, 43701,    5,  55592,      0, 117237, 169831, 32776, 0.213),
        HistoricalYear(2020, 929841, 304740, 256569, 48171,  108, 204395,      0, 319539, 102175, 36128, 0.221),
        HistoricalYear(2021,1061185, 343801, 295999, 47803,  688, 223466,      0, 500572, 138197, 35852, 0.249),
        HistoricalYear(2022, 877247, 254170, 204710, 49461, 1462, 155980,      0, 578863, 170566, 37096, 0.244),
        HistoricalYear(2023, 699239,  99878,  49106, 50773, 1688,  43885,      0, 352817, 265360, 38080, 0.248),
        HistoricalYear(2024, 730376, 173445, 127093, 46352, 4341, 106785,      0, 406014, 251406, 34764, 0.227),
    ]
    hm  = HistoricalAnalyzer(demo_hist).compute()
    lat = demo_hist[-1]
    ci  = CompanyInputs(
        revenue=hm.normalized_revenue, ebitda=hm.normalized_ebitda,
        ebit=hm.normalized_ebitda - lat.depreciation, depreciation=lat.depreciation,
        interest_expense=lat.interest_expense, tax_rate=lat.tax_rate,
        total_debt=lat.total_debt, cash=lat.cash,
        net_working_capital=lat.net_working_capital, capex=hm.normalized_capex,
        company_name="Precision Industries Corp.", currency_display="tsd USD",
        revenue_cagr_hist=hm.revenue_cagr, ebitda_margin_avg=hm.ebitda_margin_avg,
        capex_intensity=hm.capex_intensity_avg, nwc_intensity=hm.nwc_intensity_avg,
    )
    ts = pd.DataFrame([{
        "Revenue": y.revenue, "EBITDA": y.ebitda, "EBIT": y.ebit,
        "D&A": y.depreciation, "Interest": y.interest_expense,
        "Net Income": y.net_income, "Cash": y.cash,
        "Total Debt": y.total_debt, "NWC": y.net_working_capital,
        "CapEx": y.capex, "EBITDA Margin": y.ebitda_margin,
    } for y in demo_hist], index=[y.year for y in demo_hist]).sort_index()
    years_used = [y.year for y in demo_hist]
    return ci, hm, ts, years_used


@st.cache_data(show_spinner=False)
def _run_lbo(
    # CompanyInputs fields (all hashable)
    revenue, ebitda, ebit, depreciation, interest_expense, tax_rate,
    total_debt, cash, net_working_capital, capex,
    company_name, currency_display, revenue_cagr_hist, ebitda_margin_avg,
    capex_intensity, nwc_intensity,
    # hist_metrics (passed as tuple for hashability)
    hm_tuple,
    # LBOAssumptions
    entry_ev_multiple, equity_contribution_pct, senior_debt_rate,
    debt_amortization_years, exit_multiple, holding_period,
    revenue_cagr, ebitda_margin_entry, ebitda_margin_exit,
    max_leverage_covenant, min_dscr_covenant,
):
    ci = CompanyInputs(
        revenue=revenue, ebitda=ebitda, ebit=ebit, depreciation=depreciation,
        interest_expense=interest_expense, tax_rate=tax_rate,
        total_debt=total_debt, cash=cash, net_working_capital=net_working_capital,
        capex=capex, company_name=company_name, currency_display=currency_display,
        revenue_cagr_hist=revenue_cagr_hist, ebitda_margin_avg=ebitda_margin_avg,
        capex_intensity=capex_intensity, nwc_intensity=nwc_intensity,
    )
    hm = _unpack_hm(hm_tuple)
    a  = LBOAssumptions(
        entry_ev_multiple=entry_ev_multiple,
        equity_contribution_pct=equity_contribution_pct,
        senior_debt_rate=senior_debt_rate,
        debt_amortization_years=debt_amortization_years,
        exit_multiple=exit_multiple,
        holding_period=holding_period,
        revenue_cagr=revenue_cagr,
        ebitda_margin_entry=ebitda_margin_entry,
        ebitda_margin_exit=ebitda_margin_exit,
        max_leverage_covenant=max_leverage_covenant,
        min_dscr_covenant=min_dscr_covenant,
    )
    return LBOEngine(ci, a, hm).run()


@st.cache_data(show_spinner=False)
def _run_sensitivities(
    revenue, ebitda, ebit, depreciation, interest_expense, tax_rate,
    total_debt, cash, net_working_capital, capex,
    company_name, currency_display, revenue_cagr_hist, ebitda_margin_avg,
    capex_intensity, nwc_intensity,
    entry_ev_multiple, equity_contribution_pct, senior_debt_rate,
    debt_amortization_years, exit_multiple, holding_period,
    revenue_cagr, ebitda_margin_entry, ebitda_margin_exit,
    max_leverage_covenant, min_dscr_covenant,
):
    ci = CompanyInputs(
        revenue=revenue, ebitda=ebitda, ebit=ebit, depreciation=depreciation,
        interest_expense=interest_expense, tax_rate=tax_rate,
        total_debt=total_debt, cash=cash, net_working_capital=net_working_capital,
        capex=capex, company_name=company_name, currency_display=currency_display,
        revenue_cagr_hist=revenue_cagr_hist, ebitda_margin_avg=ebitda_margin_avg,
        capex_intensity=capex_intensity, nwc_intensity=nwc_intensity,
    )
    a = LBOAssumptions(
        entry_ev_multiple=entry_ev_multiple,
        equity_contribution_pct=equity_contribution_pct,
        senior_debt_rate=senior_debt_rate,
        debt_amortization_years=debt_amortization_years,
        exit_multiple=exit_multiple,
        holding_period=holding_period,
        revenue_cagr=revenue_cagr,
        ebitda_margin_entry=ebitda_margin_entry,
        ebitda_margin_exit=ebitda_margin_exit,
        max_leverage_covenant=max_leverage_covenant,
        min_dscr_covenant=min_dscr_covenant,
    )
    sens    = SensitivityEngine(ci, a)
    irr_hm  = sens.irr_heatmap(
        [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0],
        [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
    )
    dscr_hm = sens.dscr_heatmap(
        [0.04, 0.055, 0.065, 0.08, 0.095, 0.11],
        [3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0],
    )
    lev_hm  = sens.leverage_irr_heatmap(
        [5.0, 6.0, 7.0, 8.0, 9.0],
        [0.25, 0.35, 0.40, 0.50, 0.60],
    )
    return irr_hm, dscr_hm, lev_hm


def _pack_hm(hm):
    """Serialize HistoricalMetrics to a hashable tuple."""
    if hm is None:
        return None
    return (
        tuple(hm.years_used), hm.revenue_cagr, hm.ebitda_margin_avg,
        hm.ebitda_margin_med, hm.ebitda_volatility, hm.capex_intensity_avg,
        hm.nwc_intensity_avg, hm.fcf_conversion_avg, hm.interest_coverage_avg,
        tuple(hm.revenue_series), tuple(hm.ebitda_series), tuple(hm.margin_series),
        hm.revenue_volatility, hm.normalized_revenue, hm.normalized_ebitda,
        hm.normalized_capex, hm.normalized_nwc_delta,
    )


def _unpack_hm(t):
    """Deserialize tuple back to HistoricalMetrics."""
    if t is None:
        return None
    from finance_engine import HistoricalMetrics
    return HistoricalMetrics(
        years_used=list(t[0]), revenue_cagr=t[1], ebitda_margin_avg=t[2],
        ebitda_margin_med=t[3], ebitda_volatility=t[4], capex_intensity_avg=t[5],
        nwc_intensity_avg=t[6], fcf_conversion_avg=t[7], interest_coverage_avg=t[8],
        revenue_series=list(t[9]), ebitda_series=list(t[10]), margin_series=list(t[11]),
        revenue_volatility=t[12], normalized_revenue=t[13], normalized_ebitda=t[14],
        normalized_capex=t[15], normalized_nwc_delta=t[16],
    )


def _ci_args(ci):
    """Flatten CompanyInputs into kwargs for cached functions."""
    return dict(
        revenue=ci.revenue, ebitda=ci.ebitda, ebit=ci.ebit,
        depreciation=ci.depreciation, interest_expense=ci.interest_expense,
        tax_rate=ci.tax_rate, total_debt=ci.total_debt, cash=ci.cash,
        net_working_capital=ci.net_working_capital, capex=ci.capex,
        company_name=ci.company_name, currency_display=ci.currency_display,
        revenue_cagr_hist=ci.revenue_cagr_hist, ebitda_margin_avg=ci.ebitda_margin_avg,
        capex_intensity=ci.capex_intensity, nwc_intensity=ci.nwc_intensity,
    )


def _a_args(a):
    """Flatten LBOAssumptions into kwargs for cached functions."""
    return dict(
        entry_ev_multiple=a.entry_ev_multiple,
        equity_contribution_pct=a.equity_contribution_pct,
        senior_debt_rate=a.senior_debt_rate,
        debt_amortization_years=a.debt_amortization_years,
        exit_multiple=a.exit_multiple,
        holding_period=a.holding_period,
        revenue_cagr=a.revenue_cagr,
        ebitda_margin_entry=a.ebitda_margin_entry,
        ebitda_margin_exit=a.ebitda_margin_exit,
        max_leverage_covenant=a.max_leverage_covenant,
        min_dscr_covenant=a.min_dscr_covenant,
    )


# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="LBO Screening Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────
# metric-box: transparent background so no black bar in dark/light mode
st.markdown("""
<style>
.metric-box {
    background:transparent;
    border-radius:8px; padding:10px 14px;
    border-left:4px solid #4f8ef7; margin-bottom:8px;
}
.info-card {
    border-left:4px solid #4f8ef7;
    padding:10px 14px; border-radius:6px;
    margin-bottom:8px; font-size:.9em;
}
.red-flag   {background:rgba(255,75,75,0.12);border-left:4px solid #ff4b4b;border-radius:6px;padding:10px 14px;margin:6px 0;font-size:0.9em;}
.warn-flag  {background:rgba(255,170,0,0.10);border-left:4px solid #ffaa00;border-radius:6px;padding:10px 14px;margin:3px 0;font-size:.9em;}
.green-flag {background:rgba(0,204,136,0.10);border-left:4px solid #00cc88;border-radius:6px;padding:10px 14px;margin:3px 0;font-size:.9em;}
.currency-badge {
    display:inline-block;border:1px solid #4f8ef7;
    border-radius:20px;padding:3px 12px;font-size:.85em;color:#4f8ef7;
    font-weight:600;margin-left:10px;
}
.section-hdr {
    font-size:1.05em;font-weight:600;color:#4f8ef7;
    margin:18px 0 6px 0;padding-bottom:4px;border-bottom:1px solid #2a2f3e;
}
.src-tag {font-size:.72em;color:#6e7681;font-style:italic;margin-bottom:8px;}
</style>
""", unsafe_allow_html=True)

# ── DEFAULTS ───────────────────────────────────────────
DEFAULT_T = {
    "min_irr": .20, "warn_irr": .25, "min_moic": 2.0, "warn_moic": 2.5,
    "max_entry_leverage": 6.0, "warn_entry_leverage": 5.0,
    "min_dscr": 1.20, "warn_dscr": 1.50,
    "min_ebitda_margin": .10, "min_fcf_yield": .04,
    "max_debt_ebitda": 5.5, "min_interest_coverage": 2.0,
}
DEFAULT_F = {
    "debt_amort_years": 7, "equity_pct_default": .40,
    "downside_revenue_stress": .10, "downside_margin_stress": .020,
    "downside_exit_stress": 1.0,
}

# ── LANGUAGE STRINGS ───────────────────────────────────
LANG = {
    "en": {
        "nav_main":"LBO Analysis","nav_settings":"Settings",
        "data_input":"Data Input","deal_params":"Deal Parameters",
        "upload_btn":"Run LBO Analysis",
        "demo_info":"Demo: Anonymized mid-market manufacturing company | tsd USD | 10-year normalized basis [R&P]",
        "lang_label":"Language","tab_hist":"Historical Analytics",
        "tab_lbo":"LBO Structure","tab_debt":"Debt Schedule",
        "tab_sens":"Sensitivities","tab_score":"LBO Score",
        "tab_bridge":"Value Bridge","tab_screen":"Deal Screener",
        "tab_flags":"Risk & Flags","viable":"LBO VIABLE","critical":"CRITICAL",
        "no_flags":"No red flags - structure appears feasible","sources":"Sources:",
        "hist_basis":"Historical basis","yrs":"yrs","hist_cagr":"Hist. CAGR",
        "avg_margin":"Avg EBITDA Margin","fwd_cagr":"Fwd CAGR",
        "normalized":"normalized","revenue_cagr":"Revenue CAGR",
        "avg_ebitda":"Avg EBITDA Margin","ebitda_vol":"EBITDA Volatility",
        "fcf_conv":"FCF Conversion","capex_int":"CapEx Intensity",
        "nwc_int":"NWC Intensity","ic_avg":"Avg Interest Coverage",
        "norm_ebitda":"Norm. EBITDA","rev_ebitda_chart":"Revenue & EBITDA Trend",
        "margin_trend":"Margin Trend & Volatility",
        "norm_basis":"Normalized LBO Basis ",
        "metric":"Metric","value":"Value","method":"Method",
        "norm_rev_row":"Revenue (normalized)","norm_ebi_row":"EBITDA (normalized)",
        "avg_mar_row":"EBITDA Margin (avg)","norm_cap_row":"CapEx (normalized)",
        "nwc_del_row":"Delta NWC p.a.","curr_rev":"Current revenue",
        "avg_m_curr":"Avg margin x current revenue","simple_avg":"Simple avg",
        "avg_cap_int":"Avg CapEx intensity x revenue",
        "nwc_int_delt":"NWC intensity x rev. growth",
        "entry_struct":"Entry Structure","returns":"Returns",
        "input_sum":"Normalized Entry P&L","fcf_proj":"FCF Projection",
        "dscr_dev":"DSCR Development",
        "debt_sched":"Debt Schedule (straight-line + cash sweep)",
        "debt_wfall":"Debt Paydown (Waterfall)","headroom":"Headroom",
        "sens_title":"Sensitivity Analyses",
        "irr_hm":"IRR vs Exit Multiple x Rev. CAGR",
        "dscr_hm":"DSCR vs Leverage x Interest Rate",
        "lev_hm":"IRR vs Exit Multiple x Equity Contribution",
        "ds_title":"Downside Case",
        "ds_cap":"Stress: Revenue -10% | EBITDA Margin -200bps | Exit Multiple -1.0x",
        "irr_buf":"IRR Buffer","dscr_buf":"DSCR Buffer",
        "floor":"Floor","warn":"Warn","hurdle":"Hurdle",
        "used_years":"Years used","sel_years":"Years for analysis",
        "y3":"3 years (latest)","y5":"5 years","all":"All",
        "hist_only":"Historical data available after Moodys upload or in Demo mode",
        "excel_exp":"Excel Export","entry_ev":"Entry EV","entry_eq":"Entry Equity",
        "entry_debt":"Entry Debt","net_lev":"Net Leverage","debt_cap":"Debt Capacity",
        "exit_ebitda":"Exit EBITDA","exit_ev":"Exit EV","exit_eq":"Exit Equity",
        "base_irr":"Base IRR","base_moic":"Base MOIC","ds_irr":"Downside IRR",
        "ds_moic":"Downside MOIC","revenue":"Revenue","ebitda":"EBITDA","ebit":"EBIT",
        "da":"D&A","interest":"Interest Expense","tax":"Tax Rate","debt":"Financial Debt",
        "cash":"Cash","nwc":"NWC","capex":"CapEx",
        "score_title":"LBO Attractiveness Score",
        "score_sub":"Composite 0-100 · 6 weighted factors",
        "cc_title":"Cash Conversion Analysis",
        "cc_label":"Cash Conversion (FCF Y1 / EBITDA)",
        "cc_excellent":"Excellent (>70%)","cc_normal":"Normal (50-70%)",
        "cc_critical":"Critical (30-50%)","cc_flag":"Red Flag (<30%)",
        "rev_qual_title":"Revenue Quality",
        "rev_vol_label":"Revenue Volatility (StdDev/Mean)",
        "rev_stable":"Stable (<10%)","rev_moderate":"Moderate (10-20%)",
        "rev_cyclical":"Cyclical (>20%)","debt_real_title":"Realistic Debt Capacity",
        "dc_simple":"Simple (leverage x EBITDA)","dc_dscr":"DSCR-Constrained (1.30x floor)",
        "dc_effective":"Effective (min of both)","overpay_title":"Entry Multiple vs. Industry Benchmarks",
        "bridge_title":"Value Creation Bridge",
        "bridge_sub":"IRR decomposed into 3 value drivers",
        "bridge_ebitda":"EBITDA Growth","bridge_mult":"Multiple Expansion",
        "bridge_debt":"Debt Paydown","bridge_interp":"Driver Interpretation",
        "screen_title":"Comparable Deal Screener",
        "screen_sub":"Add multiple targets and rank by LBO attractiveness",
        "screen_add":"Add Company to Screener","screen_run":"Run Screening",
        "screen_clear":"Clear All","screen_name":"Company Name",
        "screen_rev":"Revenue","screen_ebitda":"EBITDA","screen_debt":"Net Debt",
        "screen_entry":"Entry Multiple","screen_exit":"Exit Multiple",
        "screen_hold":"Hold (yrs)","screen_rate":"Rate (%)","screen_cagr":"CAGR (%)",
        "screen_eq":"Equity (%)","screen_results":"Screening Results",
        "screen_rank":"Rank","screen_empty":"Add at least 1 company to run the screener",
        "thesis_title":"Investment Thesis Generator","thesis_industry":"Industry",
        "thesis_run":"Generate Theses","thesis_loading":"Generating investment theses...",
    },
    "de": {
        "nav_main":"LBO Analyse","nav_settings":"Einstellungen",
        "data_input":"Dateneingabe","deal_params":"Deal-Parameter",
        "upload_btn":"LBO-Analyse starten",
        "demo_info":"Demo: Anonymisiertes Mittelstands-Unternehmen | tsd USD | 10 Jahre normalisierte Basis [R&P]",
        "lang_label":"Sprache","tab_hist":"Historische Analyse",
        "tab_lbo":"LBO-Struktur","tab_debt":"Debt Schedule",
        "tab_sens":"Sensitivitaeten","tab_score":"LBO Score",
        "tab_bridge":"Value Bridge","tab_screen":"Deal Screener",
        "tab_flags":"Risk & Flags","viable":"LBO-FAEHIG","critical":"KRITISCH",
        "no_flags":"Keine Red Flags - Struktur erscheint tragfaehig","sources":"Quellen:",
        "hist_basis":"Historische Basis","yrs":"J.","hist_cagr":"Hist. CAGR",
        "avg_margin":"Avg EBITDA-Marge","fwd_cagr":"Fwd CAGR",
        "normalized":"normalisiert [R&P]","revenue_cagr":"Umsatz-CAGR",
        "avg_ebitda":"Avg EBITDA-Marge","ebitda_vol":"EBITDA-Volatilitaet",
        "fcf_conv":"FCF Conversion","capex_int":"CapEx Intensitaet",
        "nwc_int":"NWC Intensitaet","ic_avg":"Avg Interest Coverage",
        "norm_ebitda":"Norm. EBITDA","rev_ebitda_chart":"Umsatz & EBITDA-Entwicklung",
        "margin_trend":"Margin-Trend & Volatilitaet [MPE]",
        "norm_basis":"Normalisierte LBO-Basis [R&P]",
        "metric":"Kennzahl","value":"Wert","method":"Methode",
        "norm_rev_row":"Umsatz (normalisiert)","norm_ebi_row":"EBITDA (normalisiert)",
        "avg_mar_row":"EBITDA-Marge (Avg)","norm_cap_row":"CapEx (normalisiert)",
        "nwc_del_row":"Delta NWC p.a.","curr_rev":"Aktueller Umsatz",
        "avg_m_curr":"Avg-Marge x akt. Umsatz","simple_avg":"Einfacher Avg",
        "avg_cap_int":"Avg CapEx-Intensitaet x Umsatz",
        "nwc_int_delt":"NWC-Intensitaet x Rev.-Delta",
        "entry_struct":"Entry-Struktur [R&P p.152]","returns":"Returns",
        "input_sum":"Normalisierte Entry P&L","fcf_proj":"FCF-Projektion [McK p.167]",
        "dscr_dev":"DSCR-Entwicklung [MPE]",
        "debt_sched":"Tilgungsplan (Straight-line + Cash Sweep) [R&P p.156]",
        "debt_wfall":"Schuldenabbau (Waterfall)","headroom":"Headroom",
        "sens_title":"Sensitivitaetsanalysen [R&P p.200]",
        "irr_hm":"IRR vs. Exit Multiple x Rev. CAGR",
        "dscr_hm":"DSCR vs. Leverage x Zinssatz",
        "lev_hm":"IRR vs. Exit Multiple x Equity-Anteil",
        "ds_title":"Downside Case [MPE]",
        "ds_cap":"Stress: Revenue -10% | EBITDA-Marge -200bps | Exit Multiple -1,0x",
        "irr_buf":"IRR Puffer","dscr_buf":"DSCR Puffer",
        "floor":"Floor","warn":"Warn","hurdle":"Hurdle",
        "used_years":"Verwendete Jahre","sel_years":"Jahre fuer Analyse",
        "y3":"3 Jahre (aktuellste)","y5":"5 Jahre","all":"Alle",
        "hist_only":"Historische Daten nach Moodys Upload oder im Demo-Modus verfuegbar",
        "excel_exp":"Excel Export","entry_ev":"Entry EV","entry_eq":"Entry Equity",
        "entry_debt":"Entry Debt","net_lev":"Net Leverage","debt_cap":"Debt Capacity",
        "exit_ebitda":"Exit EBITDA","exit_ev":"Exit EV","exit_eq":"Exit Equity",
        "base_irr":"Base IRR","base_moic":"Base MOIC","ds_irr":"Downside IRR",
        "ds_moic":"Downside MOIC","revenue":"Umsatz","ebitda":"EBITDA","ebit":"EBIT",
        "da":"D&A","interest":"Zinsaufwand","tax":"Steuersatz","debt":"Finanzschulden",
        "cash":"Cash","nwc":"NWC","capex":"CapEx",
        "score_title":"LBO Attraktivitaets-Score",
        "score_sub":"Composite 0-100 · 6 gewichtete Faktoren",
        "cc_title":"Cash Conversion Analyse [McK]",
        "cc_label":"Cash Conversion (FCF J1 / EBITDA)",
        "cc_excellent":"Sehr gut (>70%)","cc_normal":"Normal (50-70%)",
        "cc_critical":"Kritisch (30-50%)","cc_flag":"Red Flag (<30%)",
        "rev_qual_title":"Revenue-Qualitaet [MPE]",
        "rev_vol_label":"Revenue-Volatilitaet (StdAbw/Mittel)",
        "rev_stable":"Stabil (<10%)","rev_moderate":"Moderat (10-20%)",
        "rev_cyclical":"Zyklisch (>20%)","debt_real_title":"Realistische Debt Capacity [MPE]",
        "dc_simple":"Einfach (Leverage x EBITDA)","dc_dscr":"DSCR-begrenzt (Floor 1,30x)",
        "dc_effective":"Effektiv (Minimum beider)","overpay_title":"Entry Multiple vs. Branchen-Benchmark [R&P]",
        "bridge_title":"Value Creation Bridge [R&P p.162]",
        "bridge_sub":"IRR aufgeteilt in 3 Wertreiber",
        "bridge_ebitda":"EBITDA-Wachstum","bridge_mult":"Multiple Expansion",
        "bridge_debt":"Schuldenabbau","bridge_interp":"Treiber-Interpretation",
        "screen_title":"Vergleichbarer Deal Screener",
        "screen_sub":"Mehrere Targets hinzufuegen und nach LBO-Attraktivitaet ranken",
        "screen_add":"Unternehmen hinzufuegen","screen_run":"Screening starten",
        "screen_clear":"Alle loeschen","screen_name":"Unternehmensname",
        "screen_rev":"Umsatz","screen_ebitda":"EBITDA","screen_debt":"Nettoverschuldung",
        "screen_entry":"Entry Multiple","screen_exit":"Exit Multiple",
        "screen_hold":"Haltedauer (J.)","screen_rate":"Zinssatz (%)","screen_cagr":"CAGR (%)",
        "screen_eq":"Eigenkapital (%)","screen_results":"Screening-Ergebnisse",
        "screen_rank":"Rang","screen_empty":"Mindestens 1 Unternehmen hinzufuegen",
        "thesis_title":"Investment-Thesen Generator","thesis_industry":"Branche",
        "thesis_run":"Thesen generieren","thesis_loading":"Thesen werden generiert...",
    }
}


def fmt_num(val, lang, dec=0, sfx=""):
    s = f"{val:,.{dec}f}"
    if lang == "de":
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s + sfx


# ── SESSION STATE INIT ─────────────────────────────────
_ss_defaults = {
    "T": DEFAULT_T.copy(), "F": DEFAULT_F.copy(),
    "ccy": detect_currency_and_unit("EUR", lang="en"), "lang": "en",
    "screener_companies": [], "screener_results": [],
    "thesis_text": "", "uploaded_data": None,
}
for k, v in _ss_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v.copy() if isinstance(v, dict) else v

T    = st.session_state.T
F    = st.session_state.F
lang = st.session_state.lang
L    = LANG[lang]

# ── NAVIGATION ─────────────────────────────────────────
page = st.sidebar.radio(
    "nav", ["📊 " + L["nav_main"], "⚙️ " + L["nav_settings"]],
    label_visibility="collapsed"
)
in_settings = L["nav_settings"] in page

# ══════════════════════════════════════════════════════
# SETTINGS PAGE
# ══════════════════════════════════════════════════════
if in_settings:
    lc = st.sidebar.selectbox(L["lang_label"], ["English","Deutsch"],
        index=0 if lang=="en" else 1, key="lang_set")
    nl = "en" if lc=="English" else "de"
    if nl != lang:
        st.session_state.lang = nl; st.rerun()

    st.title("⚙️ " + L["nav_settings"])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Returns `[R&P]`")
        T["min_irr"]   = st.number_input("Min IRR - Red Flag (%)",  value=T["min_irr"]*100,  step=0.5)/100
        T["warn_irr"]  = st.number_input("IRR - Warning (%)",        value=T["warn_irr"]*100, step=0.5)/100
        T["min_moic"]  = st.number_input("Min MOIC - Red Flag (x)",  value=T["min_moic"],     step=0.1)
        T["warn_moic"] = st.number_input("MOIC - Warning (x)",       value=T["warn_moic"],    step=0.1)
        st.markdown("### Margins & FCF `[McK]`")
        T["min_ebitda_margin"] = st.number_input("Min EBITDA Margin (%)", value=T["min_ebitda_margin"]*100, step=0.5)/100
        T["min_fcf_yield"]     = st.number_input("Min FCF Yield (%)",     value=T["min_fcf_yield"]*100,     step=0.5)/100
    with c2:
        st.markdown("### Leverage & DSCR `[MPE]`")
        T["max_entry_leverage"]  = st.number_input("Max Leverage - Red Flag (x)", value=T["max_entry_leverage"],  step=0.25)
        T["warn_entry_leverage"] = st.number_input("Leverage - Warning (x)",      value=T["warn_entry_leverage"], step=0.25)
        T["min_dscr"]  = st.number_input("Min DSCR Covenant (x)", value=T["min_dscr"],  step=0.05)
        T["warn_dscr"] = st.number_input("DSCR Warning (x)",      value=T["warn_dscr"], step=0.05)
        T["max_debt_ebitda"]       = st.number_input("Max Debt Capacity (x EBITDA)",  value=T["max_debt_ebitda"],       step=0.25)
        T["min_interest_coverage"] = st.number_input("Min Interest Coverage (x)",     value=T["min_interest_coverage"], step=0.1)
    st.markdown("---")
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("### Formula Parameters")
        F["debt_amort_years"]   = int(st.number_input("Amortization Period (yrs)", value=float(F["debt_amort_years"]), step=1.0))
        F["equity_pct_default"] = st.number_input("Default Equity (%)", value=F["equity_pct_default"]*100, step=5.0)/100
    with fc2:
        st.markdown("### Downside Stress `[MPE]`")
        F["downside_revenue_stress"] = st.number_input("Revenue Stress (%)",       value=F["downside_revenue_stress"]*100,  step=1.0) /100
        F["downside_margin_stress"]  = st.number_input("Margin Stress (bps)",      value=F["downside_margin_stress"]*10000, step=25.0)/10000
        F["downside_exit_stress"]    = st.number_input("Exit Multiple Stress (x)", value=F["downside_exit_stress"],         step=0.25)
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Save", type="primary"):
            st.session_state.T = T; st.session_state.F = F; st.success("Saved")
    with b2:
        if st.button("Reset to Defaults"):
            st.session_state.T = DEFAULT_T.copy(); st.session_state.F = DEFAULT_F.copy(); st.rerun()
    st.markdown("---")
    st.markdown("**Sources:** [R&P] Rosenbaum & Pearl 2020 · [MPE] Mastering PE · [McK] McKinsey Valuation")
    st.stop()

# ══════════════════════════════════════════════════════
# MAIN PAGE — SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## LBO Screener")
    st.caption("v4.1 - PE Analyst Edition")
    st.markdown("---")
    lc = st.selectbox(L["lang_label"], ["English","Deutsch"],
        index=0 if lang=="en" else 1, key="lang_main")
    nl = "en" if lc=="English" else "de"
    if nl != lang:
        st.session_state.lang = nl; st.rerun()

    st.markdown(f"### {L['data_input']}")
    input_mode = st.radio("mode",
        ["Moodys Orbis / Excel","Manual Input","Demo"],
        label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"### {L['deal_params']}")
    entry_multiple = st.slider("Entry EV/EBITDA",       4.0, 12.0, 6.5, 0.5)
    equity_pct     = st.slider("Equity (%)",             20,  60,   40,  5)/100
    debt_rate      = st.slider("Senior Debt Rate (%)",   3.0, 12.0, 6.5, 0.25)/100
    exit_multiple  = st.slider("Exit EV/EBITDA",        4.0, 14.0,  7.0, 0.5)
    hold_period    = st.slider("Hold Period (yrs)",       3,    7,    5)
    margin_exit    = st.slider("Exit EBITDA Margin (%)", 5.0, 40.0, 22.0, 0.5)/100
    st.markdown("---")
    st.caption("Hist. CAGR auto-applied as forward CAGR")
    override_cagr = st.checkbox("Override CAGR", False)
    manual_cagr   = st.slider("Fwd Rev. CAGR (%)", 0.0, 20.0, 4.0, 0.5)/100 if override_cagr else None

# ── DATA INPUT ─────────────────────────────────────────
company_inputs  = None
hist_metrics    = None
hist_years_used = []
parse_warnings  = []
timeseries_df   = None
ccy             = st.session_state.ccy

# ── DEMO ───────────────────────────────────────────────
if input_mode == "Demo":
    ccy = detect_currency_and_unit("tsd USD", lang=st.session_state.get("lang","en")); st.session_state.ccy = ccy
    company_inputs, hist_metrics, timeseries_df, hist_years_used = _build_demo_data()
    st.info(L["demo_info"])

# ── MANUAL INPUT ───────────────────────────────────────
elif input_mode == "Manual Input":
    unit_sel = st.selectbox("Unit", ["tsd EUR","tsd USD","Mio EUR","Mio USD"])
    ccy = detect_currency_and_unit(unit_sel, lang=st.session_state.get("lang","en")); st.session_state.ccy = ccy
    co_name = st.text_input("Company Name", "Target Co.")
    c1, c2, c3 = st.columns(3)
    with c1:
        rev    = st.number_input("Revenue",  value=56800.0, step=100.0)
        ebitda = st.number_input("EBITDA",   value=11900.0, step=100.0)
        ebit   = st.number_input("EBIT",     value=8900.0,  step=100.0)
        dep    = st.number_input("D&A",      value=3000.0,  step=100.0)
    with c2:
        interest = st.number_input("Interest Expense", value=1500.0, step=100.0)
        tax_rate = st.number_input("Tax Rate (%)",     value=26.0,   step=0.5)/100
        capex    = st.number_input("CapEx",            value=2500.0, step=100.0)
    with c3:
        debt = st.number_input("Financial Debt", value=20000.0, step=500.0)
        cash = st.number_input("Cash",           value=4500.0,  step=100.0)
        nwc  = st.number_input("NWC",            value=6800.0,  step=100.0)
    company_inputs = CompanyInputs(
        revenue=rev, ebitda=ebitda, ebit=ebit, depreciation=dep,
        interest_expense=interest, tax_rate=tax_rate, total_debt=debt,
        cash=cash, net_working_capital=nwc, capex=capex,
        company_name=co_name, currency_display=ccy["display"],
        revenue_cagr_hist=0.04, ebitda_margin_avg=ebitda/rev if rev else 0.20,
        capex_intensity=capex/rev if rev else 0.04, nwc_intensity=nwc/rev if rev else 0.10,
    )

# ── FILE UPLOAD ────────────────────────────────────────
elif input_mode == "Moodys Orbis / Excel":
    uploaded = st.file_uploader("Moodys Orbis 4-Sheet or Standard Excel", type=["xlsx","xls"])
    if uploaded:
        # Only re-parse if the file changed (compare by name+size)
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("_upload_id") != file_id:
            with st.spinner("Parsing..."):
                t0 = time.time()
                parser, is_moodys = parse_file(uploaded)
                st.session_state._upload_id    = file_id
                st.session_state._upload_parser = parser
                st.session_state._upload_is_moodys = is_moodys
                st.session_state._upload_ms    = int((time.time()-t0)*1000)
                # Clear old results
                st.session_state._upload_result = None

        parser    = st.session_state._upload_parser
        is_moodys = st.session_state._upload_is_moodys

        if is_moodys:
            p = parser; ccy = p.currency_info; st.session_state.ccy = ccy
            n_years = len(p.years)
            st.success(f"Moodys Orbis | {n_years} yrs | {min(p.years) if p.years else '?'}-{max(p.years) if p.years else '?'} | {ccy['display']} | parsed in {st.session_state._upload_ms}ms")
            ts = p.get_timeseries_df()
            if not ts.empty:
                col_rename = {
                    "revenue": "Revenue", "ebitda": "EBITDA", "ebit": "EBIT",
                    "depreciation": "D&A", "interest_expense": "Interest",
                    "net_income": "Net Income", "cash": "Cash",
                    "total_debt": "Total Debt", "net_working_capital": "NWC", "capex": "CapEx",
                }
                ts = ts.rename(columns={k: v for k, v in col_rename.items() if k in ts.columns})
                if "Revenue" in ts.columns and "EBITDA" in ts.columns:
                    ts["EBITDA Margin"] = ts["EBITDA"] / ts["Revenue"].replace(0, float("nan"))
                ts_sorted = ts.sort_index(ascending=True)
                timeseries_df = ts_sorted
                st.session_state._upload_timeseries = ts_sorted
                with st.expander("Time Series", expanded=False):
                    st.dataframe(ts.sort_index(ascending=False).style.format(
                        lambda x: f"{x:,.0f}" if isinstance(x,(float,int)) and not pd.isna(x) else "n/a"
                    ), use_container_width=True)
            elif st.session_state.get("_upload_timeseries") is not None:
                timeseries_df = st.session_state._upload_timeseries
            if n_years <= 3:
                selected_years = p.years
            elif n_years == 4:
                sel = st.radio(L["sel_years"], [L["y3"], L["all"]], horizontal=True)
                selected_years = sorted(p.years, reverse=True)[:3] if L["y3"] in sel else p.years
            else:
                sel = st.radio(L["sel_years"], [L["y3"], L["y5"], f"{L['all']} {n_years}"], horizontal=True)
                selected_years = (sorted(p.years, reverse=True)[:3] if L["y3"] in sel
                                  else sorted(p.years, reverse=True)[:5] if L["y5"] in sel
                                  else p.years)
            st.caption(f"{L['used_years']}: {sorted(selected_years)}")
            # Use cached result if years didn't change
            years_key = tuple(sorted(selected_years))
            if st.session_state.get("_upload_years_key") != years_key:
                st.session_state._upload_result = None

            if st.button(L["upload_btn"], type="primary") or st.session_state._upload_result:
                if not st.session_state._upload_result:
                    with st.spinner("Computing historical analytics..."):
                        hist_year_objs = p.build_historical_years(selected_years)
                        if len(hist_year_objs) < 2:
                            st.error("Need at least 2 years of data"); st.stop()
                        hm   = HistoricalAnalyzer(hist_year_objs).compute()
                        ci_u, pw = p.build_company_inputs(hm, hist_year_objs[-1])
                        st.session_state._upload_result = (ci_u, hm, pw, [y.year for y in hist_year_objs])
                        st.session_state._upload_years_key = years_key
                        st.session_state.ccy = ccy
                ci_u, hm_u, pw_u, yrs_u = st.session_state._upload_result
                company_inputs  = ci_u
                hist_metrics    = hm_u
                parse_warnings  = pw_u
                hist_years_used = yrs_u
                # Restore timeseries_df from session state (survives re-renders)
                if st.session_state.get("_upload_timeseries") is not None:
                    timeseries_df = st.session_state._upload_timeseries
        else:
            p = parser; ccy = p.currency_info; mapping_result = p.auto_map()
            st.warning(f"Generic parser | {len(mapping_result['mapped'])}/{len(REQUIRED_FIELDS)} fields mapped")
            manual = {}
            if mapping_result["unmapped"]:
                with st.expander("Missing Fields"):
                    for field in mapping_result["unmapped"]:
                        val = st.number_input(REQUIRED_FIELDS.get(field, field), value=0.0, key=f"m_{field}")
                        if val: manual[field] = val
            if st.button(L["upload_btn"], type="primary"):
                company_inputs, parse_warnings, ccy = p.build_company_inputs(manual)
                st.session_state.ccy = ccy
    else:
        # Clear cached upload if file removed
        st.session_state._upload_id = None
        st.session_state._upload_result = None
        st.info("Upload a Moodys Orbis Excel file or switch to Demo mode")

# ── GATE ───────────────────────────────────────────────
ccy      = st.session_state.ccy
sym      = ccy.get("symbol","")
unit     = ccy.get(f"unit_label_{lang}", ccy.get("unit_label",""))
ccy_name = ccy.get("currency","")
_raw     = ccy.get("raw_unit", "")
sfx      = " tsd." if _raw == "tsd" else " Mio." if _raw == "mio" else " Mrd." if _raw == "mrd" else ""
axis_lbl = f"{sym}{sfx}" if sfx else sym

if company_inputs is None:
    st.markdown("## Select a data input mode")
    st.markdown("""| Feature | Detail |
|---|---|
| Moodys Orbis | 4-sheet export auto-detected, up to 10 years |
| Historical Analytics | CAGR, Avg Margin, Volatility [R&P+McK] |
| Normalized Basis | Scrubbed EBITDA as LBO entry basis [R&P] |
| 3 Heatmaps | IRR, DSCR, Leverage [R&P] |
| LBO Score | Composite 0-100 across 6 dimensions |
| Value Bridge | IRR decomposed into 3 drivers [R&P p.162] |
| Deal Screener | Multi-company ranking + Thesis Generator |""")
    st.stop()

# ══════════════════════════════════════════════════════
# ENGINE — cached, only re-runs when inputs actually change
# ══════════════════════════════════════════════════════
fwd_cagr     = manual_cagr if override_cagr else company_inputs.revenue_cagr_hist
entry_margin = company_inputs.ebitda / company_inputs.revenue if company_inputs.revenue else 0.20

assumptions = LBOAssumptions(
    entry_ev_multiple=entry_multiple, equity_contribution_pct=equity_pct,
    senior_debt_rate=debt_rate, debt_amortization_years=F["debt_amort_years"],
    exit_multiple=exit_multiple, holding_period=hold_period,
    revenue_cagr=fwd_cagr, ebitda_margin_entry=entry_margin,
    ebitda_margin_exit=margin_exit,
    max_leverage_covenant=T["max_debt_ebitda"], min_dscr_covenant=T["min_dscr"],
)

hm_tuple = _pack_hm(hist_metrics)
cia = _ci_args(company_inputs)
aa  = _a_args(assumptions)

results  = _run_lbo(**cia, hm_tuple=hm_tuple, **aa)
irr_hm, dscr_hm, lev_hm = _run_sensitivities(**cia, **aa)

# ── FLAGS ──────────────────────────────────────────────
def evaluate_flags(res, ci, T, hm):
    flags = []
    def f(c, lvl, msg):
        if c: flags.append((lvl, msg))
    f(res.entry_leverage > T["max_entry_leverage"],
      "red",  f"Entry Leverage {res.entry_leverage:.1f}x > {T['max_entry_leverage']}x  [MPE]")
    f(T["warn_entry_leverage"] < res.entry_leverage <= T["max_entry_leverage"],
      "warn", f"Entry Leverage {res.entry_leverage:.1f}x in warning zone")
    f(res.dscr_base < T["min_dscr"],
      "red",  f"DSCR Y1 {res.dscr_base:.2f}x < {T['min_dscr']}x Covenant Floor  [MPE]")
    f(T["min_dscr"] <= res.dscr_base < T["warn_dscr"],
      "warn", f"DSCR {res.dscr_base:.2f}x in warning zone")
    f(res.irr < T["min_irr"],
      "red",  f"IRR {res.irr:.1%} < {T['min_irr']:.0%} Hurdle  [R&P]")
    f(T["min_irr"] <= res.irr < T["warn_irr"],
      "warn", f"IRR {res.irr:.1%} below target {T['warn_irr']:.0%}")
    f(res.moic < T["min_moic"],
      "red",  f"MOIC {res.moic:.1f}x < {T['min_moic']}x  [R&P]")
    f(res.covenant_breach_year is not None,
      "red",  f"Covenant breach projected Year {res.covenant_breach_year}  [MPE]")
    f(ci.ebitda/max(ci.revenue,1) < T["min_ebitda_margin"],
      "warn", f"EBITDA Margin {ci.ebitda/ci.revenue:.1%} < {T['min_ebitda_margin']:.0%}")
    f(res.fcf_yield < T["min_fcf_yield"],
      "warn", f"FCF Yield {res.fcf_yield:.1%} < {T['min_fcf_yield']:.0%}")
    f(res.interest_coverage < T["min_interest_coverage"],
      "warn", f"Interest Coverage {res.interest_coverage:.1f}x < {T['min_interest_coverage']}x")
    if hm and hm.ebitda_volatility > 0.08:
        flags.append(("warn", f"EBITDA Margin Volatility {hm.ebitda_volatility:.1%} - cyclical  [MPE]"))
    if res.cash_conversion < 0.30:
        flags.append(("red",  f"Cash Conversion {res.cash_conversion:.1%} < 30% — FCF quality critical  [McK]"))
    elif res.cash_conversion < 0.50:
        flags.append(("warn", f"Cash Conversion {res.cash_conversion:.1%} below 50%  [McK]"))
    if hm and hm.revenue_volatility > 0.20:
        flags.append(("warn", f"Revenue Volatility {hm.revenue_volatility:.1%} > 20% — highly cyclical  [MPE]"))
    return flags

all_flags  = evaluate_flags(results, company_inputs, T, hist_metrics)
red_flags  = [(l,m) for l,m in all_flags if l=="red"]
warn_flags = [(l,m) for l,m in all_flags if l=="warn"]
is_viable  = len(red_flags)==0 and results.irr >= T["min_irr"]

# ── HEADER ─────────────────────────────────────────────
htitle, hbadge = st.columns([3,1])
with htitle:
    st.markdown(
        f"# {company_inputs.company_name} "
        f"<span class='currency-badge'>{ccy_name} - {unit}</span>",
        unsafe_allow_html=True
    )
    if hist_years_used:
        st.caption(
            f"{L['hist_basis']}: {min(hist_years_used)}-{max(hist_years_used)} "
            f"({len(hist_years_used)} {L['yrs']}) | "
            f"{L['hist_cagr']}: {company_inputs.revenue_cagr_hist:.1%} | "
            f"{L['avg_margin']}: {company_inputs.ebitda_margin_avg:.1%} | "
            f"{L['fwd_cagr']}: {fwd_cagr:.1%} ({L['normalized']})"
        )

vc = "#00cc88" if is_viable else "#ff4b4b"
vl = ("✅ " if is_viable else "❌ ") + L["viable" if is_viable else "critical"]
with hbadge:
    st.markdown(
        f'<div style="text-align:right;margin-top:12px">'
        f'<span style="border:1px solid {vc};border-radius:8px;padding:6px 14px;'
        f'color:{vc};font-weight:700">{vl}</span><br>'
        f'<span style="font-size:.8em;color:#aaa;margin-top:4px;display:block">'
        f'🔴 {len(red_flags)} · ⚠️ {len(warn_flags)}</span></div>',
        unsafe_allow_html=True
    )
st.markdown("---")

# ── KPI ROWS ───────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
for col,lbl,val in [
    (k1,"Entry EV",       f"{sym}{fmt_num(results.entry_ev, lang)}"),
    (k2,"Entry Leverage", f"{results.entry_leverage:.1f}x"),
    (k3,"DSCR Y1",        f"{results.dscr_base:.2f}x"),
    (k4,"Base IRR",       f"{results.irr:.1%}"),
    (k5,"MOIC",           f"{results.moic:.2f}x"),
    (k6,"FCF Yield Y1",   f"{results.fcf_yield:.1%}"),
]:
    with col: st.metric(lbl, val)

r2a,r2b,r2c,r2d,r2e,r2f = st.columns(6)
for col,lbl,val in [
    (r2a,"Downside IRR",  f"{results.downside_irr:.1%}"),
    (r2b,"Downside MOIC", f"{results.downside_moic:.2f}x"),
    (r2c,"Cash Conv.",    f"{results.cash_conversion:.1%}"),
    (r2d,"Exit Equity",   f"{sym}{fmt_num(results.exit_equity, lang)}"),
    (r2e,"Interest Cov.", f"{results.interest_coverage:.1f}x"),
    (r2f,"LBO Score",     f"{results.lbo_score:.0f}/100"),
]:
    with col: st.metric(lbl, val)
st.markdown("---")

# ══════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════
tab_hist, tab_lbo, tab_debt, tab_sens, tab_score, tab_bridge, tab_screen, tab_flags = st.tabs([
    "📈 "+L["tab_hist"], "🏦 "+L["tab_lbo"], "📅 "+L["tab_debt"], "🗺️ "+L["tab_sens"],
    "🏆 "+L["tab_score"], "🔗 "+L["tab_bridge"], "🔎 "+L["tab_screen"], "🚩 "+L["tab_flags"],
])

# ── Helper: Heatmap figure ──────────────────────────────
def _hm_fig(df, title, fmt_str, zmid, note=""):
    vals = df.values.astype(float)
    cs   = [[0.0,"#ff4b4b"],[0.35,"#ffaa00"],[0.55,"#4f8ef7"],[1.0,"#00cc88"]]
    fig  = go.Figure(go.Heatmap(
        z=vals, x=df.columns.tolist(), y=df.index.tolist(),
        colorscale=cs, zmid=zmid,
        text=[[fmt_str.format(v) for v in row] for row in vals],
        texttemplate="%{text}", colorbar=dict(thickness=12),
    ))
    if note:
        fig.add_annotation(text=note, xref="paper", yref="paper", x=0.01, y=-0.14,
            showarrow=False, font=dict(color="#6e7681",size=9))
    fig.update_layout(template="plotly_dark", height=370, title=title,
        margin=dict(t=40,b=55,l=100,r=20))
    return fig

# ── Helper: inline colored badge (no background, just border) ─
def _badge(text, color, subtext=""):
    sub = f'<div style="font-size:.78em;color:#aaa;margin-top:3px">{subtext}</div>' if subtext else ""
    return (f'<div style="border-left:3px solid {color};padding:6px 12px;margin:4px 0">'
            f'<span style="color:{color};font-weight:700">{text}</span>{sub}</div>')


# ══ TAB 1: HISTORICAL ══════════════════════════════════
with tab_hist:
    st.markdown('<div class="src-tag">[R&P p.204 + McK p.193] Historical basis for LBO normalization</div>', unsafe_allow_html=True)
    if hist_metrics:
        h = hist_metrics
        hc1,hc2,hc3,hc4 = st.columns(4)
        hc1.metric(L["revenue_cagr"], f"{h.revenue_cagr:.1%}")
        hc2.metric(L["avg_ebitda"],   f"{h.ebitda_margin_avg:.1%}", help=f"Median {h.ebitda_margin_med:.1%}")
        hc3.metric(L["ebitda_vol"],   f"{h.ebitda_volatility:.1%}")
        hc4.metric(L["fcf_conv"],     f"{h.fcf_conversion_avg:.1%}")
        hc5,hc6,hc7,hc8 = st.columns(4)
        hc5.metric(L["capex_int"],   f"{h.capex_intensity_avg:.1%}")
        hc6.metric(L["nwc_int"],     f"{h.nwc_intensity_avg:.1%}")
        hc7.metric(L["ic_avg"],      f"{h.interest_coverage_avg:.1f}x")
        hc8.metric(L["norm_ebitda"], f"{sym}{fmt_num(h.normalized_ebitda, lang, sfx=sfx)}")

        # Build timeseries from hist_metrics if timeseries_df is not available
        _chart_df = timeseries_df
        if (_chart_df is None or _chart_df.empty) and h.years_used and h.revenue_series and h.ebitda_series:
            _chart_df = pd.DataFrame({
                "Revenue": h.revenue_series,
                "EBITDA":  h.ebitda_series,
                "EBITDA Margin": h.margin_series if h.margin_series else [e/max(r,1) for r,e in zip(h.revenue_series, h.ebitda_series)],
            }, index=h.years_used).sort_index()

        if _chart_df is not None and not _chart_df.empty:
            df_s  = _chart_df.sort_index(ascending=True)
            yidx  = [str(y) for y in df_s.index]

            st.markdown(f'<div class="section-hdr">{L["rev_ebitda_chart"]}</div>', unsafe_allow_html=True)
            fig = go.Figure()
            if "Revenue" in df_s.columns:
                fig.add_trace(go.Bar(x=yidx, y=list(df_s["Revenue"]),
                    name="Revenue", marker_color="#4f8ef7", opacity=0.7))
            if "EBITDA" in df_s.columns:
                fig.add_trace(go.Scatter(x=yidx, y=list(df_s["EBITDA"]),
                    name="EBITDA", mode="lines+markers", line=dict(color="#00cc88",width=2.5)))
            if "EBITDA Margin" in df_s.columns:
                fig.add_trace(go.Scatter(x=yidx, y=list(df_s["EBITDA Margin"]),
                    name="EBITDA Margin %", mode="lines+markers",
                    line=dict(color="#ffaa00",width=1.5,dash="dot"), yaxis="y2"))
            fig.update_layout(template="plotly_dark", height=320, barmode="group",
                yaxis=dict(title=axis_lbl, tickformat=",.0f"),
                yaxis2=dict(title="Margin %", overlaying="y", side="right", tickformat=".0%"),
                legend=dict(x=0.01,y=0.99,bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=20,b=20,l=60,r=60))
            st.plotly_chart(fig, use_container_width=True)

            if h.margin_series:
                st.markdown(f'<div class="section-hdr">{L["margin_trend"]}</div>', unsafe_allow_html=True)
                mp = sorted(zip(h.years_used, h.margin_series))
                my = [str(y) for y,_ in mp]; mv = [v for _,v in mp]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=my, y=mv, mode="lines+markers",
                    line=dict(color="#4f8ef7",width=2),
                    fill="tozeroy", fillcolor="rgba(79,142,247,0.08)"))
                fig2.add_hline(y=h.ebitda_margin_avg, line_dash="dash", line_color="#00cc88",
                    annotation_text=f"Avg {h.ebitda_margin_avg:.1%}")
                fig2.add_hline(y=T["min_ebitda_margin"], line_dash="dot", line_color="#ff4b4b",
                    annotation_text=f"Min {T['min_ebitda_margin']:.0%}")
                fig2.update_layout(template="plotly_dark", height=200,
                    yaxis=dict(tickformat=".0%"), showlegend=False,
                    margin=dict(t=20,b=20,l=50,r=20))
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f'<div class="section-hdr">{L["norm_basis"]}</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            L["metric"]: [L["norm_rev_row"],L["norm_ebi_row"],L["avg_mar_row"],L["norm_cap_row"],L["nwc_del_row"]],
            L["value"]:  [f"{sym}{fmt_num(h.normalized_revenue, lang, sfx=sfx)}",
                          f"{sym}{fmt_num(h.normalized_ebitda, lang, sfx=sfx)}",
                          f"{h.ebitda_margin_avg:.1%}",
                          f"{sym}{fmt_num(h.normalized_capex, lang, sfx=sfx)}",
                          f"{sym}{fmt_num(h.normalized_nwc_delta, lang, sfx=sfx)}"],
            L["method"]: [L["curr_rev"],L["avg_m_curr"],
                          f"{L['simple_avg']} ({len(h.years_used)} {L['yrs']})",
                          L["avg_cap_int"],L["nwc_int_delt"]],
        }).set_index(L["metric"]), use_container_width=True)
        st.caption("Normalization per Rosenbaum & Pearl Ch.4")
    else:
        st.info(L["hist_only"])

# ══ TAB 2: LBO STRUCTURE ═══════════════════════════════
with tab_lbo:
    st.markdown('<div class="src-tag">[R&P Ch.4-5] LBO structure · FCF projection · Returns</div>', unsafe_allow_html=True)
    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown(f'<div class="section-hdr">{L["entry_struct"]}</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "": [L["entry_ev"],L["entry_eq"],L["entry_debt"],L["net_lev"],L["debt_cap"]],
            axis_lbl: [
                f"{sym}{fmt_num(results.entry_ev, lang, sfx=sfx)}",
                f"{sym}{fmt_num(results.entry_equity, lang, sfx=sfx)} ({equity_pct:.0%})",
                f"{sym}{fmt_num(results.entry_debt, lang, sfx=sfx)} ({1-equity_pct:.0%})",
                f"{results.entry_leverage:.1f}x Net Debt/EBITDA",
                f"{sym}{fmt_num(results.debt_capacity, lang, sfx=sfx)} ({T['max_debt_ebitda']}x EBITDA)",
            ]
        }).set_index(""), use_container_width=True)

        st.markdown(f'<div class="section-hdr">{L["returns"]}</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "": [L["exit_ebitda"],L["exit_ev"],L["exit_eq"],
                 L["base_irr"],L["base_moic"],L["ds_irr"],L["ds_moic"]],
            "Value": [
                f"{sym}{fmt_num(results.exit_ebitda, lang, sfx=sfx)}",
                f"{sym}{fmt_num(results.exit_ev, lang, sfx=sfx)}",
                f"{sym}{fmt_num(results.exit_equity, lang, sfx=sfx)}",
                f"{results.irr:.1%}", f"{results.moic:.2f}x",
                f"{results.downside_irr:.1%}", f"{results.downside_moic:.2f}x",
            ]
        }).set_index(""), use_container_width=True)

        st.markdown(f'<div class="section-hdr">{L["input_sum"]}</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            L["metric"]: [L["revenue"],L["ebitda"],L["ebit"],L["da"],
                          L["interest"],L["tax"],L["debt"],L["cash"],L["nwc"],L["capex"]],
            axis_lbl: [
                fmt_num(company_inputs.revenue, lang, sfx=sfx),
                f"{fmt_num(company_inputs.ebitda, lang, sfx=sfx)} ({entry_margin:.1%})",
                fmt_num(company_inputs.ebit, lang, sfx=sfx),
                fmt_num(company_inputs.depreciation, lang, sfx=sfx),
                fmt_num(company_inputs.interest_expense, lang, sfx=sfx),
                f"{company_inputs.tax_rate:.1%}",
                fmt_num(company_inputs.total_debt, lang, sfx=sfx),
                fmt_num(company_inputs.cash, lang, sfx=sfx),
                fmt_num(company_inputs.net_working_capital, lang, sfx=sfx),
                fmt_num(company_inputs.capex, lang, sfx=sfx),
            ]
        }).set_index(L["metric"]), use_container_width=True)

    with lc2:
        yrs_x = [f"Y{y}" for y in range(1, hold_period+1)]
        st.markdown(f'<div class="section-hdr">{L["fcf_proj"]}</div>', unsafe_allow_html=True)
        fig_fcf = go.Figure()
        fig_fcf.add_trace(go.Bar(x=yrs_x, y=results.revenue_proj,
            name="Revenue", marker_color="#4f8ef7", opacity=0.5))
        fig_fcf.add_trace(go.Bar(x=yrs_x, y=results.ebitda_proj,
            name="EBITDA", marker_color="#00cc88", opacity=0.6))
        fig_fcf.add_trace(go.Scatter(x=yrs_x, y=results.fcf_series,
            mode="lines+markers", name="FCF", line=dict(color="#ffaa00",width=2.5)))
        fig_fcf.update_layout(template="plotly_dark", height=270, barmode="group",
            yaxis=dict(title=axis_lbl, tickformat=",.0f"), showlegend=False,   # ← no legend in chart
            margin=dict(t=10,b=20,l=60,r=20))
        st.plotly_chart(fig_fcf, use_container_width=True)
        # legend as text instead
        st.caption("Revenue (bars blue) · EBITDA (bars green) · FCF (line orange)  |  FCF = NOPAT + D&A - CapEx - dNWC  [McK p.163]")

        st.markdown(f'<div class="section-hdr">{L["dscr_dev"]}</div>', unsafe_allow_html=True)
        fig_dscr = go.Figure()
        fig_dscr.add_trace(go.Scatter(x=yrs_x, y=results.dscr_series,
            mode="lines+markers", line=dict(color="#4f8ef7",width=2.5)))
        fig_dscr.add_hline(y=T["min_dscr"], line_dash="dash", line_color="#ff4b4b",
            annotation_text=f"{L['floor']} {T['min_dscr']}x")
        fig_dscr.add_hline(y=T["warn_dscr"], line_dash="dot", line_color="#ffaa00",
            annotation_text=f"{L['warn']} {T['warn_dscr']}x")
        fig_dscr.update_layout(template="plotly_dark", height=210, showlegend=False,
            margin=dict(t=10,b=20,l=50,r=20))
        st.plotly_chart(fig_dscr, use_container_width=True)

# ══ TAB 3: DEBT SCHEDULE ═══════════════════════════════
with tab_debt:
    st.markdown(f'<div class="src-tag">{L["debt_sched"]}</div>', unsafe_allow_html=True)
    ds = results.debt_schedule
    st.dataframe(ds.style.format({
        "Opening":      lambda v: f"{v:,.1f}{sfx}",
        "Interest":     lambda v: f"{v:,.1f}{sfx}",
        "Amortization": lambda v: f"{v:,.1f}{sfx}",
        "Cash Sweep":   lambda v: f"{v:,.1f}{sfx}",
        "Closing":      lambda v: f"{v:,.1f}{sfx}",
        "Coverage":     lambda v: f"{v:.2f}x",
    }).background_gradient(subset=["Closing"], cmap="RdYlGn"), use_container_width=True)

    fig_wf = go.Figure(go.Waterfall(
        x=[f"Y{y}" for y in ds.index],
        y=[-row["Amortization"] for _,row in ds.iterrows()],
        base=results.entry_debt, measure=["relative"]*len(ds),
        connector=dict(line=dict(color="#2a2f3e")),
        decreasing=dict(marker_color="#00cc88"),
    ))
    fig_wf.update_layout(template="plotly_dark", height=250, title=L["debt_wfall"],
        yaxis=dict(title=axis_lbl, tickformat=",.0f"), showlegend=False,
        margin=dict(t=40,b=20,l=60,r=20))
    st.plotly_chart(fig_wf, use_container_width=True)

    dc1,dc2,dc3 = st.columns(3)
    dc1.metric(f"Debt Capacity ({T['max_debt_ebitda']}x)", f"{sym}{fmt_num(results.debt_capacity, lang, sfx=sfx)}")
    dc2.metric("Entry Debt", f"{sym}{fmt_num(results.entry_debt, lang, sfx=sfx)}")
    hw = results.debt_capacity - results.entry_debt
    dc3.metric(L["headroom"], f"{sym}{fmt_num(hw, lang, sfx=sfx)}", delta="OK" if hw>0 else "Exceeded")

# ══ TAB 4: SENSITIVITIES ═══════════════════════════════
with tab_sens:
    st.markdown(f'<div class="src-tag">[R&P p.200] {L["sens_title"]}</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        st.plotly_chart(_hm_fig(irr_hm,  L["irr_hm"],  "{:.1f}%",  T["min_irr"]*100,
            f"{L['hurdle']} {T['min_irr']:.0%}"), use_container_width=True)
    with s2:
        st.plotly_chart(_hm_fig(dscr_hm, L["dscr_hm"], "{:.2f}x",  T["warn_dscr"],
            f"{L['floor']} {T['min_dscr']}x"), use_container_width=True)
    st.markdown(f'<div class="section-hdr">{L["lev_hm"]} </div>', unsafe_allow_html=True)
    st.plotly_chart(_hm_fig(lev_hm, L["lev_hm"], "{:.1f}%", T["min_irr"]*100), use_container_width=True)
    st.caption(f"Thresholds: IRR >= {T['min_irr']:.0%} · DSCR >= {T['min_dscr']}x · Leverage <= {T['max_entry_leverage']}x")

# ══ TAB 5: LBO SCORE ═══════════════════════════════════
with tab_score:
    st.markdown('<div class="src-tag">[R&P + MPE + McK] Composite scoring · Cash Conversion · Revenue Quality · Realistic Debt Capacity</div>', unsafe_allow_html=True)

    sc = results.lbo_score
    sc_color = "#00cc88" if sc >= 65 else "#ffaa00" if sc >= 45 else "#ff4b4b"
    sc_label  = "Strong Buy" if sc>=75 else "Attractive" if sc>=65 else "Borderline" if sc>=50 else "Weak" if sc>=35 else "Pass"

    gc1, gc2 = st.columns([1,2])
    with gc1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sc,
            title=dict(text=L["score_title"], font=dict(size=14)),
            gauge=dict(
                axis=dict(range=[0,100], tickwidth=1),
                bar=dict(color=sc_color, thickness=0.25),
                bgcolor="rgba(0,0,0,0)",
                steps=[
                    dict(range=[0,35],  color="rgba(255,75,75,0.12)"),
                    dict(range=[35,50], color="rgba(255,75,75,0.06)"),
                    dict(range=[50,65], color="rgba(255,170,0,0.12)"),
                    dict(range=[65,80], color="rgba(0,204,136,0.12)"),
                    dict(range=[80,100],color="rgba(0,204,136,0.20)"),
                ],
                threshold=dict(line=dict(color="#4f8ef7",width=3), value=60),
            )
        ))
        fig_gauge.update_layout(template="plotly_dark", height=240,
            margin=dict(t=30,b=10,l=20,r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown(_badge(sc_label, sc_color, L["score_sub"]), unsafe_allow_html=True)

    with gc2:
        bkd = results.lbo_score_breakdown
        fnames  = list(bkd.keys())
        fvals   = list(bkd.values())
        max_pts = [30,20,15,15,10,10]
        f_colors = ["#00cc88" if v/m>=0.65 else "#ffaa00" if v/m>=0.40 else "#ff4b4b"
                    for v,m in zip(fvals, max_pts)]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(y=fnames, x=max_pts, orientation="h",
            name="Max", marker_color="rgba(30,40,60,0.5)", showlegend=False))
        fig_bar.add_trace(go.Bar(y=fnames, x=fvals, orientation="h",
            name="Score", marker_color=f_colors, showlegend=False,
            text=["" for _ in fvals],  # no inline text
            textposition="outside"))
        # Add labels as annotations always to the right of the max bar
        for i, (fname, fval, mpt) in enumerate(zip(fnames, fvals, max_pts)):
            fig_bar.add_annotation(
                x=mpt + 1.5, y=fname,
                text=f"<b>{fval:.1f}/{mpt}</b>",
                showarrow=False,
                xanchor="left",
                font=dict(size=11, color="#e0e0e0"),
                xref="x", yref="y",
            )
        fig_bar.update_layout(template="plotly_dark", height=260, barmode="overlay",
            xaxis=dict(range=[0, 40], title="Score"),
            margin=dict(t=10,b=20,l=110,r=80))
        st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("---")

    # ── A. Cash Conversion ─────────────────────────────
    st.markdown(f'<div class="section-hdr">{L["cc_title"]}</div>', unsafe_allow_html=True)
    cc = results.cash_conversion
    cc_color = "#00cc88" if cc>=0.70 else "#4f8ef7" if cc>=0.50 else "#ffaa00" if cc>=0.30 else "#ff4b4b"
    cc_text  = L["cc_excellent"] if cc>=0.70 else L["cc_normal"] if cc>=0.50 else L["cc_critical"] if cc>=0.30 else L["cc_flag"]

    cca, ccb, ccc = st.columns([1,1,2])
    with cca:
        st.metric(L["cc_label"], f"{cc:.1%}")
    with ccb:
        st.markdown(_badge(cc_text, cc_color,
            f"FCF Y1 {sym}{fmt_num(results.fcf_series[0] if results.fcf_series else 0, lang, sfx=sfx)} / EBITDA {sym}{fmt_num(company_inputs.ebitda, lang, sfx=sfx)}"),
            unsafe_allow_html=True)
    with ccc:
        ebi   = company_inputs.ebitda
        dep   = company_inputs.depreciation
        capex = company_inputs.capex
        nwc_d = company_inputs.net_working_capital * company_inputs.revenue_cagr_hist
        tax_  = company_inputs.ebit * company_inputs.tax_rate
        fig_cc = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute","relative","relative","relative","relative","total"],
            x=["EBITDA","+D&A","-CapEx","-dNWC","-Tax (approx)","FCF Y1"],
            y=[ebi, dep, -capex, -nwc_d, -tax_, 0],
            connector=dict(line=dict(color="#2a2f3e")),
            increasing=dict(marker_color="#00cc88"),
            decreasing=dict(marker_color="#ff4b4b"),
            totals=dict(marker_color=cc_color),
            texttemplate="%{y:,.0f}", textposition="outside",
        ))
        fig_cc.update_layout(template="plotly_dark", height=220,
            yaxis=dict(title=axis_lbl, tickformat=",.0f"), showlegend=False,
            margin=dict(t=10,b=20,l=60,r=20))
        st.plotly_chart(fig_cc, use_container_width=True)
    st.caption("Cash Conversion = FCF Y1 / EBITDA  [McK]")

    # ── B. Revenue Quality ─────────────────────────────
    st.markdown(f'<div class="section-hdr">{L["rev_qual_title"]}</div>', unsafe_allow_html=True)
    if hist_metrics:
        rv = hist_metrics.revenue_volatility
        rv_color = "#00cc88" if rv<0.10 else "#ffaa00" if rv<0.20 else "#ff4b4b"
        rv_text  = L["rev_stable"] if rv<0.10 else L["rev_moderate"] if rv<0.20 else L["rev_cyclical"]
        rqa,rqb,rqc = st.columns([1,1,2])
        with rqa:
            st.metric(L["rev_vol_label"], f"{rv:.1%}")
        with rqb:
            st.markdown(_badge(rv_text, rv_color), unsafe_allow_html=True)
        with rqc:
            if hist_metrics.revenue_series and hist_metrics.years_used:
                rp = sorted(zip(hist_metrics.years_used, hist_metrics.revenue_series))
                ry = [str(y) for y,_ in rp]; rv_s = [v for _,v in rp]
                mean_r = sum(rv_s)/len(rv_s)
                fig_rv = go.Figure()
                fig_rv.add_trace(go.Scatter(x=ry, y=rv_s, mode="lines+markers",
                    line=dict(color="#4f8ef7",width=2), showlegend=False))
                fig_rv.add_hline(y=mean_r, line_dash="dash", line_color="#ffaa00",
                    annotation_text=f"Avg {sym}{fmt_num(mean_r, lang, sfx=sfx)}")
                fig_rv.update_layout(template="plotly_dark", height=200,
                    yaxis=dict(title=axis_lbl, tickformat=",.0f"), showlegend=False,
                    margin=dict(t=10,b=20,l=60,r=20))
                st.plotly_chart(fig_rv, use_container_width=True)
        st.caption("Revenue Volatility = StdDev/Mean  [MPE]")
    else:
        st.info(L["hist_only"])

    # ── C. Realistic Debt Capacity ─────────────────────
    st.markdown(f'<div class="section-hdr">{L["debt_real_title"]}</div>', unsafe_allow_html=True)
    dc_simple = results.debt_capacity
    dc_dscr   = results.debt_capacity_dscr
    dc_eff    = min(dc_simple, dc_dscr)
    dca,dcb,dcc,dcd = st.columns(4)
    dca.metric(L["dc_simple"],    f"{sym}{fmt_num(dc_simple, lang, sfx=sfx)}", help=f"{T['max_debt_ebitda']}x EBITDA")
    dcb.metric(L["dc_dscr"],      f"{sym}{fmt_num(dc_dscr, lang, sfx=sfx)}",  help="Binary search: max debt @ DSCR>=1.30x")
    dcc.metric(L["dc_effective"], f"{sym}{fmt_num(dc_eff, lang, sfx=sfx)}",   delta=f"{dc_eff/max(company_inputs.ebitda,1):.1f}x EBITDA")
    dcd.metric("Entry Debt",      f"{sym}{fmt_num(results.entry_debt, lang, sfx=sfx)}",
               delta=f"Headroom {sym}{fmt_num(dc_eff-results.entry_debt, lang, sfx=sfx)}")
    fig_dc = go.Figure()
    bar_labs = [L["dc_simple"],L["dc_dscr"],L["dc_effective"],"Entry Debt"]
    bar_vals = [dc_simple,dc_dscr,dc_eff,results.entry_debt]
    bar_cols = ["#4f8ef7","#ffaa00","#00cc88","#ff4b4b"]
    fig_dc.add_trace(go.Bar(x=bar_labs, y=bar_vals, marker_color=bar_cols,
        text=[f"{sym}{fmt_num(v, lang, sfx=sfx)}" for v in bar_vals], textposition="outside",
        showlegend=False))
    fig_dc.update_layout(template="plotly_dark", height=220,
        yaxis=dict(title=axis_lbl, tickformat=",.0f"), showlegend=False,
        margin=dict(t=10,b=20,l=60,r=20))
    st.plotly_chart(fig_dc, use_container_width=True)
    st.caption("Realistic Debt Capacity = min(MaxLev x EBITDA, DSCR-constrained max)  [MPE]")

    # ── D. Entry Multiple Benchmark ────────────────────
    st.markdown(f'<div class="section-hdr">{L["overpay_title"]}</div>', unsafe_allow_html=True)
    BENCHMARKS = {
        "Manufacturing":(6.0,7.5),"Technology (SaaS)":(10.0,14.0),
        "Technology (Other)":(7.0,10.0),"Healthcare":(8.0,12.0),
        "Business Services":(7.0,9.0),"Consumer / Retail":(6.0,8.5),
        "Industrials":(5.5,7.5),"Financial Services":(7.0,10.0),
        "Energy / Infra":(6.5,9.0),"Distribution":(5.5,7.0),
        "Food & Beverage":(7.0,9.5),"Media / Telecom":(6.0,8.0),
        "Pharma / Life Sci.":(9.0,13.0),"Real Estate":(8.0,12.0),
    }
    sel_ind = st.selectbox(L["thesis_industry"], list(BENCHMARKS.keys()), key="ind_sel")
    bm_lo,bm_hi = BENCHMARKS[sel_ind]; bm_mid=(bm_lo+bm_hi)/2
    em = entry_multiple; ovp=(em-bm_mid)/bm_mid
    ovc = "#00cc88" if em<=bm_hi else "#ffaa00" if em<=bm_hi*1.10 else "#ff4b4b"
    ovl = "In range" if em<=bm_hi else "Slight premium" if em<=bm_hi*1.10 else "Overpay risk"

    bmc1,bmc2 = st.columns([1,2])
    with bmc1:
        st.metric("Entry Multiple",        f"{em:.1f}x")
        st.metric(f"{sel_ind} Median",     f"{bm_mid:.1f}x")
        st.metric(f"{sel_ind} Range",      f"{bm_lo:.1f}x - {bm_hi:.1f}x")
        st.markdown(_badge(ovl, ovc, f"{ovp:+.1%} vs. median"), unsafe_allow_html=True)
    with bmc2:
        alls = list(BENCHMARKS.keys())
        bls  = [BENCHMARKS[i][0] for i in alls]
        bhs  = [BENCHMARKS[i][1] for i in alls]
        bms  = [(l+h)/2 for l,h in zip(bls,bhs)]
        fig_bm = go.Figure()
        fig_bm.add_trace(go.Bar(name="Range", x=alls,
            y=[h-l for l,h in zip(bls,bhs)], base=bls,
            marker_color="rgba(79,142,247,0.15)",
            marker_line=dict(color="#4f8ef7",width=1), showlegend=False))
        fig_bm.add_trace(go.Scatter(x=alls, y=bms, mode="markers",
            marker=dict(color="#4f8ef7",size=8,symbol="diamond"),
            name="Median", showlegend=False))
        fig_bm.add_hline(y=em, line_dash="solid", line_color="#ffaa00", line_width=2,
            annotation_text=f"Entry {em:.1f}x", annotation_position="top right")
        fig_bm.update_layout(template="plotly_dark", height=280,
            yaxis=dict(title="EV/EBITDA x"),
            xaxis=dict(tickangle=-35),
            margin=dict(t=20,b=100,l=60,r=20))
        st.plotly_chart(fig_bm, use_container_width=True)
    st.caption("Benchmarks: indicative industry medians  [R&P + Pitchbook]")


# ══ TAB 6: VALUE BRIDGE ════════════════════════════════
with tab_bridge:
    st.markdown('<div class="src-tag">[R&P p.162] IRR decomposition into 3 value creation drivers</div>', unsafe_allow_html=True)
    st.markdown(f"### {L['bridge_title']}")
    st.caption(L["bridge_sub"])

    eg = results.vc_ebitda_growth
    me = results.vc_multiple_exp
    dp = results.vc_debt_paydown

    bc1,bc2,bc3,bc4 = st.columns(4)
    bc1.metric(L["bridge_ebitda"], f"{eg:.1%}")
    bc2.metric(L["bridge_mult"],   f"{me:.1%}")
    bc3.metric(L["bridge_debt"],   f"{dp:.1%}")
    bc4.metric("Base IRR",         f"{results.irr:.1%}")
    st.markdown("---")

    br1, br2 = st.columns(2)
    with br1:
        fig_br = go.Figure()
        for nm,vl,cl in [(L["bridge_ebitda"],eg,"#4f8ef7"),
                         (L["bridge_mult"],  me,"#00cc88"),
                         (L["bridge_debt"],  dp,"#ffaa00")]:
            fig_br.add_trace(go.Bar(x=["Value Creation Mix"], y=[vl*100],
                name=nm, marker_color=cl,
                text=[f"{vl:.1%}"], textposition="inside"))
        fig_br.update_layout(template="plotly_dark", height=300, barmode="stack",
            yaxis=dict(title="Share (%)"),
            legend=dict(orientation="h",y=-0.20),
            margin=dict(t=10,b=70,l=60,r=20))
        st.plotly_chart(fig_br, use_container_width=True)

    with br2:
        fig_pie = go.Figure(go.Pie(
            labels=[L["bridge_ebitda"],L["bridge_mult"],L["bridge_debt"]],
            values=[eg*100,me*100,dp*100],
            marker_colors=["#4f8ef7","#00cc88","#ffaa00"],
            textinfo="label+percent", hole=0.4,
        ))
        fig_pie.update_layout(template="plotly_dark", height=300,
            showlegend=False, margin=dict(t=10,b=10,l=20,r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Interpretation
    st.markdown(f'<div class="section-hdr">{L["bridge_interp"]}</div>', unsafe_allow_html=True)
    interp_lines = []
    dom = max([(eg,L["bridge_ebitda"]),(me,L["bridge_mult"]),(dp,L["bridge_debt"])], key=lambda x:x[0])[1]
    interp_lines.append((dom, "#4f8ef7", "Dominant driver of value creation"))
    if eg>0.50:
        interp_lines.append((L["bridge_ebitda"],"#4f8ef7","Growth story — execution risk on CAGR assumptions"))
    elif eg>0.30:
        interp_lines.append((L["bridge_ebitda"],"#4f8ef7","Moderate EBITDA growth contribution"))
    else:
        interp_lines.append((L["bridge_ebitda"],"#4f8ef7","Limited organic growth — may need operational improvement"))
    if me>0.30:
        interp_lines.append((L["bridge_mult"],"#00cc88","High reliance on multiple expansion — market risk"))
    elif me>0.10:
        interp_lines.append((L["bridge_mult"],"#00cc88","Moderate multiple uplift baked in"))
    else:
        interp_lines.append((L["bridge_mult"],"#00cc88","Conservative — entry/exit multiples similar"))
    if dp>0.30:
        interp_lines.append((L["bridge_debt"],"#ffaa00","Strong deleveraging — FCF quality critical"))
    else:
        interp_lines.append((L["bridge_debt"],"#ffaa00","Moderate debt paydown contribution"))
    for lbl, col, txt in interp_lines:
        st.markdown(
            f'<div style="border-left:3px solid {col};padding:6px 12px;margin:3px 0">'
            f'<span style="color:{col};font-weight:600;font-size:.85em">{lbl}: </span>'
            f'<span style="font-size:.9em">{txt}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    brc1,brc2 = st.columns(2)
    with brc1:
        st.markdown('<div class="section-hdr">Cumulative FCF vs Debt Paydown</div>', unsafe_allow_html=True)
        yrs_x = [f"Y{y}" for y in range(1,hold_period+1)]
        cum_fcf = list(np.cumsum(results.fcf_series))
        paid_d  = [results.entry_debt - results.debt_schedule["Closing"].iloc[i]
                   for i in range(len(results.debt_schedule))]
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=yrs_x, y=cum_fcf, mode="lines+markers",
            line=dict(color="#4f8ef7",width=2.5),
            fill="tozeroy", fillcolor="rgba(79,142,247,0.08)", name="Cumul. FCF"))
        fig_cum.add_trace(go.Scatter(x=yrs_x, y=paid_d, mode="lines+markers",
            line=dict(color="#00cc88",width=2,dash="dot"), name="Debt Paid Down"))
        fig_cum.update_layout(template="plotly_dark", height=240,
            yaxis_title=axis_lbl,
            legend=dict(x=0.01,y=0.99,bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10,b=20,l=60,r=20))
        st.plotly_chart(fig_cum, use_container_width=True)
    with brc2:
        st.markdown('<div class="section-hdr">FCF Conversion Trend</div>', unsafe_allow_html=True)
        cc_series = [f/max(e,1) for f,e in zip(results.fcf_series, results.ebitda_proj)]
        fig_ccs = go.Figure()
        fig_ccs.add_trace(go.Scatter(x=yrs_x, y=[v*100 for v in cc_series],
            mode="lines+markers", line=dict(color="#ffaa00",width=2.5), showlegend=False))
        fig_ccs.add_hline(y=50, line_dash="dot", line_color="#4f8ef7", annotation_text="50% target")
        fig_ccs.update_layout(template="plotly_dark", height=240,
            yaxis=dict(title="Cash Conversion %"),
            showlegend=False, margin=dict(t=10,b=20,l=60,r=20))
        st.plotly_chart(fig_ccs, use_container_width=True)
    st.caption("Value Bridge per Rosenbaum & Pearl p.162")


# ══ TAB 7: DEAL SCREENER ═══════════════════════════════
with tab_screen:
    st.markdown('<div class="src-tag">[R&P + MPE] Multi-company LBO ranking</div>', unsafe_allow_html=True)
    st.markdown(f"### {L['screen_title']}")
    st.caption(L["screen_sub"])

    with st.expander(L["screen_add"], expanded=len(st.session_state.screener_companies)==0):
        sa1,sa2,sa3 = st.columns(3)
        with sa1:
            sc_name = st.text_input(L["screen_name"],"Company A",key="sc_name")
            sc_rev  = st.number_input(L["screen_rev"],  value=50000.0,step=1000.0,key="sc_rev")
            sc_ebi  = st.number_input(L["screen_ebitda"],value=10000.0,step=500.0, key="sc_ebi")
            sc_debt = st.number_input(L["screen_debt"],  value=15000.0,step=500.0, key="sc_debt")
        with sa2:
            sc_entry = st.number_input(L["screen_entry"],value=6.5,step=0.5,key="sc_entry")
            sc_exit  = st.number_input(L["screen_exit"], value=7.0,step=0.5,key="sc_exit")
            sc_hold  = st.number_input(L["screen_hold"], value=5,  step=1,  key="sc_hold",min_value=3,max_value=10)
        with sa3:
            sc_rate = st.number_input(L["screen_rate"],value=6.5,step=0.25,key="sc_rate")
            sc_cagr = st.number_input(L["screen_cagr"],value=4.0,step=0.5, key="sc_cagr")
            sc_eq   = st.number_input(L["screen_eq"],  value=40.0,step=5.0, key="sc_eq")
        if st.button(L["screen_add"], type="primary"):
            em = sc_ebi/max(sc_rev,1)
            ci_sc = CompanyInputs(
                revenue=sc_rev,ebitda=sc_ebi,ebit=sc_ebi*0.75,
                depreciation=sc_ebi*0.15,interest_expense=sc_debt*0.065,
                tax_rate=0.25,total_debt=sc_debt,cash=0,
                net_working_capital=sc_rev*0.10,capex=sc_rev*0.04,
                company_name=sc_name,currency_display=ccy.get("display",""),
                revenue_cagr_hist=sc_cagr/100,ebitda_margin_avg=em,
                capex_intensity=0.04,nwc_intensity=0.10,
            )
            st.session_state.screener_companies.append({
                "name":sc_name,"ci":ci_sc,
                "entry":sc_entry,"exit":sc_exit,"hold":int(sc_hold),
                "rate":sc_rate/100,"cagr":sc_cagr/100,"eq":sc_eq/100,
            })
            st.success(f"Added: {sc_name}"); st.rerun()

    if st.session_state.screener_companies:
        st.markdown(f"**{len(st.session_state.screener_companies)} companies in screener**")
        rm_idx = None
        for i,co in enumerate(st.session_state.screener_companies):
            cc1,cc2 = st.columns([5,1])
            with cc1:
                st.caption(f"{i+1}. {co['name']} — Rev {sym}{fmt_num(co['ci'].revenue, lang, sfx=sfx)} | "
                           f"EBITDA {sym}{fmt_num(co['ci'].ebitda, lang, sfx=sfx)} | "
                           f"Entry {co['entry']}x | Hold {co['hold']}yr")
            with cc2:
                if st.button("Remove",key=f"rm_{i}"): rm_idx=i
        if rm_idx is not None:
            st.session_state.screener_companies.pop(rm_idx); st.rerun()

    btn1,btn2 = st.columns([1,5])
    with btn1:
        run_screen = st.button(L["screen_run"],type="primary",
                               disabled=len(st.session_state.screener_companies)<1)
    with btn2:
        if st.button(L["screen_clear"]):
            st.session_state.screener_companies=[]; st.session_state.screener_results=[]; st.rerun()

    if run_screen:
        sr = [{"name":company_inputs.company_name+" (current)",
               "irr":results.irr,"moic":results.moic,"dscr":results.dscr_base,
               "leverage":results.entry_leverage,"cc":results.cash_conversion,
               "score":results.lbo_score,"debt_cap":results.debt_capacity_dscr,
               "entry_ev":results.entry_ev}]
        for co in st.session_state.screener_companies:
            try:
                a_sc = LBOAssumptions(
                    entry_ev_multiple=co["entry"],equity_contribution_pct=co["eq"],
                    senior_debt_rate=co["rate"],debt_amortization_years=F["debt_amort_years"],
                    exit_multiple=co["exit"],holding_period=co["hold"],
                    revenue_cagr=co["cagr"],ebitda_margin_entry=co["ci"].ebitda_margin_avg,
                    ebitda_margin_exit=co["ci"].ebitda_margin_avg*1.05,
                    max_leverage_covenant=T["max_debt_ebitda"],min_dscr_covenant=T["min_dscr"],
                )
                cia_s = _ci_args(co["ci"]); aa_s = _a_args(a_sc)
                r_sc  = _run_lbo(**cia_s, hm_tuple=None, **aa_s)
                sr.append({"name":co["name"],"irr":r_sc.irr,"moic":r_sc.moic,
                           "dscr":r_sc.dscr_base,"leverage":r_sc.entry_leverage,
                           "cc":r_sc.cash_conversion,"score":r_sc.lbo_score,
                           "debt_cap":r_sc.debt_capacity_dscr,"entry_ev":r_sc.entry_ev})
            except Exception as e:
                st.warning(f"Error {co['name']}: {e}")
        sr.sort(key=lambda x:x["score"],reverse=True)
        st.session_state.screener_results = sr

    if st.session_state.screener_results:
        sr = st.session_state.screener_results
        st.markdown(f'<div class="section-hdr">{L["screen_results"]}</div>', unsafe_allow_html=True)
        df_sr = pd.DataFrame([{
            L["screen_rank"]:i+1,"Company":r["name"],
            "Score":f"{r['score']:.0f}/100","IRR":f"{r['irr']:.1%}",
            "MOIC":f"{r['moic']:.2f}x","DSCR Y1":f"{r['dscr']:.2f}x",
            "Leverage":f"{r['leverage']:.1f}x","Cash Conv.":f"{r['cc']:.1%}",
        } for i,r in enumerate(sr)])
        st.dataframe(df_sr.set_index(L["screen_rank"]),use_container_width=True)

        fig_rank = go.Figure()
        fig_rank.add_trace(go.Bar(
            y=[r["name"] for r in sr], x=[r["score"] for r in sr],
            orientation="h",
            marker_color=["#00cc88" if r["score"]>=65 else "#ffaa00" if r["score"]>=45 else "#ff4b4b" for r in sr],
            text=[f"{r['score']:.0f}" for r in sr], textposition="outside",
            showlegend=False,
        ))
        fig_rank.add_vline(x=60,line_dash="dash",line_color="#4f8ef7",annotation_text="Target 60")
        fig_rank.update_layout(template="plotly_dark",height=max(200,len(sr)*55),
            xaxis=dict(range=[0,105],title="LBO Attractiveness Score"),
            margin=dict(t=20,b=20,l=150,r=70))
        st.plotly_chart(fig_rank,use_container_width=True)

        if len(sr)<=6:
            cats = ["IRR","DSCR","Cash Conv.","Leverage (inv)","Score"]
            fig_rd = go.Figure()
            for r in sr:
                fig_rd.add_trace(go.Scatterpolar(
                    r=[min(100,r["irr"]*200),min(100,r["dscr"]*40),
                       min(100,r["cc"]*100),max(0,(10-r["leverage"])/10*100),r["score"]],
                    theta=cats,fill="toself",name=r["name"],opacity=0.7))
            fig_rd.update_layout(template="plotly_dark",height=350,
                polar=dict(radialaxis=dict(visible=True,range=[0,100])),
                legend=dict(orientation="h",y=-0.15),
                margin=dict(t=30,b=60,l=20,r=20))
            st.plotly_chart(fig_rd,use_container_width=True)
    elif len(st.session_state.screener_companies)==0:
        st.info(L["screen_empty"])

    # ── Investment Thesis / Deal Screening ─────────────
    st.markdown("---")
    st.markdown(f'<div class="section-hdr">{L["thesis_title"]}</div>', unsafe_allow_html=True)
    th1, th2 = st.columns([3, 1])
    with th2:
        thesis_ind = st.selectbox(L["thesis_industry"], list(BENCHMARKS.keys()), key="thesis_ind")
    with th1:
        st.caption("AI-powered LBO screening memo — verdict, value creation drivers, key risks and investment conditions")

    if st.button(L["thesis_run"], type="primary"):
        import urllib.request as _ul, urllib.error as _ue, json as _uj

        bm_lo2, bm_hi2 = BENCHMARKS[thesis_ind]
        bm_m2  = (bm_lo2 + bm_hi2) / 2
        ovp2   = entry_multiple - bm_m2
        _rel   = "above" if ovp2 > 0 else "below"
        _rv    = f"{hist_metrics.revenue_volatility:.1%}" if hist_metrics else "n/a"
        _mdscr = f"{min(results.dscr_series):.2f}x" if results.dscr_series else "n/a"

        prompt = (
            "You are a private equity screening analyst.\n"
            "Assess whether this company is suitable for an LBO at early screening stage.\n"
            "Every argument must reference a specific metric from the data below.\n"
            "No general statements. No markdown. No bold. Plain text only.\n\n"
            "Output exactly these four sections, each header on its own line:\n\n"
            "VERDICT\n"
            "WHY THIS DEAL COULD WORK\n"
            "KEY RISKS\n"
            "WHAT MUST BE TRUE\n\n"
            "VERDICT: one sentence — LBO-suitable or not and why.\n"
            "WHY THIS DEAL COULD WORK: 3 numbered points each citing a specific metric.\n"
            "KEY RISKS: 3 numbered points. Analyze revenue volatility, DSCR cushion, "
            "entry leverage, cash conversion, and value creation concentration "
            "(reliance on multiple expansion vs operational growth). Each must cite the metric.\n"
            "WHAT MUST BE TRUE: 3 numbered quantified conditions required to hit target returns.\n\n"
            "--- DEAL DATA ---\n"
            f"Company: {company_inputs.company_name}\n"
            f"Industry: {thesis_ind}\n"
            f"Revenue: {sym}{fmt_num(company_inputs.revenue, lang, sfx=sfx)}\n"
            f"EBITDA Margin: {entry_margin:.1%}\n"
            f"Entry EV/EBITDA: {entry_multiple:.1f}x ({_rel} sector median by {abs(ovp2):.1f}x)\n"
            f"Revenue CAGR (hist.): {company_inputs.revenue_cagr_hist:.1%}\n"
            f"Revenue Volatility: {_rv}\n"
            f"Cash Conversion: {results.cash_conversion:.1%}\n"
            f"Entry Leverage: {results.entry_leverage:.1f}x Net Debt/EBITDA\n"
            f"Minimum DSCR (hold period): {_mdscr}\n"
            f"Base IRR: {results.irr:.1%}  |  MOIC: {results.moic:.2f}x\n"
            f"LBO Score: {results.lbo_score:.0f}/100\n"
            f"Value Creation Split: EBITDA Growth {eg:.0%} | Multiple Expansion {me:.0%} | Debt Paydown {dp:.0%}\n"
        )

        def _gcall(key, model, api_ver, text, max_tok=1500):
            body = _uj.dumps({
                "contents": [{"parts": [{"text": text}]}],
                "generationConfig": {"maxOutputTokens": max_tok, "temperature": 0.5},
            }).encode()
            req = _ul.Request(
                f"https://generativelanguage.googleapis.com/{api_ver}/{model}:generateContent?key={key}",
                data=body, headers={"Content-Type": "application/json"}, method="POST")
            with _ul.urlopen(req, timeout=30) as r:
                return _uj.loads(r.read())["candidates"][0]["content"]["parts"][0]["text"]

        def _grun(key, text, max_tok=1500):
            # Model discovery
            for av in ("v1", "v1beta"):
                try:
                    req = _ul.Request(
                        f"https://generativelanguage.googleapis.com/{av}/models?key={key}",
                        headers={"Content-Type": "application/json"}, method="GET")
                    with _ul.urlopen(req, timeout=10) as r:
                        found = [
                            (m["name"], av)
                            for m in _uj.loads(r.read()).get("models", [])
                            if "generateContent" in m.get("supportedGenerationMethods", [])
                            and "flash" in m["name"].lower()
                        ]
                        if found:
                            return _gcall(key, found[0][0], found[0][1], text, max_tok)
                except Exception:
                    pass
            # Brute-force fallback
            last_err = None
            for av in ("v1", "v1beta"):
                for mn in ("models/gemini-2.0-flash-001", "models/gemini-2.0-flash",
                           "models/gemini-1.5-flash-001", "models/gemini-1.5-flash"):
                    try:
                        return _gcall(key, mn, av, text, max_tok)
                    except _ue.HTTPError as e:
                        last_err = e
                        if e.code in (401, 403): raise
            raise last_err or RuntimeError("No Gemini model responded")

        try:
            api_key = st.secrets.get("apikey", "").strip()
            if not api_key:
                st.session_state.thesis_text = "⚠️ API key not configured — add 'apikey' to Streamlit secrets."
            else:
                st.session_state.thesis_text = _grun(api_key, prompt)
        except _ue.HTTPError as e:
            try:    msg = _uj.loads(e.read()).get("error", {}).get("message", str(e))
            except: msg = str(e)
            st.session_state.thesis_text = f"⚠️ HTTP {e.code}: {msg}"
        except Exception as e:
            st.session_state.thesis_text = f"⚠️ Error: {e}"
        st.rerun()

    # ── Display ────────────────────────────────────────────────────────────
    if st.session_state.thesis_text:
        import re as _re
        raw = st.session_state.thesis_text.strip()
        if raw.startswith("⚠️"):
            st.warning(raw)
        else:
            text = raw.replace("\\n", "\n").strip()

            # Section config: header keyword → (display label, color)
            SECTIONS = [
                ("VERDICT",                 "Verdict",                "#4f8ef7"),
                ("WHY THIS DEAL COULD WORK","Why This Deal Could Work","#00cc88"),
                ("KEY RISKS",               "Key Risks",              "#ff6b6b"),
                ("WHAT MUST BE TRUE",       "What Must Be True",      "#ffaa00"),
            ]

            # Split on the exact section headers the prompt requests
            header_pattern = r'(?m)^(VERDICT|WHY THIS DEAL COULD WORK|KEY RISKS|WHAT MUST BE TRUE)\s*:?\s*$'
            parts = _re.split(header_pattern, text)
            # parts = [pre_text, "VERDICT", content, "WHY...", content, ...]

            # Build ordered dict: header → content
            found = {}
            i = 1
            while i < len(parts) - 1:
                hdr     = parts[i].strip()
                content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                # Strip stray markdown bold from content
                content = _re.sub(r'\*\*', '', content).strip()
                found[hdr] = content
                i += 2

            if found:
                for key, label, color in SECTIONS:
                    body = found.get(key, "")
                    if not body:
                        continue
                    # Colored section header
                    st.markdown(
                        f'<div style="border-left:4px solid {color};'
                        f'padding:4px 0 4px 12px;margin:20px 0 6px 0">'
                        f'<span style="color:{color};font-weight:700;font-size:.95em">'
                        f'{label}</span></div>',
                        unsafe_allow_html=True
                    )
                    # Native st.markdown — correct contrast, full text, no clipping
                    st.markdown(body)
            else:
                # Fallback: render everything verbatim
                st.markdown(text)
        st.caption("Generated by Google Gemini Flash — analytical support only, not investment advice")

# ══ TAB 8: RISK & FLAGS ════════════════════════════════
with tab_flags:
    if red_flags:
        st.markdown(f"### 🔴 {len(red_flags)} Critical Flag(s)")
        for _,msg in red_flags:
            st.markdown(f'<div class="red-flag">{msg}</div>',unsafe_allow_html=True)
    if warn_flags:
        st.markdown(f"### ⚠️ {len(warn_flags)} Warning(s)")
        for _,msg in warn_flags:
            st.markdown(f'<div class="warn-flag">{msg}</div>',unsafe_allow_html=True)
    if not red_flags and not warn_flags:
        st.markdown(f'<div class="green-flag">{L["no_flags"]}</div>',unsafe_allow_html=True)
    if parse_warnings:
        st.markdown("### Parser Notes")
        for w in parse_warnings: st.warning(w)

    st.markdown("---")
    st.markdown(f"### 📉 {L['ds_title']}")
    st.caption(L["ds_cap"])
    dc1,dc2,dc3,dc4 = st.columns(4)
    dc1.metric(L["ds_irr"],   f"{results.downside_irr:.1%}",
               delta=f"{results.downside_irr-results.irr:.1%} vs Base")
    dc2.metric(L["ds_moic"],  f"{results.downside_moic:.2f}x",
               delta=f"{results.downside_moic-results.moic:.2f}x vs Base")
    dc3.metric(L["irr_buf"],  f"{results.irr-T['min_irr']:.1%}")
    dc4.metric(L["dscr_buf"], f"{results.dscr_base-T['min_dscr']:.2f}x")

    st.markdown("---")
    st.markdown(f"""**{L['sources']}**
- **[R&P]** Rosenbaum & Pearl - Investment Banking: Valuation, LBOs, M&A, and IPOs (2020)
- **[MPE]** Mastering Private Equity / Private Equity at Work
- **[McK]** McKinsey - Valuation (7th ed.)""")

# ── FOOTER ─────────────────────────────────────────────
st.markdown("---")
fc1, fc2 = st.columns([4,1])
with fc1:
    st.caption(f"LBO Screener v4.1 · {ccy_name} {unit} · [R&P] Rosenbaum & Pearl 2020 · [MPE] Mastering PE · [McK] McKinsey Valuation · Not investment advice")
with fc2:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        results.debt_schedule.to_excel(w, sheet_name="Debt Schedule")
        irr_hm.to_excel(w, sheet_name="IRR Heatmap")
        dscr_hm.to_excel(w, sheet_name="DSCR Heatmap")
        lev_hm.to_excel(w, sheet_name="Leverage Heatmap")
        pd.DataFrame({
            "Revenue":results.revenue_proj,"EBITDA":results.ebitda_proj,
            "FCF":results.fcf_series,"DSCR":results.dscr_series,
        },index=[f"Y{i+1}" for i in range(hold_period)]).to_excel(w,sheet_name="Projections")
        pd.DataFrame({
            "Metric":["LBO Score","Cash Conversion","EBITDA Growth %","Multiple Exp %",
                      "Debt Paydown %","Rev Volatility","Debt Cap (Simple)","Debt Cap (DSCR)"],
            "Value":[results.lbo_score,results.cash_conversion,
                     results.vc_ebitda_growth,results.vc_multiple_exp,results.vc_debt_paydown,
                     hist_metrics.revenue_volatility if hist_metrics else 0,
                     results.debt_capacity,results.debt_capacity_dscr],
        }).to_excel(w,sheet_name="Score & Bridge",index=False)
        if timeseries_df is not None and not timeseries_df.empty:
            timeseries_df.to_excel(w,sheet_name="Historical")
        if st.session_state.screener_results:
            pd.DataFrame(st.session_state.screener_results).to_excel(w,sheet_name="Deal Screener",index=False)
    out.seek(0)
    st.download_button(L["excel_exp"],data=out,
        file_name=f"LBO_{company_inputs.company_name.replace(' ','_')}_v4.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")