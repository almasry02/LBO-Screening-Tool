"""
LBO Screening Tool v3.1
[R&P] Rosenbaum & Pearl - Investment Banking (2020)
[MPE] Mastering Private Equity / Private Equity at Work
[McK] McKinsey - Valuation (7th ed.)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
from io import BytesIO
import sys, os

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from finance_engine import (
    CompanyInputs, LBOAssumptions, LBOEngine,
    SensitivityEngine, HistoricalAnalyzer, HistoricalYear, HistoricalMetrics,
)
from data_parser import (
    MoodysOrbisParser, GenericExcelParser, parse_file,
    REQUIRED_FIELDS, detect_currency_and_unit,
)

# ── PAGE CONFIG ────────────────────────────────────────
st.set_page_config(
    page_title="LBO Screening Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: original working design restored ──────────────
st.markdown("""
<style>
.metric-box {
    background:#1a1f2e; border-radius:8px; padding:14px 18px;
    border-left:4px solid #4f8ef7; margin-bottom:8px;
}
.red-flag   {background:rgba(255,75,75,0.12);border-left:4px solid #ff4b4b;border-radius:6px;padding:10px 14px;margin:6px 0;font-size:0.9em;color:#ffb3b3}
.warn-flag  {background:#2d2a1a;border-left:4px solid #ffaa00;border-radius:6px;padding:10px 14px;margin:3px 0;font-size:.9em;}
.green-flag {background:#1a2d1a;border-left:4px solid #00cc88;border-radius:6px;padding:10px 14px;margin:3px 0;font-size:.9em;}
.currency-badge {
    display:inline-block;background:#1e3a5f;border:1px solid #4f8ef7;
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
        "nav_main":        "LBO Analysis",
        "nav_settings":    "Settings",
        "data_input":      "Data Input",
        "deal_params":     "Deal Parameters",
        "upload_btn":      "Run LBO Analysis",
        "demo_info":       "Demo: Anonymized mid-market manufacturing company | tsd USD | 10-year normalized basis [R&P]",
        "lang_label":      "Language",
        "tab_hist":        "Historical Analytics",
        "tab_lbo":         "LBO Structure",
        "tab_debt":        "Debt Schedule",
        "tab_sens":        "Sensitivities",
        "tab_flags":       "Risk & Flags",
        "viable":          "LBO VIABLE",
        "critical":        "CRITICAL",
        "no_flags":        "No red flags - structure appears feasible",
        "sources":         "Sources:",
        "num_sep":         ",",
        "dec_sep":         ".",
        "hist_basis":      "Historical basis",
        "yrs":             "yrs",
        "hist_cagr":       "Hist. CAGR",
        "avg_margin":      "Avg EBITDA Margin",
        "fwd_cagr":        "Fwd CAGR",
        "normalized":      "normalized [R&P]",
        "revenue_cagr":    "Revenue CAGR",
        "avg_ebitda":      "Avg EBITDA Margin",
        "ebitda_vol":      "EBITDA Volatility",
        "fcf_conv":        "FCF Conversion",
        "capex_int":       "CapEx Intensity",
        "nwc_int":         "NWC Intensity",
        "ic_avg":          "Avg Interest Coverage",
        "norm_ebitda":     "Norm. EBITDA",
        "rev_ebitda_chart":"Revenue & EBITDA Trend",
        "margin_trend":    "Margin Trend & Volatility [MPE]",
        "norm_basis":      "Normalized LBO Basis [R&P]",
        "metric":          "Metric",
        "value":           "Value",
        "method":          "Method",
        "norm_rev_row":    "Revenue (normalized)",
        "norm_ebi_row":    "EBITDA (normalized)",
        "avg_mar_row":     "EBITDA Margin (avg)",
        "norm_cap_row":    "CapEx (normalized)",
        "nwc_del_row":     "Delta NWC p.a.",
        "curr_rev":        "Current revenue",
        "avg_m_curr":      "Avg margin x current revenue",
        "simple_avg":      "Simple avg",
        "avg_cap_int":     "Avg CapEx intensity x revenue",
        "nwc_int_delt":    "NWC intensity x rev. growth",
        "entry_struct":    "Entry Structure [R&P p.152]",
        "returns":         "Returns [R&P p.160]",
        "input_sum":       "Normalized Entry P&L",
        "fcf_proj":        "FCF Projection [McK p.167]",
        "dscr_dev":        "DSCR Development [MPE]",
        "debt_sched":      "Debt Schedule (straight-line + cash sweep) [R&P p.156]",
        "debt_wfall":      "Debt Paydown (Waterfall)",
        "headroom":        "Headroom",
        "sens_title":      "Sensitivity Analyses [R&P p.200]",
        "irr_hm":          "IRR vs Exit Multiple x Rev. CAGR",
        "dscr_hm":         "DSCR vs Leverage x Interest Rate",
        "lev_hm":          "IRR vs Exit Multiple x Equity Contribution",
        "ds_title":        "Downside Case [MPE]",
        "ds_cap":          "Stress: Revenue -10% | EBITDA Margin -200bps | Exit Multiple -1.0x",
        "irr_buf":         "IRR Buffer",
        "dscr_buf":        "DSCR Buffer",
        "floor":           "Floor",
        "warn":            "Warn",
        "hurdle":          "Hurdle",
        "used_years":      "Years used",
        "sel_years":       "Years for analysis",
        "y3":              "3 years (latest)",
        "y5":              "5 years",
        "all":             "All",
        "hist_only":       "Historical data available after Moodys upload or in Demo mode",
        "excel_exp":       "Excel Export",
        "entry_ev":        "Entry EV",
        "entry_eq":        "Entry Equity",
        "entry_debt":      "Entry Debt",
        "net_lev":         "Net Leverage",
        "debt_cap":        "Debt Capacity",
        "exit_ebitda":     "Exit EBITDA",
        "exit_ev":         "Exit EV",
        "exit_eq":         "Exit Equity",
        "base_irr":        "Base IRR",
        "base_moic":       "Base MOIC",
        "ds_irr":          "Downside IRR",
        "ds_moic":         "Downside MOIC",
        "revenue":         "Revenue",
        "ebitda":          "EBITDA",
        "ebit":            "EBIT",
        "da":              "D&A",
        "interest":        "Interest Expense",
        "tax":             "Tax Rate",
        "debt":            "Financial Debt",
        "cash":            "Cash",
        "nwc":             "NWC",
        "capex":           "CapEx",
    },
    "de": {
        "nav_main":        "LBO Analyse",
        "nav_settings":    "Einstellungen",
        "data_input":      "Dateneingabe",
        "deal_params":     "Deal-Parameter",
        "upload_btn":      "LBO-Analyse starten",
        "demo_info":       "Demo: Anonymisiertes Mittelstands-Unternehmen | tsd USD | 10 Jahre normalisierte Basis [R&P]",
        "lang_label":      "Sprache",
        "tab_hist":        "Historische Analyse",
        "tab_lbo":         "LBO-Struktur",
        "tab_debt":        "Debt Schedule",
        "tab_sens":        "Sensitivitaeten",
        "tab_flags":       "Risk & Flags",
        "viable":          "LBO-FAEHIG",
        "critical":        "KRITISCH",
        "no_flags":        "Keine Red Flags - Struktur erscheint tragfaehig",
        "sources":         "Quellen:",
        "num_sep":         ".",
        "dec_sep":         ",",
        "hist_basis":      "Historische Basis",
        "yrs":             "J.",
        "hist_cagr":       "Hist. CAGR",
        "avg_margin":      "Avg EBITDA-Marge",
        "fwd_cagr":        "Fwd CAGR",
        "normalized":      "normalisiert [R&P]",
        "revenue_cagr":    "Umsatz-CAGR",
        "avg_ebitda":      "Avg EBITDA-Marge",
        "ebitda_vol":      "EBITDA-Volatilitaet",
        "fcf_conv":        "FCF Conversion",
        "capex_int":       "CapEx Intensitaet",
        "nwc_int":         "NWC Intensitaet",
        "ic_avg":          "Avg Interest Coverage",
        "norm_ebitda":     "Norm. EBITDA",
        "rev_ebitda_chart":"Umsatz & EBITDA-Entwicklung",
        "margin_trend":    "Margin-Trend & Volatilitaet [MPE]",
        "norm_basis":      "Normalisierte LBO-Basis [R&P]",
        "metric":          "Kennzahl",
        "value":           "Wert",
        "method":          "Methode",
        "norm_rev_row":    "Umsatz (normalisiert)",
        "norm_ebi_row":    "EBITDA (normalisiert)",
        "avg_mar_row":     "EBITDA-Marge (Avg)",
        "norm_cap_row":    "CapEx (normalisiert)",
        "nwc_del_row":     "Delta NWC p.a.",
        "curr_rev":        "Aktueller Umsatz",
        "avg_m_curr":      "Avg-Marge x akt. Umsatz",
        "simple_avg":      "Einfacher Avg",
        "avg_cap_int":     "Avg CapEx-Intensitaet x Umsatz",
        "nwc_int_delt":    "NWC-Intensitaet x Rev.-Delta",
        "entry_struct":    "Entry-Struktur [R&P p.152]",
        "returns":         "Returns [R&P p.160]",
        "input_sum":       "Normalisierte Entry P&L",
        "fcf_proj":        "FCF-Projektion [McK p.167]",
        "dscr_dev":        "DSCR-Entwicklung [MPE]",
        "debt_sched":      "Tilgungsplan (Straight-line + Cash Sweep) [R&P p.156]",
        "debt_wfall":      "Schuldenabbau (Waterfall)",
        "headroom":        "Headroom",
        "sens_title":      "Sensitivitaetsanalysen [R&P p.200]",
        "irr_hm":          "IRR vs. Exit Multiple x Rev. CAGR",
        "dscr_hm":         "DSCR vs. Leverage x Zinssatz",
        "lev_hm":          "IRR vs. Exit Multiple x Equity-Anteil",
        "ds_title":        "Downside Case [MPE]",
        "ds_cap":          "Stress: Revenue -10% | EBITDA-Marge -200bps | Exit Multiple -1,0x",
        "irr_buf":         "IRR Puffer",
        "dscr_buf":        "DSCR Puffer",
        "floor":           "Floor",
        "warn":            "Warn",
        "hurdle":          "Hurdle",
        "used_years":      "Verwendete Jahre",
        "sel_years":       "Jahre fuer Analyse",
        "y3":              "3 Jahre (aktuellste)",
        "y5":              "5 Jahre",
        "all":             "Alle",
        "hist_only":       "Historische Daten nach Moodys Upload oder im Demo-Modus verfuegbar",
        "excel_exp":       "Excel Export",
        "entry_ev":        "Entry EV",
        "entry_eq":        "Entry Equity",
        "entry_debt":      "Entry Debt",
        "net_lev":         "Net Leverage",
        "debt_cap":        "Debt Capacity",
        "exit_ebitda":     "Exit EBITDA",
        "exit_ev":         "Exit EV",
        "exit_eq":         "Exit Equity",
        "base_irr":        "Base IRR",
        "base_moic":       "Base MOIC",
        "ds_irr":          "Downside IRR",
        "ds_moic":         "Downside MOIC",
        "revenue":         "Umsatz",
        "ebitda":          "EBITDA",
        "ebit":            "EBIT",
        "da":              "D&A",
        "interest":        "Zinsaufwand",
        "tax":             "Steuersatz",
        "debt":            "Finanzschulden",
        "cash":            "Cash",
        "nwc":             "NWC",
        "capex":           "CapEx",
    }
}


def fmt_num(val, lang, dec=0):
    """Format number with locale-aware separators."""
    s = f"{val:,.{dec}f}"
    if lang == "de":
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


# ── SESSION STATE ──────────────────────────────────────
for k, v in [("T", DEFAULT_T), ("F", DEFAULT_F),
              ("ccy", detect_currency_and_unit("EUR")), ("lang", "en")]:
    if k not in st.session_state:
        st.session_state[k] = v.copy() if isinstance(v, dict) else v

T    = st.session_state.T
F    = st.session_state.F
lang = st.session_state.lang
L    = LANG[lang]

# ── NAVIGATION ─────────────────────────────────────────
page = st.sidebar.radio(
    "nav",
    ["📊 " + L["nav_main"], "⚙️ " + L["nav_settings"]],
    label_visibility="collapsed"
)
in_settings = L["nav_settings"] in page

# ══════════════════════════════════════════════════════
# SETTINGS PAGE
# ══════════════════════════════════════════════════════
if in_settings:
    lang_choice = st.sidebar.selectbox(
        L["lang_label"], ["English", "Deutsch"],
        index=0 if lang == "en" else 1, key="lang_set"
    )
    new_lang = "en" if lang_choice == "English" else "de"
    if new_lang != lang:
        st.session_state.lang = new_lang
        st.rerun()

    st.title("⚙️ " + L["nav_settings"])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Returns `[R&P]`")
        T["min_irr"]   = st.number_input("Min IRR - Red Flag (%)",  value=T["min_irr"]*100,  step=0.5) / 100
        T["warn_irr"]  = st.number_input("IRR - Warning (%)",        value=T["warn_irr"]*100, step=0.5) / 100
        T["min_moic"]  = st.number_input("Min MOIC - Red Flag (x)",  value=T["min_moic"],     step=0.1)
        T["warn_moic"] = st.number_input("MOIC - Warning (x)",       value=T["warn_moic"],    step=0.1)
        st.markdown("### Margins & FCF `[McK]`")
        T["min_ebitda_margin"] = st.number_input("Min EBITDA Margin (%)", value=T["min_ebitda_margin"]*100, step=0.5) / 100
        T["min_fcf_yield"]     = st.number_input("Min FCF Yield (%)",     value=T["min_fcf_yield"]*100,     step=0.5) / 100
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
        F["equity_pct_default"] = st.number_input("Default Equity (%)", value=F["equity_pct_default"]*100, step=5.0) / 100
    with fc2:
        st.markdown("### Downside Stress `[MPE]`")
        F["downside_revenue_stress"] = st.number_input("Revenue Stress (%)",       value=F["downside_revenue_stress"]*100,  step=1.0)  / 100
        F["downside_margin_stress"]  = st.number_input("Margin Stress (bps)",      value=F["downside_margin_stress"]*10000, step=25.0) / 10000
        F["downside_exit_stress"]    = st.number_input("Exit Multiple Stress (x)", value=F["downside_exit_stress"],         step=0.25)
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Save", type="primary"):
            st.session_state.T = T; st.session_state.F = F
            st.success("Saved")
    with b2:
        if st.button("Reset to Defaults"):
            st.session_state.T = DEFAULT_T.copy(); st.session_state.F = DEFAULT_F.copy()
            st.rerun()
    st.markdown("---")
    st.markdown("""**Sources:**
- **[R&P]** Rosenbaum & Pearl - Investment Banking: Valuation, LBOs, M&A, and IPOs (2020)
- **[MPE]** Mastering Private Equity / Private Equity at Work
- **[McK]** McKinsey - Valuation (7th ed.)""")
    st.stop()

# ══════════════════════════════════════════════════════
# MAIN PAGE – SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## LBO Screener")
    st.caption("v3.1 - PE Analyst Edition")
    st.markdown("---")

    lang_choice = st.selectbox(
        L["lang_label"], ["English", "Deutsch"],
        index=0 if lang == "en" else 1, key="lang_main"
    )
    new_lang = "en" if lang_choice == "English" else "de"
    if new_lang != lang:
        st.session_state.lang = new_lang
        st.rerun()

    st.markdown(f"### {L['data_input']}")
    input_mode = st.radio(
        "mode",
        ["Moodys Orbis / Excel", "Manual Input", "Demo"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(f"### {L['deal_params']}")
    entry_multiple = st.slider("Entry EV/EBITDA",       4.0, 12.0, 6.5, 0.5)
    equity_pct     = st.slider("Equity (%)",             20,  60,   40,  5) / 100
    debt_rate      = st.slider("Senior Debt Rate (%)",   3.0, 12.0, 6.5, 0.25) / 100
    exit_multiple  = st.slider("Exit EV/EBITDA",        4.0, 14.0,  7.0, 0.5)
    hold_period    = st.slider("Hold Period (yrs)",       3,    7,    5)
    margin_exit    = st.slider("Exit EBITDA Margin (%)", 5.0, 40.0, 22.0, 0.5) / 100
    st.markdown("---")
    st.caption("Hist. CAGR auto-applied as forward CAGR")
    override_cagr = st.checkbox("Override CAGR", False)
    manual_cagr   = st.slider("Fwd Rev. CAGR (%)", 0.0, 20.0, 4.0, 0.5) / 100 if override_cagr else None

# ── DATA INPUT ─────────────────────────────────────────
company_inputs  = None
hist_metrics    = None
hist_years_used = []
parse_warnings  = []
timeseries_df   = None
ccy             = st.session_state.ccy

# ── DEMO (fully anonymized) ────────────────────────────
if input_mode == "Demo":
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
    ccy = detect_currency_and_unit("tsd USD"); st.session_state.ccy = ccy
    analyzer        = HistoricalAnalyzer(demo_hist)
    hist_metrics    = analyzer.compute()
    hist_years_used = [y.year for y in demo_hist]
    latest          = demo_hist[-1]
    company_inputs  = CompanyInputs(
        revenue             = hist_metrics.normalized_revenue,
        ebitda              = hist_metrics.normalized_ebitda,
        ebit                = hist_metrics.normalized_ebitda - latest.depreciation,
        depreciation        = latest.depreciation,
        interest_expense    = latest.interest_expense,
        tax_rate            = latest.tax_rate,
        total_debt          = latest.total_debt,
        cash                = latest.cash,
        net_working_capital = latest.net_working_capital,
        capex               = hist_metrics.normalized_capex,
        company_name        = "Precision Industries Corp.",
        currency_display    = ccy["display"],
        revenue_cagr_hist   = hist_metrics.revenue_cagr,
        ebitda_margin_avg   = hist_metrics.ebitda_margin_avg,
        capex_intensity     = hist_metrics.capex_intensity_avg,
        nwc_intensity       = hist_metrics.nwc_intensity_avg,
    )
    # Sorted ascending: oldest left, newest right
    timeseries_df = pd.DataFrame([{
        "Revenue": y.revenue, "EBITDA": y.ebitda, "EBIT": y.ebit,
        "D&A": y.depreciation, "Interest": y.interest_expense,
        "Net Income": y.net_income, "Cash": y.cash,
        "Total Debt": y.total_debt, "NWC": y.net_working_capital,
        "CapEx": y.capex, "EBITDA Margin": y.ebitda_margin,
    } for y in demo_hist], index=[y.year for y in demo_hist]).sort_index(ascending=True)
    st.info(L["demo_info"])

# ── MANUAL INPUT ───────────────────────────────────────
elif input_mode == "Manual Input":
    unit_sel = st.selectbox("Unit", ["tsd EUR", "tsd USD", "Mio EUR", "Mio USD"])
    ccy = detect_currency_and_unit(unit_sel); st.session_state.ccy = ccy
    co_name = st.text_input("Company Name", "Target Co.")
    c1, c2, c3 = st.columns(3)
    with c1:
        rev    = st.number_input("Revenue",  value=56800.0, step=100.0)
        ebitda = st.number_input("EBITDA",   value=11900.0, step=100.0)
        ebit   = st.number_input("EBIT",     value=8900.0,  step=100.0)
        dep    = st.number_input("D&A",      value=3000.0,  step=100.0)
    with c2:
        interest = st.number_input("Interest Expense", value=1500.0, step=100.0)
        tax_rate = st.number_input("Tax Rate (%)",     value=26.0,   step=0.5) / 100
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
        revenue_cagr_hist=0.04,
        ebitda_margin_avg=ebitda/rev if rev else 0.20,
        capex_intensity=capex/rev if rev else 0.04,
        nwc_intensity=nwc/rev if rev else 0.10,
    )

# ── FILE UPLOAD ────────────────────────────────────────
elif input_mode == "Moodys Orbis / Excel":
    uploaded = st.file_uploader("Moodys Orbis 4-Sheet or Standard Excel", type=["xlsx","xls"])
    if uploaded:
        with st.spinner("Parsing..."):
            parser, is_moodys = parse_file(uploaded)
        if is_moodys:
            p = parser; ccy = p.currency_info; st.session_state.ccy = ccy
            n_years = len(p.years)
            st.success(f"Moodys Orbis | {n_years} yrs | {min(p.years) if p.years else '?'}-{max(p.years) if p.years else '?'} | {ccy['display']}")
            ts = p.get_timeseries_df()
            if not ts.empty:
                timeseries_df = ts.sort_index(ascending=True)
                with st.expander("Time Series", expanded=False):
                    st.dataframe(ts.style.format(lambda x: f"{x:,.0f}" if isinstance(x,(float,int)) and not pd.isna(x) else "n/a"), use_container_width=True)
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
            if st.button(L["upload_btn"], type="primary"):
                hist_year_objs  = p.build_historical_years(selected_years)
                if len(hist_year_objs) < 2:
                    st.error("Need at least 2 years of data"); st.stop()
                hist_metrics    = HistoricalAnalyzer(hist_year_objs).compute()
                hist_years_used = [y.year for y in hist_year_objs]
                company_inputs, parse_warnings = p.build_company_inputs(hist_metrics, hist_year_objs[-1])
                st.session_state.ccy = ccy
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
        st.info("Upload a Moodys Orbis Excel file or switch to Demo mode")

# ── GATE ───────────────────────────────────────────────
ccy      = st.session_state.ccy
sym      = ccy.get("symbol", "")
unit     = ccy.get("unit_label", "")
ccy_name = ccy.get("currency", "")

if company_inputs is None:
    st.markdown("## Select a data input mode")
    st.markdown("""| Feature | Detail |
|---|---|
| Moodys Orbis | 4-sheet export auto-detected, up to 10 years |
| Historical Analytics | CAGR, Avg Margin, Volatility [R&P+McK] |
| Normalized Basis | Scrubbed EBITDA as LBO entry basis [R&P] |
| 3 Heatmaps | IRR, DSCR, Leverage [R&P] |
| Downside Case | Stress test [MPE] |
| Settings | Custom thresholds & formulas |""")
    st.stop()

# ══════════════════════════════════════════════════════
# ENGINE – re-runs automatically on every slider change
# (Streamlit re-runs full script on any widget interaction)
# ══════════════════════════════════════════════════════
fwd_cagr     = manual_cagr if override_cagr else company_inputs.revenue_cagr_hist
entry_margin = company_inputs.ebitda / company_inputs.revenue if company_inputs.revenue else 0.20

assumptions = LBOAssumptions(
    entry_ev_multiple       = entry_multiple,
    equity_contribution_pct = equity_pct,
    senior_debt_rate        = debt_rate,
    debt_amortization_years = F["debt_amort_years"],
    exit_multiple           = exit_multiple,
    holding_period          = hold_period,
    revenue_cagr            = fwd_cagr,
    ebitda_margin_entry     = entry_margin,
    ebitda_margin_exit      = margin_exit,
    max_leverage_covenant   = T["max_debt_ebitda"],
    min_dscr_covenant       = T["min_dscr"],
)
results = LBOEngine(company_inputs, assumptions, hist_metrics).run()

# ── FLAGS ──────────────────────────────────────────────
def evaluate_flags(res, ci, T, hm):
    flags = []
    def f(cond, lvl, msg):
        if cond: flags.append((lvl, msg))
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
    f(ci.ebitda / max(ci.revenue, 1) < T["min_ebitda_margin"],
      "warn", f"EBITDA Margin {ci.ebitda/ci.revenue:.1%} < {T['min_ebitda_margin']:.0%}")
    f(res.fcf_yield < T["min_fcf_yield"],
      "warn", f"FCF Yield {res.fcf_yield:.1%} < {T['min_fcf_yield']:.0%}")
    f(res.interest_coverage < T["min_interest_coverage"],
      "warn", f"Interest Coverage {res.interest_coverage:.1f}x < {T['min_interest_coverage']}x")
    if hm and hm.ebitda_volatility > 0.08:
        flags.append(("warn", f"EBITDA Margin Volatility {hm.ebitda_volatility:.1%} - cyclical profile  [MPE]"))
    return flags

all_flags  = evaluate_flags(results, company_inputs, T, hist_metrics)
red_flags  = [(l, m) for l, m in all_flags if l == "red"]
warn_flags = [(l, m) for l, m in all_flags if l == "warn"]
is_viable  = len(red_flags) == 0 and results.irr >= T["min_irr"]

# ── SENSITIVITIES ──────────────────────────────────────
sens    = SensitivityEngine(company_inputs, assumptions)
irr_hm  = sens.irr_heatmap([5.0,5.5,6.0,6.5,7.0,7.5,8.0,9.0,10.0], [0.0,0.02,0.04,0.06,0.08,0.10,0.12])
dscr_hm = sens.dscr_heatmap([0.04,0.055,0.065,0.08,0.095,0.11], [3.0,4.0,5.0,5.5,6.0,6.5,7.0])
lev_hm  = sens.leverage_irr_heatmap([5.0,6.0,7.0,8.0,9.0], [0.25,0.35,0.40,0.50,0.60])

# ── HEADER ─────────────────────────────────────────────
htitle, hbadge = st.columns([3, 1])
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

viable_color = "#00cc88" if is_viable else "#ff4b4b"
viable_label = ("✅ " if is_viable else "❌ ") + L["viable" if is_viable else "critical"]

with hbadge:
    st.markdown(
        f'<div style="text-align:right;margin-top:12px">'
        f'<span style="background:{viable_color}22;border:1px solid {viable_color};'
        f'border-radius:8px;padding:6px 14px;color:{viable_color};font-weight:700">'
        f'{viable_label}</span><br>'
        f'<span style="font-size:.8em;color:#aaa;margin-top:4px;display:block">'
        f'🔴 {len(red_flags)} · ⚠️ {len(warn_flags)}</span></div>',
        unsafe_allow_html=True
    )
st.markdown("---")

# ── KPI ROWS ───────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
for col, label, val in [
    (k1, "Entry EV",       f"{sym}{fmt_num(results.entry_ev, lang)}"),
    (k2, "Entry Leverage", f"{results.entry_leverage:.1f}x"),
    (k3, "DSCR Y1",        f"{results.dscr_base:.2f}x"),
    (k4, "Base IRR",       f"{results.irr:.1%}"),
    (k5, "MOIC",           f"{results.moic:.2f}x"),
    (k6, "FCF Yield Y1",   f"{results.fcf_yield:.1%}"),
]:
    with col: st.metric(label, val)

r2a,r2b,r2c,r2d,r2e,r2f = st.columns(6)
for col, label, val in [
    (r2a, "Downside IRR",  f"{results.downside_irr:.1%}"),
    (r2b, "Downside MOIC", f"{results.downside_moic:.2f}x"),
    (r2c, "Exit EV",       f"{sym}{fmt_num(results.exit_ev, lang)}"),
    (r2d, "Exit Equity",   f"{sym}{fmt_num(results.exit_equity, lang)}"),
    (r2e, "Interest Cov.", f"{results.interest_coverage:.1f}x"),
    (r2f, "Debt Capacity", f"{sym}{fmt_num(results.debt_capacity, lang)}"),
]:
    with col: st.metric(label, val)

st.markdown("---")

# ── TABS ───────────────────────────────────────────────
tab_hist, tab_lbo, tab_debt, tab_sens, tab_flags = st.tabs([
    "📈 " + L["tab_hist"],
    "🏦 " + L["tab_lbo"],
    "📅 " + L["tab_debt"],
    "🗺️ " + L["tab_sens"],
    "🚩 " + L["tab_flags"],
])

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
        hc5.metric(L["capex_int"],  f"{h.capex_intensity_avg:.1%}")
        hc6.metric(L["nwc_int"],    f"{h.nwc_intensity_avg:.1%}")
        hc7.metric(L["ic_avg"],     f"{h.interest_coverage_avg:.1f}x")
        hc8.metric(L["norm_ebitda"],f"{sym}{fmt_num(h.normalized_ebitda, lang)}")

        if timeseries_df is not None and not timeseries_df.empty:
            # Always sort oldest→newest (left→right) for both charts
            df_sorted = timeseries_df.sort_index(ascending=True)
            yidx = [str(y) for y in df_sorted.index]

            st.markdown(f'<div class="section-hdr">{L["rev_ebitda_chart"]}</div>', unsafe_allow_html=True)
            fig = go.Figure()
            if "Revenue" in df_sorted.columns:
                fig.add_trace(go.Bar(x=yidx, y=list(df_sorted["Revenue"]),
                    name="Revenue", marker_color="#4f8ef7", opacity=0.7))
            if "EBITDA" in df_sorted.columns:
                fig.add_trace(go.Scatter(x=yidx, y=list(df_sorted["EBITDA"]),
                    name="EBITDA", mode="lines+markers",
                    line=dict(color="#00cc88", width=2.5)))
            if "EBITDA Margin" in df_sorted.columns:
                fig.add_trace(go.Scatter(x=yidx, y=list(df_sorted["EBITDA Margin"]),
                    name="EBITDA Margin %", mode="lines+markers",
                    line=dict(color="#ffaa00", width=1.5, dash="dot"), yaxis="y2"))
            fig.update_layout(
                template="plotly_dark", height=320, barmode="group",
                yaxis=dict(title=f"{sym} {unit}"),
                yaxis2=dict(title="Margin %", overlaying="y", side="right", tickformat=".0%"),
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=20, b=20, l=60, r=60),
            )
            st.plotly_chart(fig, use_container_width=True)

            if h.margin_series:
                st.markdown(f'<div class="section-hdr">{L["margin_trend"]}</div>', unsafe_allow_html=True)
                # Sort oldest→newest explicitly
                margin_pairs = sorted(zip(h.years_used, h.margin_series), key=lambda x: x[0])
                m_yrs  = [str(y) for y, _ in margin_pairs]
                m_vals = [v        for _, v in margin_pairs]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=m_yrs, y=m_vals,
                    mode="lines+markers", line=dict(color="#4f8ef7", width=2),
                    fill="tozeroy", fillcolor="rgba(79,142,247,.08)"))
                fig2.add_hline(y=h.ebitda_margin_avg, line_dash="dash", line_color="#00cc88",
                    annotation_text=f"Avg {h.ebitda_margin_avg:.1%}")
                fig2.add_hline(y=T["min_ebitda_margin"], line_dash="dot", line_color="#ff4b4b",
                    annotation_text=f"Min {T['min_ebitda_margin']:.0%}")
                fig2.update_layout(
                    template="plotly_dark", height=200,
                    yaxis=dict(tickformat=".0%"),
                    margin=dict(t=20, b=20, l=50, r=20),
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f'<div class="section-hdr">{L["norm_basis"]}</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            L["metric"]: [L["norm_rev_row"], L["norm_ebi_row"], L["avg_mar_row"],
                          L["norm_cap_row"], L["nwc_del_row"]],
            L["value"]:  [f"{sym}{fmt_num(h.normalized_revenue, lang)}",
                          f"{sym}{fmt_num(h.normalized_ebitda, lang)}",
                          f"{h.ebitda_margin_avg:.1%}",
                          f"{sym}{fmt_num(h.normalized_capex, lang)}",
                          f"{sym}{fmt_num(h.normalized_nwc_delta, lang)}"],
            L["method"]: [L["curr_rev"], L["avg_m_curr"],
                          f"{L['simple_avg']} ({len(h.years_used)} {L['yrs']})",
                          L["avg_cap_int"], L["nwc_int_delt"]],
        }).set_index(L["metric"]), use_container_width=True)
        st.caption("Normalization per Rosenbaum & Pearl Ch.4 - Scrubbed EBITDA as LBO entry basis")
    else:
        st.info(L["hist_only"])

# ══ TAB 2: LBO STRUCTURE ═══════════════════════════════
with tab_lbo:
    st.markdown('<div class="src-tag">[R&P Ch.4-5] LBO structure · FCF projection · Returns</div>', unsafe_allow_html=True)
    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown(f'<div class="section-hdr">{L["entry_struct"]}</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "": [L["entry_ev"], L["entry_eq"], L["entry_debt"], L["net_lev"], L["debt_cap"]],
            f"{sym} {unit}": [
                f"{sym}{fmt_num(results.entry_ev, lang)}",
                f"{sym}{fmt_num(results.entry_equity, lang)} ({equity_pct:.0%})",
                f"{sym}{fmt_num(results.entry_debt, lang)} ({1-equity_pct:.0%})",
                f"{results.entry_leverage:.1f}x Net Debt/EBITDA",
                f"{sym}{fmt_num(results.debt_capacity, lang)} ({T['max_debt_ebitda']}x EBITDA)",
            ]
        }).set_index(""), use_container_width=True)

        st.markdown(f'<div class="section-hdr">{L["returns"]}</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "": [L["exit_ebitda"], L["exit_ev"], L["exit_eq"],
                 L["base_irr"], L["base_moic"], L["ds_irr"], L["ds_moic"]],
            "Value": [
                f"{sym}{fmt_num(results.exit_ebitda, lang)}",
                f"{sym}{fmt_num(results.exit_ev, lang)}",
                f"{sym}{fmt_num(results.exit_equity, lang)}",
                f"{results.irr:.1%}", f"{results.moic:.2f}x",
                f"{results.downside_irr:.1%}", f"{results.downside_moic:.2f}x",
            ]
        }).set_index(""), use_container_width=True)

        st.markdown(f'<div class="section-hdr">{L["input_sum"]}</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            L["metric"]: [L["revenue"], L["ebitda"], L["ebit"], L["da"],
                          L["interest"], L["tax"], L["debt"], L["cash"], L["nwc"], L["capex"]],
            f"{sym} {unit}": [
                fmt_num(company_inputs.revenue, lang),
                f"{fmt_num(company_inputs.ebitda, lang)} ({entry_margin:.1%})",
                fmt_num(company_inputs.ebit, lang),
                fmt_num(company_inputs.depreciation, lang),
                fmt_num(company_inputs.interest_expense, lang),
                f"{company_inputs.tax_rate:.1%}",
                fmt_num(company_inputs.total_debt, lang),
                fmt_num(company_inputs.cash, lang),
                fmt_num(company_inputs.net_working_capital, lang),
                fmt_num(company_inputs.capex, lang),
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
            mode="lines+markers", name="FCF", line=dict(color="#ffaa00", width=2.5)))
        fig_fcf.update_layout(template="plotly_dark", height=270, barmode="group",
            yaxis_title=f"{sym} {unit}",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=20, l=60, r=20))
        st.plotly_chart(fig_fcf, use_container_width=True)
        st.caption("FCF = NOPAT + D&A - CapEx - delta NWC  [McK p.163]")

        st.markdown(f'<div class="section-hdr">{L["dscr_dev"]}</div>', unsafe_allow_html=True)
        fig_dscr = go.Figure()
        fig_dscr.add_trace(go.Scatter(x=yrs_x, y=results.dscr_series,
            mode="lines+markers", line=dict(color="#4f8ef7", width=2.5), name="DSCR"))
        fig_dscr.add_hline(y=T["min_dscr"], line_dash="dash", line_color="#ff4b4b",
            annotation_text=f"{L['floor']} {T['min_dscr']}x")
        fig_dscr.add_hline(y=T["warn_dscr"], line_dash="dot", line_color="#ffaa00",
            annotation_text=f"{L['warn']} {T['warn_dscr']}x")
        fig_dscr.update_layout(template="plotly_dark", height=210,
            margin=dict(t=10, b=20, l=50, r=20), showlegend=False)
        st.plotly_chart(fig_dscr, use_container_width=True)

# ══ TAB 3: DEBT SCHEDULE ═══════════════════════════════
with tab_debt:
    st.markdown(f'<div class="src-tag">{L["debt_sched"]}</div>', unsafe_allow_html=True)
    ds = results.debt_schedule
    st.dataframe(ds.style.format({
        "Opening": "{:,.1f}", "Interest": "{:,.1f}", "Amortization": "{:,.1f}",
        "Cash Sweep": "{:,.1f}", "Closing": "{:,.1f}", "Coverage": "{:.2f}x",
    }).background_gradient(subset=["Closing"], cmap="RdYlGn"), use_container_width=True)

    fig_wf = go.Figure(go.Waterfall(
        x=[f"Y{y}" for y in ds.index],
        y=[-row["Amortization"] for _, row in ds.iterrows()],
        base=results.entry_debt, measure=["relative"] * len(ds),
        connector=dict(line=dict(color="#2a2f3e")),
        decreasing=dict(marker_color="#00cc88"),
    ))
    fig_wf.update_layout(template="plotly_dark", height=250, title=L["debt_wfall"],
        yaxis_title=f"{sym} {unit}", margin=dict(t=40, b=20, l=60, r=20))
    st.plotly_chart(fig_wf, use_container_width=True)

    dc1, dc2, dc3 = st.columns(3)
    dc1.metric(f"Debt Capacity ({T['max_debt_ebitda']}x)", f"{sym}{fmt_num(results.debt_capacity, lang)}")
    dc2.metric("Entry Debt", f"{sym}{fmt_num(results.entry_debt, lang)}")
    hw = results.debt_capacity - results.entry_debt
    dc3.metric(L["headroom"], f"{sym}{fmt_num(hw, lang)}", delta="OK" if hw > 0 else "Exceeded")

# ══ TAB 4: SENSITIVITIES ═══════════════════════════════
with tab_sens:
    st.markdown(f'<div class="src-tag">[R&P p.200] {L["sens_title"]}</div>', unsafe_allow_html=True)

    def hm_fig(df, title, fmt_str, zmid, note=""):
        vals = df.values.astype(float)
        cs = [[0.0,"#ff4b4b"],[0.35,"#ffaa00"],[0.55,"#4f8ef7"],[1.0,"#00cc88"]]
        fig = go.Figure(go.Heatmap(
            z=vals, x=df.columns.tolist(), y=df.index.tolist(),
            colorscale=cs, zmid=zmid,
            text=[[fmt_str.format(v) for v in row] for row in vals],
            texttemplate="%{text}", colorbar=dict(thickness=12),
        ))
        if note:
            fig.add_annotation(text=note, xref="paper", yref="paper", x=0.01, y=-0.14,
                showarrow=False, font=dict(color="#6e7681", size=9))
        fig.update_layout(template="plotly_dark", height=370, title=title,
            margin=dict(t=40, b=55, l=100, r=20))
        return fig

    s1, s2 = st.columns(2)
    with s1:
        st.plotly_chart(hm_fig(irr_hm, L["irr_hm"], "{:.1f}%",
            T["min_irr"]*100, f"{L['hurdle']} {T['min_irr']:.0%}"), use_container_width=True)
    with s2:
        st.plotly_chart(hm_fig(dscr_hm, L["dscr_hm"], "{:.2f}x",
            T["warn_dscr"], f"{L['floor']} {T['min_dscr']}x"), use_container_width=True)
    st.markdown(f'<div class="section-hdr">{L["lev_hm"]} [R&P]</div>', unsafe_allow_html=True)
    st.plotly_chart(hm_fig(lev_hm, L["lev_hm"], "{:.1f}%", T["min_irr"]*100), use_container_width=True)
    st.caption(f"Thresholds: IRR >= {T['min_irr']:.0%} · DSCR >= {T['min_dscr']}x · Leverage <= {T['max_entry_leverage']}x")

# ══ TAB 5: RISK & FLAGS ════════════════════════════════
with tab_flags:
    if red_flags:
        st.markdown(f"### 🔴 {len(red_flags)} Critical Flag(s)")
        for _, msg in red_flags:
            st.markdown(f'<div class="red-flag">{msg}</div>', unsafe_allow_html=True)
    if warn_flags:
        st.markdown(f"### ⚠️ {len(warn_flags)} Warning(s)")
        for _, msg in warn_flags:
            st.markdown(f'<div class="warn-flag">{msg}</div>', unsafe_allow_html=True)
    if not red_flags and not warn_flags:
        st.markdown(f'<div class="green-flag">{L["no_flags"]}</div>', unsafe_allow_html=True)
    if parse_warnings:
        st.markdown("### Parser Notes")
        for w in parse_warnings: st.warning(w)

    st.markdown("---")
    st.markdown(f"### 📉 {L['ds_title']}")
    st.caption(L["ds_cap"])
    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric(L["ds_irr"],   f"{results.downside_irr:.1%}",
               delta=f"{(results.downside_irr - results.irr):.1%} vs. Base")
    dc2.metric(L["ds_moic"],  f"{results.downside_moic:.2f}x",
               delta=f"{(results.downside_moic - results.moic):.2f}x vs. Base")
    dc3.metric(L["irr_buf"],  f"{(results.irr - T['min_irr']):.1%}",
               help="Base IRR minus hurdle")
    dc4.metric(L["dscr_buf"], f"{(results.dscr_base - T['min_dscr']):.2f}x",
               help="DSCR Y1 minus covenant floor")

    st.markdown("---")
    st.markdown(f"""**{L['sources']}**
- **[R&P]** Rosenbaum & Pearl - Investment Banking: Valuation, LBOs, M&A, and IPOs (2020)
- **[MPE]** Mastering Private Equity / Private Equity at Work
- **[McK]** McKinsey - Valuation (7th ed.)""")

# ── FOOTER ─────────────────────────────────────────────
st.markdown("---")
fc1, fc2 = st.columns([4, 1])
with fc1:
    st.caption(
        f"LBO Screener v3.1 · {ccy_name} {unit} · "
        f"[R&P] Rosenbaum & Pearl 2020 · [MPE] Mastering PE · [McK] McKinsey Valuation · "
        f"Not investment advice"
    )
with fc2:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        results.debt_schedule.to_excel(w, sheet_name="Debt Schedule")
        irr_hm.to_excel(w, sheet_name="IRR Heatmap")
        dscr_hm.to_excel(w, sheet_name="DSCR Heatmap")
        lev_hm.to_excel(w, sheet_name="Leverage Heatmap")
        pd.DataFrame({
            "Revenue": results.revenue_proj, "EBITDA": results.ebitda_proj,
            "FCF": results.fcf_series, "DSCR": results.dscr_series,
        }, index=[f"Y{i+1}" for i in range(hold_period)]).to_excel(w, sheet_name="Projections")
        if timeseries_df is not None and not timeseries_df.empty:
            timeseries_df.to_excel(w, sheet_name="Historical")
    out.seek(0)
    st.download_button(
        L["excel_exp"], data=out,
        file_name=f"LBO_{company_inputs.company_name.replace(' ','_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )