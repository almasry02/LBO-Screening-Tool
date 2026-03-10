"""
LBO Screening Tool – Phase 1 MVP v2
Streamlit App | Moodys Orbis + Generic Excel | Currency-aware | Configurable Thresholds
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import sys, os

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from finance_engine import CompanyInputs, LBOAssumptions, LBOEngine, SensitivityEngine
from data_parser import (
    MoodysOrbisParser, GenericExcelParser, parse_file,
    REQUIRED_FIELDS, generate_sample_excel, detect_currency_and_unit
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="LBO Screening Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-box {
    background:#1a1f2e; border-radius:8px; padding:14px 18px;
    border-left:4px solid #4f8ef7; margin-bottom:8px;
}
.red-flag {background: rgba(255, 75, 75, 0.12); border-left: 4px solid #ff4b4b; border-radius: 6px; padding: 10px 14px; margin: 6px 0; font-size: 0.9em; color: #ffb3b3;}
.green-flag { background:#1a2d1a; border-left:4px solid #00cc88; border-radius:6px; padding:10px 14px; margin:3px 0; font-size:.9em; }
.warn-flag  { background:#2d2a1a; border-left:4px solid #ffaa00; border-radius:6px; padding:10px 14px; margin:3px 0; font-size:.9em; }
.currency-badge {
    display:inline-block; background:#1e3a5f; border:1px solid #4f8ef7;
    border-radius:20px; padding:3px 12px; font-size:.85em; color:#4f8ef7;
    font-weight:600; margin-left:10px;
}
.section-hdr { font-size:1.05em; font-weight:600; color:#4f8ef7;
    margin:18px 0 6px 0; padding-bottom:4px; border-bottom:1px solid #2a2f3e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DEFAULT THRESHOLDS (scientific basis: Rosenbaum & Pearl)
# ─────────────────────────────────────────────

DEFAULT_THRESHOLDS = {
    "min_irr":              0.20,   # Rosenbaum & Pearl: typical PE hurdle 20%+
    "min_moic":             2.0,    # Standard 2x MOIC floor
    "max_entry_leverage":   6.0,    # Net Debt/EBITDA entry cap
    "warn_entry_leverage":  5.0,    # Warning level
    "min_dscr":             1.20,   # DSCR covenant floor
    "warn_dscr":            1.35,   # DSCR warning zone
    "min_ebitda_margin":    0.08,   # 8% minimum viable margin
    "min_fcf_yield":        0.04,   # 4% FCF yield on entry EV
    "max_debt_ebitda":      5.5,    # Debt capacity ceiling
}

DEFAULT_FORMULAS = {
    "fcf_nwc_build_pct":    0.01,   # 1% of revenue as NWC build per year
    "debt_amort_years":     7,      # Standard 7-year straight-line
    "equity_pct_default":   0.40,   # 40% equity contribution
}

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────

if "thresholds" not in st.session_state:
    st.session_state.thresholds = DEFAULT_THRESHOLDS.copy()
if "formulas" not in st.session_state:
    st.session_state.formulas = DEFAULT_FORMULAS.copy()
if "currency_info" not in st.session_state:
    st.session_state.currency_info = detect_currency_and_unit("EUR")
if "display_unit" not in st.session_state:
    st.session_state.display_unit = "original"  # "original" | "millions"

T = st.session_state.thresholds
F = st.session_state.formulas

# ─────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────

page = st.sidebar.radio(
    "Navigation",
    ["📊 LBO Analyse", "⚙️ Einstellungen / Thresholds"],
    label_visibility="collapsed"
)

# ═════════════════════════════════════════════
# PAGE: SETTINGS
# ═════════════════════════════════════════════

if page == "⚙️ Einstellungen / Thresholds":
    st.title("⚙️ Einstellungen & Schwellenwerte")
    st.markdown("""
    Alle Defaults basieren auf **Rosenbaum & Pearl – Investment Banking (Wiley Finance)**,
    dem Standardwerk für LBO-Modellierung in der Praxis.
    Passe sie an deine unternehmensinternen Richtlinien an.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📐 Return-Schwellen")
        T["min_irr"] = st.number_input(
            "Mindest-IRR (%)",
            value=T["min_irr"]*100, min_value=0.0, max_value=100.0, step=0.5,
            help="Rosenbaum & Pearl: PE Hurdle Rate typisch 20–25%"
        ) / 100
        T["min_moic"] = st.number_input(
            "Mindest-MOIC (x)",
            value=T["min_moic"], min_value=0.5, max_value=10.0, step=0.1,
            help="Standard: 2.0x–3.0x über 5 Jahre Haltedauer"
        )

        st.markdown("### 📊 FCF & Margin")
        T["min_ebitda_margin"] = st.number_input(
            "Min. EBITDA-Marge (%)",
            value=T["min_ebitda_margin"]*100, min_value=0.0, max_value=100.0, step=0.5,
            help="Unter 8%: zu wenig Puffer für Debt Service"
        ) / 100
        T["min_fcf_yield"] = st.number_input(
            "Min. FCF Yield auf Entry EV (%)",
            value=T["min_fcf_yield"]*100, min_value=0.0, max_value=50.0, step=0.5,
        ) / 100

    with col2:
        st.markdown("### 🏦 Leverage & DSCR")
        T["max_entry_leverage"] = st.number_input(
            "Max. Entry Leverage – Red Flag (x)",
            value=T["max_entry_leverage"], min_value=1.0, max_value=15.0, step=0.25,
            help="Net Debt / EBITDA; über diesem Wert → Red Flag"
        )
        T["warn_entry_leverage"] = st.number_input(
            "Entry Leverage – Warnstufe (x)",
            value=T["warn_entry_leverage"], min_value=1.0, max_value=15.0, step=0.25,
        )
        T["min_dscr"] = st.number_input(
            "Min. DSCR – Covenant Floor (x)",
            value=T["min_dscr"], min_value=0.5, max_value=5.0, step=0.05,
            help="Unter diesem Wert → Covenant Breach → Red Flag"
        )
        T["warn_dscr"] = st.number_input(
            "DSCR – Warnstufe (x)",
            value=T["warn_dscr"], min_value=0.5, max_value=5.0, step=0.05,
        )
        T["max_debt_ebitda"] = st.number_input(
            "Max. Debt Capacity (x EBITDA)",
            value=T["max_debt_ebitda"], min_value=1.0, max_value=12.0, step=0.25,
            help="Maximale Verschuldungskapazität für Debt Capacity Berechnung"
        )

    st.markdown("---")
    st.markdown("### 🔧 Formel-Parameter")
    col3, col4 = st.columns(2)
    with col3:
        F["fcf_nwc_build_pct"] = st.number_input(
            "NWC-Aufbau (% Umsatz p.a.)",
            value=F["fcf_nwc_build_pct"]*100, min_value=0.0, max_value=10.0, step=0.1,
            help="Jährlicher NWC-Aufbau als % des Umsatzes (FCF-Abzug)"
        ) / 100
        F["debt_amort_years"] = int(st.number_input(
            "Tilgungszeitraum Senior Debt (Jahre)",
            value=float(F["debt_amort_years"]), min_value=3.0, max_value=15.0, step=1.0,
            help="Straight-line Amortisation über diesen Zeitraum"
        ))
    with col4:
        F["equity_pct_default"] = st.number_input(
            "Standard Equity-Anteil (%)",
            value=F["equity_pct_default"]*100, min_value=10.0, max_value=80.0, step=5.0,
        ) / 100

    st.markdown("---")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("💾 Einstellungen speichern", type="primary"):
            st.session_state.thresholds = T
            st.session_state.formulas = F
            st.success("✅ Gespeichert – Einstellungen aktiv bis zum Ende der Session")
    with col_r2:
        if st.button("🔄 Auf Defaults zurücksetzen"):
            st.session_state.thresholds = DEFAULT_THRESHOLDS.copy()
            st.session_state.formulas = DEFAULT_FORMULAS.copy()
            st.rerun()

    st.markdown("---")
    st.markdown("""
    **📚 Wissenschaftliche Grundlage dieser Defaults:**

    - **Rosenbaum & Pearl – Investment Banking** (Wiley Finance, 3rd ed.): LBO-Modellstruktur,
      DSCR-Berechnung, Leverage-Limits, IRR/MOIC-Benchmarks für PE
    - **Kaplan & Strömberg (2009)** – *Leveraged Buyouts and Private Equity* (Journal of Economic Perspectives):
      Empirische Daten zu Entry-Leverage (Ø 5–6x), Hold Period (Ø 5 Jahre), IRR-Verteilungen
    - **Axelson et al. (2013)** – *Borrow Cheap, Buy High* (Journal of Finance):
      Debt Capacity und Covenant-Strukturen in LBO-Transaktionen
    """)
    st.stop()


# ═════════════════════════════════════════════
# PAGE: LBO ANALYSE
# ═════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 LBO Screener")
    st.markdown("---")

    # ── INPUT MODE ──────────────────────────
    st.markdown("### 📂 Dateneingabe")
    input_mode = st.radio(
        "Modus",
        ["📤 Moodys Orbis / Excel Upload", "✏️ Manuelle Eingabe", "🧪 Demo (Glock)"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Deal-Annahmen")
    entry_multiple  = st.slider("Entry EV/EBITDA",     4.0, 12.0,  6.5, 0.5)
    equity_pct      = st.slider("Equity Anteil (%)",   20,  60,    40,  5) / 100
    debt_rate       = st.slider("Zinssatz (%)",         3.0, 12.0,  6.5, 0.25) / 100
    exit_multiple   = st.slider("Exit EV/EBITDA",      4.0, 14.0,  7.0, 0.5)
    hold_period     = st.slider("Haltedauer (Jahre)",  3,    7,     5)
    rev_cagr        = st.slider("Umsatz-CAGR (%)",     0.0, 15.0,  4.0, 0.5) / 100
    margin_imp      = st.slider("Margin Improvement (bps/Jahr)", 0, 200, 50) / 10000

# ─────────────────────────────────────────────
# DATA INPUT & PARSING
# ─────────────────────────────────────────────

company_inputs  = None
parse_warnings  = []
currency_info   = st.session_state.currency_info
timeseries_df   = None
is_moodys       = False
parser_instance = None

# ── DEMO DATA (Glock-like) ─────────────────
if input_mode == "🧪 Demo (Glock)":
    # Values in tsd USD as in real Glock export – latest year 2024
    company_inputs = CompanyInputs(
        revenue=696398.0, ebitda=173445.0, ebit=127093.0,
        depreciation=46352.0, interest_expense=4341.0,
        tax_rate=0.227, total_debt=0.0, cash=406014.0,
        net_working_capital=251406.0, capex=46352.0,
        company_name="GLOCK GESELLSCHAFT M.B.H.",
    )
    currency_info = detect_currency_and_unit("tsd USD")
    st.session_state.currency_info = currency_info
    st.info("🧪 Demo: GLOCK GmbH – tsd USD – Werte aus Moodys Orbis Export 2024")

# ── MANUAL INPUT ───────────────────────────
elif input_mode == "✏️ Manuelle Eingabe":
    st.markdown("### 📋 Manuelle Eingabe")

    unit_sel = st.selectbox("Einheit der Werte", ["tsd EUR", "tsd USD", "Mio EUR", "Mio USD", "Mrd EUR"])
    currency_info = detect_currency_and_unit(unit_sel)
    company_name_inp = st.text_input("Unternehmensname", "Target GmbH")

    col1, col2, col3 = st.columns(3)
    with col1:
        rev    = st.number_input("Umsatz", value=56800.0, step=100.0)
        ebitda = st.number_input("EBITDA", value=11900.0, step=100.0)
        ebit   = st.number_input("EBIT",   value=8900.0,  step=100.0)
        dep    = st.number_input("D&A",    value=3000.0,  step=100.0)
    with col2:
        interest = st.number_input("Zinsaufwand", value=1500.0, step=100.0)
        tax_rate = st.number_input("Steuersatz (%)", value=26.0, step=0.5) / 100
        capex    = st.number_input("CapEx",      value=2500.0,  step=100.0)
    with col3:
        debt = st.number_input("Finanzschulden", value=20000.0, step=500.0)
        cash = st.number_input("Cash",           value=4500.0,  step=100.0)
        nwc  = st.number_input("NWC",            value=6800.0,  step=100.0)

    company_inputs = CompanyInputs(
        revenue=rev, ebitda=ebitda, ebit=ebit, depreciation=dep,
        interest_expense=interest, tax_rate=tax_rate,
        total_debt=debt, cash=cash, net_working_capital=nwc,
        capex=capex, company_name=company_name_inp,
    )
    st.session_state.currency_info = currency_info

# ── FILE UPLOAD ────────────────────────────
elif input_mode == "📤 Moodys Orbis / Excel Upload":
    uploaded = st.file_uploader(
        "Excel-Datei hochladen (Moodys Orbis oder Standard GuV/Bilanz)",
        type=["xlsx", "xls"],
        help="Moodys Orbis 4-Sheet Export wird automatisch erkannt"
    )

    if uploaded:
        with st.spinner("Analysiere Dateiformat..."):
            parser_instance, is_moodys = parse_file(uploaded)

        if is_moodys:
            p: MoodysOrbisParser = parser_instance
            currency_info = p.currency_info
            st.session_state.currency_info = currency_info

            st.success(
                f"✅ **Moodys Orbis Format erkannt** · "
                f"{len(p.years)} Jahre ({min(p.years) if p.years else '?'}–{max(p.years) if p.years else '?'}) · "
                f"{currency_info['display']}"
            )

            # Year selector
            if p.years:
                selected_year = st.selectbox(
                    "Basisjahr für LBO-Analyse",
                    options=sorted(p.years, reverse=True),
                    index=0
                )
            else:
                selected_year = None

            # Show timeseries preview
            ts = p.get_timeseries()
            if not ts.empty:
                with st.expander("📈 Zeitreihe (alle Jahre)", expanded=False):
                    sym = currency_info["symbol"]
                    unit = currency_info["unit_label"]
                    st.caption(f"Werte in {sym} {unit}")
                    st.dataframe(
                        ts.style.format(lambda x: f"{x:,.0f}" if isinstance(x, float) else str(x)),
                        use_container_width=True
                    )

            if st.button("🚀 LBO-Analyse starten", type="primary"):
                company_inputs, parse_warnings, currency_info = p.build_company_inputs(
                    use_year=selected_year
                )
                timeseries_df = ts
                st.session_state.currency_info = currency_info

        else:
            # Generic fallback
            p: GenericExcelParser = parser_instance
            currency_info = p.currency_info
            mapping_result = p.auto_map()
            st.warning(
                f"⚠️ Kein Moodys Orbis Format erkannt – generischer Parser aktiv. "
                f"{len(mapping_result['mapped'])}/{len(REQUIRED_FIELDS)} Felder erkannt."
            )
            # Manual override inputs
            manual = {}
            if mapping_result["unmapped"]:
                with st.expander("🔧 Fehlende Felder manuell eingeben"):
                    for field in mapping_result["unmapped"]:
                        label = REQUIRED_FIELDS[field]
                        val = st.number_input(f"{label}", value=0.0, key=f"man_{field}")
                        if val != 0.0:
                            manual[field] = val
            if st.button("🚀 Analyse starten", type="primary"):
                company_inputs, parse_warnings, currency_info = p.build_company_inputs(manual)
                st.session_state.currency_info = currency_info
    else:
        st.info("Lade eine Moodys Orbis Excel-Datei hoch")


# ─────────────────────────────────────────────
# GATE: No data yet
# ─────────────────────────────────────────────

currency_info = st.session_state.currency_info
sym  = currency_info.get("symbol", "")
unit = currency_info.get("unit_label", "")
ccy  = currency_info.get("currency", "")

if company_inputs is None:
    st.markdown("## 👈 Wähle einen Eingabemodus")
    st.markdown(f"""
    **Unterstützte Formate:**
    - 📁 **Moodys Orbis** 4-Sheet Export (Cover, Bilanz, GuV-Rechnung, Kennzahlen)
    - 📁 Standard Excel GuV/Bilanz
    - ✏️ Manuelle Eingabe
    - 🧪 Demo-Daten (GLOCK GmbH)

    **Features:**
    - Automatische Währungserkennung (USD/EUR, tsd/Mio/Mrd)
    - 10-Jahres Zeitreihe
    - Konfigurierbare Schwellenwerte (⚙️ Einstellungen)
    - IRR/MOIC/DSCR/FCF · Heatmaps · Red Flags
    """)
    st.stop()

# ─────────────────────────────────────────────
# RUN ENGINE
# ─────────────────────────────────────────────

assumptions = LBOAssumptions(
    entry_ev_multiple=entry_multiple,
    equity_contribution_pct=equity_pct,
    senior_debt_rate=debt_rate,
    debt_amortization_years=F["debt_amort_years"],
    exit_multiple=exit_multiple,
    holding_period=hold_period,
    revenue_cagr=rev_cagr,
    ebitda_margin_improvement=margin_imp,
    max_leverage_covenant=T["max_debt_ebitda"],
    min_dscr_covenant=T["min_dscr"],
)

# Override thresholds in engine evaluation
engine = LBOEngine(company_inputs, assumptions)
# Patch thresholds into engine before run
engine._T = T
results = engine.run()

# Re-evaluate flags with custom thresholds
results.red_flags = []
c = company_inputs
a = assumptions

def add_flag(condition, level, msg):
    if condition:
        prefix = "🔴" if level == "red" else "⚠️"
        results.red_flags.append(f"{prefix} {msg}")

add_flag(results.entry_leverage > T["max_entry_leverage"],  "red",  f"Entry Leverage {results.entry_leverage:.1f}x > {T['max_entry_leverage']}x Threshold")
add_flag(results.entry_leverage > T["warn_entry_leverage"] and results.entry_leverage <= T["max_entry_leverage"], "warn", f"Entry Leverage {results.entry_leverage:.1f}x im Warnbereich (>{T['warn_entry_leverage']}x)")
add_flag(results.dscr_base < T["min_dscr"],    "red",  f"DSCR Year 1 = {results.dscr_base:.2f}x < Covenant Floor {T['min_dscr']}x")
add_flag(results.dscr_base < T["warn_dscr"] and results.dscr_base >= T["min_dscr"], "warn", f"DSCR {results.dscr_base:.2f}x im Warnbereich (< {T['warn_dscr']}x)")
add_flag(results.irr < T["min_irr"],           "red",  f"IRR {results.irr:.1%} < Hurdle Rate {T['min_irr']:.0%}")
add_flag(results.moic < T["min_moic"],         "red",  f"MOIC {results.moic:.1f}x < {T['min_moic']}x Threshold")
add_flag(results.covenant_breach_year is not None, "red", f"Covenant Breach projiziert in Jahr {results.covenant_breach_year}")
add_flag(c.ebitda / max(c.revenue, 1) < T["min_ebitda_margin"], "warn", f"EBITDA-Marge {c.ebitda/c.revenue:.1%} < {T['min_ebitda_margin']:.0%} Minimum")
add_flag(results.fcf_yield < T["min_fcf_yield"], "warn", f"FCF Yield {results.fcf_yield:.1%} < {T['min_fcf_yield']:.0%} Minimum")

results.is_lbo_viable = (
    results.irr >= T["min_irr"] and
    results.moic >= T["min_moic"] and
    results.dscr_base >= T["min_dscr"] and
    results.entry_leverage <= T["max_entry_leverage"]
)

# Sensitivities
sens = SensitivityEngine(company_inputs, assumptions)
irr_hm  = sens.irr_heatmap([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], [0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
dscr_hm = sens.dscr_heatmap([0.04, 0.055, 0.07, 0.085, 0.10], [3.0, 4.0, 5.0, 5.5, 6.0, 6.5])

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown(
        f"# {company_inputs.company_name}"
        f"<span class='currency-badge'>{ccy} · {unit}</span>",
        unsafe_allow_html=True
    )

viable_color = "#00cc88" if results.is_lbo_viable else "#ff4b4b"
viable_label = "✅ LBO-FÄHIG" if results.is_lbo_viable else "❌ KRITISCH"
red_count    = sum(1 for f in results.red_flags if "🔴" in f)
warn_count   = sum(1 for f in results.red_flags if "⚠️" in f)

with col_badge:
    st.markdown(
        f'<div style="text-align:right;margin-top:12px">'
        f'<span style="background:{viable_color}22;border:1px solid {viable_color};'
        f'border-radius:8px;padding:6px 14px;color:{viable_color};font-weight:700">'
        f'{viable_label}</span><br>'
        f'<span style="font-size:.8em;color:#aaa;margin-top:4px;display:block">'
        f'🔴 {red_count} Red Flags · ⚠️ {warn_count} Warnungen</span></div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────

k1, k2, k3, k4, k5, k6 = st.columns(6)
kpis = [
    (k1, "Entry EV",        f"{sym}{results.entry_ev:,.0f}",         None),
    (k2, "Entry Leverage",  f"{results.entry_leverage:.1f}x",         "inverse"),
    (k3, "DSCR Year 1",     f"{results.dscr_base:.2f}x",              "normal"),
    (k4, "Base IRR",        f"{results.irr:.1%}",                     "normal"),
    (k5, "MOIC",            f"{results.moic:.1f}x",                   "normal"),
    (k6, "FCF Yield",       f"{results.fcf_yield:.1%}",               "normal"),
]
for col, label, val, _ in kpis:
    with col:
        st.metric(label, val)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Ergebnisse", "🗺️ Heatmaps", "📅 Debt Schedule", "🚩 Red Flags", "📈 Historisch"
])

# ── Tab 1: Results ─────────────────────────
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-hdr">Entry-Struktur</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Kennzahl": ["Entry EV", "Entry Equity", "Entry Debt", "Net Leverage", "Debt Capacity"],
            "Wert": [
                f"{sym}{results.entry_ev:,.0f} ({unit})",
                f"{sym}{results.entry_equity:,.0f} ({equity_pct:.0%})",
                f"{sym}{results.entry_debt:,.0f} ({1-equity_pct:.0%})",
                f"{results.entry_leverage:.1f}x Net Debt/EBITDA",
                f"{sym}{results.debt_capacity:,.0f} ({T['max_debt_ebitda']}x EBITDA)",
            ]
        }).set_index("Kennzahl"), use_container_width=True)

        st.markdown('<div class="section-hdr">Exit & Returns</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Kennzahl": ["Exit EV", "Exit Equity", "IRR", "MOIC"],
            "Wert": [
                f"{sym}{results.exit_ev:,.0f}",
                f"{sym}{results.exit_equity:,.0f}",
                f"{results.irr:.1%}",
                f"{results.moic:.2f}x",
            ]
        }).set_index("Kennzahl"), use_container_width=True)

        st.markdown('<div class="section-hdr">Input-Summary</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Kennzahl": ["Umsatz", "EBITDA", "EBIT", "D&A", "Zinsaufwand", "Steuersatz",
                         "Finanzschulden", "Cash", "NWC", "CapEx"],
            f"Wert ({sym} {unit})": [
                f"{company_inputs.revenue:,.0f}", f"{company_inputs.ebitda:,.0f}",
                f"{company_inputs.ebit:,.0f}", f"{company_inputs.depreciation:,.0f}",
                f"{company_inputs.interest_expense:,.0f}", f"{company_inputs.tax_rate:.1%}",
                f"{company_inputs.total_debt:,.0f}", f"{company_inputs.cash:,.0f}",
                f"{company_inputs.net_working_capital:,.0f}", f"{company_inputs.capex:,.0f}",
            ]
        }).set_index("Kennzahl"), use_container_width=True)

    with c2:
        years_x = [f"Jahr {y}" for y in range(1, hold_period+1)]

        fig_fcf = go.Figure(go.Bar(
            x=years_x, y=results.fcf_series,
            marker_color=["#4f8ef7" if v >= 0 else "#ff4b4b" for v in results.fcf_series],
        ))
        fig_fcf.update_layout(template="plotly_dark", height=200, title="FCF-Projektion",
                              margin=dict(t=30,b=20,l=20,r=20), yaxis_title=f"{sym} {unit}")
        st.plotly_chart(fig_fcf, use_container_width=True)

        fig_dscr = go.Figure()
        fig_dscr.add_trace(go.Scatter(x=years_x, y=results.dscr_series,
                                      mode="lines+markers", line=dict(color="#4f8ef7", width=2)))
        fig_dscr.add_hline(y=T["min_dscr"],  line_dash="dash", line_color="#ff4b4b",
                           annotation_text=f"Covenant {T['min_dscr']}x")
        fig_dscr.add_hline(y=T["warn_dscr"], line_dash="dot",  line_color="#ffaa00",
                           annotation_text=f"Warnstufe {T['warn_dscr']}x")
        fig_dscr.update_layout(template="plotly_dark", height=200, title="DSCR-Entwicklung",
                               margin=dict(t=30,b=20,l=20,r=20))
        st.plotly_chart(fig_dscr, use_container_width=True)

# ── Tab 2: Heatmaps ────────────────────────
with tab2:
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("#### IRR – Exit Multiple vs. EBITDA CAGR")
        irr_vals = irr_hm.values.astype(float)
        fig_irr = go.Figure(go.Heatmap(
            z=irr_vals, x=irr_hm.columns.tolist(), y=irr_hm.index.tolist(),
            colorscale=[[0,"#ff4b4b"],[0.35,"#ffaa00"],[0.55,"#4f8ef7"],[1,"#00cc88"]],
            zmid=T["min_irr"]*100,
            text=[[f"{v:.1f}%" for v in row] for row in irr_vals],
            texttemplate="%{text}", colorbar=dict(title="IRR %"),
        ))
        fig_irr.add_annotation(text=f"Hurdle: {T['min_irr']:.0%}", xref="paper", yref="paper",
                               x=0.01, y=0.01, showarrow=False, font=dict(color="#ffaa00", size=10))
        fig_irr.update_layout(template="plotly_dark", height=350,
                              xaxis_title="EBITDA CAGR", yaxis_title="Exit Multiple",
                              margin=dict(t=30,b=40,l=80,r=20))
        st.plotly_chart(fig_irr, use_container_width=True)

    with h2:
        st.markdown("#### DSCR – Leverage vs. Zinssatz")
        dscr_vals = dscr_hm.values.astype(float)
        fig_dscr_hm = go.Figure(go.Heatmap(
            z=dscr_vals, x=dscr_hm.columns.tolist(), y=dscr_hm.index.tolist(),
            colorscale=[[0,"#ff4b4b"],[0.4,"#ffaa00"],[0.7,"#4f8ef7"],[1,"#00cc88"]],
            zmid=T["warn_dscr"],
            text=[[f"{v:.2f}x" for v in row] for row in dscr_vals],
            texttemplate="%{text}", colorbar=dict(title="DSCR"),
        ))
        fig_dscr_hm.update_layout(template="plotly_dark", height=350,
                                  xaxis_title="Leverage", yaxis_title="Zinssatz",
                                  margin=dict(t=30,b=40,l=80,r=20))
        st.plotly_chart(fig_dscr_hm, use_container_width=True)

    st.caption(f"Schwellenwerte: IRR Hurdle {T['min_irr']:.0%} · DSCR Floor {T['min_dscr']}x · Leverage Max {T['max_entry_leverage']}x")

# ── Tab 3: Debt Schedule ───────────────────
with tab3:
    st.markdown(f"#### Tilgungsplan ({sym} {unit})")
    st.dataframe(
        results.debt_schedule.style.format("{:,.1f}").background_gradient(
            subset=["Closing Debt (€M)"], cmap="RdYlGn_r"
        ), use_container_width=True
    )
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.metric(f"Debt Capacity ({T['max_debt_ebitda']}x EBITDA)", f"{sym}{results.debt_capacity:,.0f}")
    with dc2:
        st.metric("Entry Debt", f"{sym}{results.entry_debt:,.0f}")
    with dc3:
        headroom = results.debt_capacity - results.entry_debt
        st.metric("Headroom", f"{sym}{headroom:,.0f}", delta="✅ OK" if headroom > 0 else "❌ Exceeded")

# ── Tab 4: Red Flags ───────────────────────
with tab4:
    reds  = [f for f in results.red_flags if "🔴" in f]
    warns = [f for f in results.red_flags if "⚠️" in f]

    if reds:
        st.markdown(f"### 🔴 {len(reds)} kritische Red Flag(s)")
        for f in reds:
            st.markdown(f'<div class="red-flag">{f}</div>', unsafe_allow_html=True)

    if warns:
        st.markdown(f"### ⚠️ {len(warns)} Warnung(en)")
        for f in warns:
            st.markdown(f'<div class="warn-flag">{f}</div>', unsafe_allow_html=True)

    if not results.red_flags:
        st.markdown('<div class="green-flag">✅ Keine Red Flags – Struktur erscheint tragfähig</div>',
                    unsafe_allow_html=True)

    if parse_warnings:
        st.markdown("### ℹ️ Parser-Hinweise")
        for w in parse_warnings:
            st.warning(w)

    st.markdown("---")
    st.markdown(f"*Schwellenwerte konfiguriert in ⚙️ Einstellungen · Basis: Rosenbaum & Pearl*")

# ── Tab 5: Historical Timeseries ──────────
with tab5:
    if timeseries_df is not None and not timeseries_df.empty:
        st.markdown(f"#### Historische Zeitreihe – {company_inputs.company_name} ({sym} {unit})")
        st.dataframe(
            timeseries_df.style.format(
                lambda x: f"{x:,.0f}" if isinstance(x, (float, int)) and not pd.isna(x) else "n.v."
            ), use_container_width=True
        )
        # Revenue & EBITDA trend chart
        if "revenue" in timeseries_df.columns and "ebitda" in timeseries_df.columns:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(
                x=timeseries_df.index.astype(str),
                y=timeseries_df["revenue"],
                name="Umsatz", marker_color="#4f8ef7", opacity=0.7,
            ))
            fig_hist.add_trace(go.Scatter(
                x=timeseries_df.index.astype(str),
                y=timeseries_df["ebitda"],
                name="EBITDA", mode="lines+markers", line=dict(color="#00cc88", width=2),
                yaxis="y2",
            ))
            fig_hist.update_layout(
                template="plotly_dark", height=350,
                title=f"Umsatz & EBITDA ({sym} {unit})",
                yaxis=dict(title=f"Umsatz ({sym})"),
                yaxis2=dict(title="EBITDA", overlaying="y", side="right"),
                legend=dict(x=0.01, y=0.99),
                margin=dict(t=40, b=20, l=60, r=60),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Historische Daten nur verfügbar bei Moodys Orbis Upload")

# ─────────────────────────────────────────────
# FOOTER & EXPORT
# ─────────────────────────────────────────────

st.markdown("---")
col_f1, col_f2, col_f3 = st.columns([3, 1, 1])
with col_f1:
    st.caption(f"LBO Screener v2.0 · {ccy} {unit} · Thresholds: IRR ≥{T['min_irr']:.0%} | MOIC ≥{T['min_moic']}x | DSCR ≥{T['min_dscr']}x | Leverage ≤{T['max_entry_leverage']}x · Not investment advice")
with col_f2:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results.debt_schedule.to_excel(writer, sheet_name="Debt Schedule")
        irr_hm.to_excel(writer, sheet_name="IRR Heatmap")
        dscr_hm.to_excel(writer, sheet_name="DSCR Heatmap")
        pd.DataFrame({
            "FCF": results.fcf_series, "DSCR": results.dscr_series
        }, index=[f"Jahr {i+1}" for i in range(hold_period)]).to_excel(writer, sheet_name="FCF & DSCR")
        if timeseries_df is not None and not timeseries_df.empty:
            timeseries_df.to_excel(writer, sheet_name="Zeitreihe")
    output.seek(0)
    st.download_button(
        "📥 Excel Export",
        data=output,
        file_name=f"LBO_{company_inputs.company_name.replace(' ','_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )