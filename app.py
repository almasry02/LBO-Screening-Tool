"""
LBO Screening Tool – Phase 1 MVP
Streamlit Web App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from finance_engine import CompanyInputs, LBOAssumptions, LBOEngine, SensitivityEngine
from data_parser import FinancialDataParser, REQUIRED_FIELDS, generate_sample_excel

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
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #4f8ef7;
        margin-bottom: 10px;
    }
    .red-flag {
        background: #2d1a1a;
        border-left: 4px solid #ff4b4b;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 0.9em;
    }
    .green-flag {
        background: #1a2d1a;
        border-left: 4px solid #00cc88;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 0.9em;
    }
    .section-header {
        font-size: 1.1em;
        font-weight: 600;
        color: #4f8ef7;
        margin: 20px 0 8px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #2a2f3e;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR – INPUTS
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/investment-portfolio.png", width=40)
    st.title("LBO Screener")
    st.caption("Phase 1 MVP · Private Equity")

    st.markdown("---")
    st.markdown("### 📂 Daten-Upload")

    input_mode = st.radio(
        "Eingabemodus",
        ["📤 Excel / CSV hochladen", "✏️ Manuelle Eingabe", "🧪 Demo-Daten"],
        index=2
    )

    company_name = st.text_input("Unternehmensname", value="Target GmbH")
    scale = st.selectbox("Werte in", ["€ Millionen", "€ Tausend (÷1000)", "€ Milliarden (×1000)"])
    scale_factor = {"€ Millionen": 1.0, "€ Tausend (÷1000)": 0.001, "€ Milliarden (×1000)": 1000.0}[scale]

    st.markdown("---")
    st.markdown("### ⚙️ Deal-Annahmen")

    entry_multiple = st.slider("Entry EV/EBITDA", 4.0, 12.0, 6.5, 0.5)
    equity_pct     = st.slider("Equity Anteil (%)", 20, 60, 40, 5) / 100
    debt_rate      = st.slider("Zinssatz Senior Debt (%)", 3.0, 12.0, 6.5, 0.25) / 100
    exit_multiple  = st.slider("Exit EV/EBITDA", 4.0, 14.0, 7.0, 0.5)
    hold_period    = st.slider("Haltedauer (Jahre)", 3, 7, 5)
    rev_cagr       = st.slider("Umsatz-CAGR (%)", 0.0, 15.0, 4.0, 0.5) / 100
    margin_imp     = st.slider("EBITDA-Margin Improvement (bps/Jahr)", 0, 200, 50) / 10000


# ─────────────────────────────────────────────
# DATA INPUT
# ─────────────────────────────────────────────

parser = FinancialDataParser()
company_inputs = None
parse_warnings = []
mapping_overrides = {}

if input_mode == "🧪 Demo-Daten":
    company_inputs = CompanyInputs(
        revenue=56.8, ebitda=11.9, ebit=8.9,
        depreciation=3.0, interest_expense=1.5,
        tax_rate=0.26, total_debt=20.0, cash=4.5,
        net_working_capital=6.8, capex=2.5,
        company_name=company_name,
    )
    st.info("🧪 Demo-Daten aktiv – Lade eigene Daten hoch für echte Analyse")

elif input_mode == "✏️ Manuelle Eingabe":
    st.markdown("### 📋 Manuelle Finanzeingaben")
    col1, col2, col3 = st.columns(3)
    with col1:
        rev   = st.number_input("Umsatz (€M)", value=56.8, step=0.1)
        ebitda = st.number_input("EBITDA (€M)", value=11.9, step=0.1)
        ebit  = st.number_input("EBIT (€M)", value=8.9, step=0.1)
        dep   = st.number_input("D&A (€M)", value=3.0, step=0.1)
    with col2:
        interest = st.number_input("Zinsaufwand (€M)", value=1.5, step=0.1)
        tax_rate = st.number_input("Steuersatz (%)", value=26.0, step=0.5) / 100
        capex    = st.number_input("CapEx (€M)", value=2.5, step=0.1)
    with col3:
        debt  = st.number_input("Finanzschulden (€M)", value=20.0, step=0.5)
        cash  = st.number_input("Cash (€M)", value=4.5, step=0.1)
        nwc   = st.number_input("NWC (€M)", value=6.8, step=0.1)

    company_inputs = CompanyInputs(
        revenue=rev, ebitda=ebitda, ebit=ebit,
        depreciation=dep, interest_expense=interest,
        tax_rate=tax_rate, total_debt=debt, cash=cash,
        net_working_capital=nwc, capex=capex,
        company_name=company_name,
    )

elif input_mode == "📤 Excel / CSV hochladen":
    upload_format = st.radio("Format", ["Excel (Bilanz/GuV)", "Moodys / Nortdata CSV"])
    uploaded = st.file_uploader(
        "Datei hochladen",
        type=["xlsx", "xls", "csv"],
        help="Unterstützt Standard-Excel-Exporte sowie Moodys/Nortdata CSV"
    )

    if uploaded:
        with st.spinner("Parsing..."):
            if upload_format == "Excel (Bilanz/GuV)":
                parser.load_excel(uploaded)
            else:
                parser.load_moodys_csv(uploaded)

            mapping_result = parser.auto_map_columns()

        st.success(f"✅ {len(mapping_result['mapped'])} Felder automatisch erkannt")

        if mapping_result["unmapped"]:
            st.warning(f"⚠️ {len(mapping_result['unmapped'])} Felder benötigen manuelles Mapping")
            with st.expander("🔧 Column Mapping", expanded=True):
                available = ["– nicht vorhanden –"] + mapping_result["available_columns"]
                for field in mapping_result["unmapped"]:
                    label = REQUIRED_FIELDS[field]
                    sel = st.selectbox(f"{label}", available, key=f"map_{field}")
                    if sel != "– nicht vorhanden –":
                        mapping_overrides[field] = sel

        if st.button("🚀 Analyse starten"):
            parser.mapping.update(mapping_overrides)
            extracted = parser.extract_latest_year()
            company_inputs, parse_warnings = parser.build_company_inputs(
                extracted, company_name=company_name, scale_factor=scale_factor
            )
    else:
        st.info("Bitte Datei hochladen oder Demo-Daten verwenden")


# ─────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────

if company_inputs is None:
    st.markdown("## 👈 Wähle einen Eingabemodus in der Seitenleiste")
    st.markdown("""
    Dieses Tool führt ein schnelles **LBO-Screening** durch:
    - 📊 **Deterministische Kennzahlen** (FCF, DSCR, IRR, MOIC)
    - 🗺️ **Sensitivitäts-Heatmaps** (IRR vs. Exit Multiple / EBITDA-Wachstum)
    - 🚩 **Red Flag Detection** (automatische Warnsignale)
    - 📄 **Download** als PDF/Excel
    """)
    st.stop()

# Build assumptions
assumptions = LBOAssumptions(
    entry_ev_multiple=entry_multiple,
    equity_contribution_pct=equity_pct,
    senior_debt_rate=debt_rate,
    exit_multiple=exit_multiple,
    holding_period=hold_period,
    revenue_cagr=rev_cagr,
    ebitda_margin_improvement=margin_imp,
)

# Run engine
engine = LBOEngine(company_inputs, assumptions)
results = engine.run()

# Run sensitivities
sens = SensitivityEngine(company_inputs, assumptions)
irr_heatmap_df = sens.irr_heatmap(
    exit_multiples=[5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    ebitda_cagrs=[0.0, 0.02, 0.04, 0.06, 0.08, 0.10],
)
dscr_heatmap_df = sens.dscr_heatmap(
    interest_rates=[0.04, 0.055, 0.07, 0.085, 0.10],
    leverage_multiples=[3.0, 4.0, 5.0, 5.5, 6.0, 6.5],
)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown(f"# LBO Screening: {company_inputs.company_name}")
viable_color = "#00cc88" if results.is_lbo_viable else "#ff4b4b"
viable_label = "✅ LBO-FÄHIG" if results.is_lbo_viable else "❌ KRITISCH"
st.markdown(
    f'<div style="display:inline-block;background:{viable_color}22;border:1px solid {viable_color};'
    f'border-radius:8px;padding:6px 16px;color:{viable_color};font-weight:700;font-size:1.1em">'
    f'{viable_label}</div>',
    unsafe_allow_html=True
)
st.markdown("---")


# ─────────────────────────────────────────────
# KPI METRICS ROW
# ─────────────────────────────────────────────

col1, col2, col3, col4, col5, col6 = st.columns(6)

metrics = [
    (col1, "Entry EV", f"€{results.entry_ev:.1f}M", None),
    (col2, "Entry Leverage", f"{results.entry_leverage:.1f}x", "inverse"),
    (col3, "DSCR Year 1", f"{results.dscr_base:.2f}x", "normal"),
    (col4, "Base IRR", f"{results.irr:.1%}", "normal"),
    (col5, "MOIC", f"{results.moic:.1f}x", "normal"),
    (col6, "FCF Yield", f"{results.fcf_yield:.1%}", "normal"),
]

for col, label, value, direction in metrics:
    with col:
        st.metric(label, value)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Ergebnisse", "🗺️ Heatmaps", "📅 Debt Schedule", "🚩 Red Flags"
])


# ── Tab 1: Results ─────────────────────────────

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Entry-Struktur</div>', unsafe_allow_html=True)
        entry_df = pd.DataFrame({
            "Kennzahl": ["Entry EV", "Entry Equity", "Entry Debt", "Net Leverage"],
            "Wert": [
                f"€{results.entry_ev:.1f}M",
                f"€{results.entry_equity:.1f}M ({equity_pct:.0%})",
                f"€{results.entry_debt:.1f}M ({1-equity_pct:.0%})",
                f"{results.entry_leverage:.1f}x Net Debt/EBITDA",
            ]
        }).set_index("Kennzahl")
        st.dataframe(entry_df, use_container_width=True)

        st.markdown('<div class="section-header">Exit & Returns</div>', unsafe_allow_html=True)
        exit_df = pd.DataFrame({
            "Kennzahl": ["Exit EV", "Exit Equity", "IRR", "MOIC"],
            "Wert": [
                f"€{results.exit_ev:.1f}M",
                f"€{results.exit_equity:.1f}M",
                f"{results.irr:.1%}",
                f"{results.moic:.2f}x",
            ]
        }).set_index("Kennzahl")
        st.dataframe(exit_df, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">FCF-Entwicklung</div>', unsafe_allow_html=True)
        fcf_fig = go.Figure()
        years = list(range(1, hold_period + 1))
        fcf_fig.add_trace(go.Bar(
            x=[f"Jahr {y}" for y in years],
            y=results.fcf_series,
            marker_color=["#4f8ef7" if v >= 0 else "#ff4b4b" for v in results.fcf_series],
            name="FCF"
        ))
        fcf_fig.update_layout(
            template="plotly_dark", height=200,
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis_title="€M", showlegend=False
        )
        st.plotly_chart(fcf_fig, use_container_width=True)

        st.markdown('<div class="section-header">DSCR-Entwicklung</div>', unsafe_allow_html=True)
        dscr_fig = go.Figure()
        dscr_fig.add_trace(go.Scatter(
            x=[f"Jahr {y}" for y in years],
            y=results.dscr_series,
            mode="lines+markers",
            line=dict(color="#4f8ef7", width=2),
            name="DSCR"
        ))
        dscr_fig.add_hline(y=1.20, line_dash="dash", line_color="#ff4b4b",
                           annotation_text="Covenant Floor 1.20x")
        dscr_fig.update_layout(
            template="plotly_dark", height=200,
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis_title="DSCR", showlegend=False
        )
        st.plotly_chart(dscr_fig, use_container_width=True)


# ── Tab 2: Heatmaps ────────────────────────────

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### IRR Sensitivität – Exit Multiple vs. EBITDA-CAGR")

        irr_vals = irr_heatmap_df.values.astype(float)

        fig_irr = go.Figure(go.Heatmap(
            z=irr_vals,
            x=irr_heatmap_df.columns.tolist(),
            y=irr_heatmap_df.index.tolist(),
            colorscale=[
                [0.0, "#ff4b4b"],
                [0.3, "#ffaa00"],
                [0.5, "#4f8ef7"],
                [1.0, "#00cc88"],
            ],
            zmid=20,
            text=[[f"{v:.1f}%" for v in row] for row in irr_vals],
            texttemplate="%{text}",
            colorbar=dict(title="IRR %"),
        ))
        fig_irr.update_layout(
            template="plotly_dark", height=350,
            xaxis_title="EBITDA CAGR",
            yaxis_title="Exit Multiple",
            margin=dict(t=30, b=40, l=80, r=20),
        )
        st.plotly_chart(fig_irr, use_container_width=True)

    with col2:
        st.markdown("#### DSCR Sensitivität – Leverage vs. Zinssatz")

        dscr_vals = dscr_heatmap_df.values.astype(float)

        fig_dscr = go.Figure(go.Heatmap(
            z=dscr_vals,
            x=dscr_heatmap_df.columns.tolist(),
            y=dscr_heatmap_df.index.tolist(),
            colorscale=[
                [0.0, "#ff4b4b"],
                [0.4, "#ffaa00"],
                [0.7, "#4f8ef7"],
                [1.0, "#00cc88"],
            ],
            zmid=1.5,
            text=[[f"{v:.2f}x" for v in row] for row in dscr_vals],
            texttemplate="%{text}",
            colorbar=dict(title="DSCR"),
        ))
        fig_dscr.update_layout(
            template="plotly_dark", height=350,
            xaxis_title="Leverage (Net Debt/EBITDA)",
            yaxis_title="Zinssatz",
            margin=dict(t=30, b=40, l=80, r=20),
        )
        st.plotly_chart(fig_dscr, use_container_width=True)

    st.caption("Grün = attraktiv · Gelb = Grenzbereich · Rot = problematisch")


# ── Tab 3: Debt Schedule ───────────────────────

with tab3:
    st.markdown("#### Tilgungsplan")
    st.dataframe(
        results.debt_schedule.style.format("{:.2f}").background_gradient(
            subset=["Closing Debt (€M)"], cmap="RdYlGn_r"
        ),
        use_container_width=True,
    )

    st.markdown('<div class="section-header">Debt Capacity Check</div>', unsafe_allow_html=True)
    dc_col1, dc_col2, dc_col3 = st.columns(3)
    with dc_col1:
        st.metric("Max Debt Capacity (5x EBITDA)", f"€{results.debt_capacity:.1f}M")
    with dc_col2:
        st.metric("Actual Entry Debt", f"€{results.entry_debt:.1f}M")
    with dc_col3:
        headroom = results.debt_capacity - results.entry_debt
        st.metric("Headroom", f"€{headroom:.1f}M", delta=f"{'✅ OK' if headroom > 0 else '❌ Exceeded'}")


# ── Tab 4: Red Flags ───────────────────────────

with tab4:
    if results.red_flags:
        st.markdown(f"### ⚠️ {len(results.red_flags)} Red Flag(s) identifiziert")
        for flag in results.red_flags:
            st.markdown(f'<div class="red-flag">{flag}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="green-flag">✅ Keine kritischen Red Flags – Struktur erscheint tragfähig</div>',
                    unsafe_allow_html=True)

    if parse_warnings:
        st.markdown("### ℹ️ Daten-Parsing Warnungen")
        for w in parse_warnings:
            st.warning(w)

    st.markdown("---")
    st.markdown("### 📋 Input-Zusammenfassung")
    input_summary = pd.DataFrame({
        "Kennzahl": ["Umsatz", "EBITDA", "EBIT", "D&A", "Zinsaufwand", "Steuersatz",
                     "Finanzschulden", "Cash", "NWC", "CapEx"],
        "Wert (€M)": [
            company_inputs.revenue, company_inputs.ebitda, company_inputs.ebit,
            company_inputs.depreciation, company_inputs.interest_expense,
            f"{company_inputs.tax_rate:.1%}",
            company_inputs.total_debt, company_inputs.cash,
            company_inputs.net_working_capital, company_inputs.capex,
        ]
    }).set_index("Kennzahl")
    st.dataframe(input_summary, use_container_width=True)


# ─────────────────────────────────────────────
# FOOTER / EXPORT
# ─────────────────────────────────────────────

st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.caption("LBO Screener v1.0 · Phase 1 MVP · For indicative purposes only – not investment advice")
with col2:
    # Excel export
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results.debt_schedule.to_excel(writer, sheet_name="Debt Schedule")
        irr_heatmap_df.to_excel(writer, sheet_name="IRR Heatmap")
        dscr_heatmap_df.to_excel(writer, sheet_name="DSCR Heatmap")
        pd.DataFrame({"FCF (€M)": results.fcf_series, "DSCR": results.dscr_series},
                     index=[f"Jahr {i+1}" for i in range(hold_period)]).to_excel(writer, sheet_name="FCF & DSCR")
    output.seek(0)
    st.download_button(
        "📥 Download Excel",
        data=output,
        file_name=f"LBO_{company_inputs.company_name.replace(' ','_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
