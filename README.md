# LBO Screening Tool – Phase 1 MVP

> Schnelles LBO-Screening für Private Equity Investoren  
> Built with Python + Streamlit · 100% Free · Deployable on Streamlit Cloud

---

## 🚀 Quick Start (lokal)

```bash
# 1. Repo klonen
git clone https://github.com/DEIN_USERNAME/lbo-screener.git
cd lbo-screener

# 2. Dependencies installieren
pip install -r requirements.txt

# 3. App starten
streamlit run app.py
```

## ☁️ Deployment (Streamlit Cloud – kostenlos)

1. Repo auf GitHub pushen (kann auch privat sein)
2. https://share.streamlit.io → "New app"
3. Repo + Branch + `app.py` auswählen
4. Deploy → fertig, öffentliche URL

---

## 📁 Projektstruktur

```
lbo_tool/
├── app.py                    ← Streamlit App (UI)
├── requirements.txt
├── engine/
│   └── finance_engine.py     ← Core LBO-Berechnungen
├── utils/
│   └── data_parser.py        ← Excel & Moodys/Nortdata Parsing
├── data/
│   └── sample/               ← Demo-Dateien
└── reports/                  ← PDF-Export (Phase 2)
```

---

## ⚙️ Features Phase 1

| Feature | Status |
|---|---|
| Excel Upload (Bilanz/GuV) | ✅ |
| Moodys/Nortdata CSV | ✅ |
| Manuelles Column Mapping | ✅ |
| FCF Berechnung | ✅ |
| Debt Capacity & Tilgungsplan | ✅ |
| DSCR Base Case | ✅ |
| IRR & MOIC | ✅ |
| IRR Heatmap (Exit Multiple vs CAGR) | ✅ |
| DSCR Heatmap (Leverage vs Zins) | ✅ |
| Red Flag Detection | ✅ |
| Excel Export | ✅ |
| PDF Export | 🔜 Phase 2 |
| KI Interpretation | 🔜 Phase 2 |
| Monte Carlo | 🔜 Phase 3 |

---

## 📊 Unterstützte Datenformate

### Excel (Bilanz/GuV)
- Standard German/Austrian export format
- Auto-Erkennung von Zeilen-/Spalten-Format
- Manuelle Mapping-Option falls Spalten unbekannt

### Moodys / Nortdata CSV
- UTF-8, Latin-1, CP1252 Encoding
- Automatisches Alias-Matching

---

## 📐 Berechnungslogik

**Free Cash Flow:**
```
FCF = NOPAT + D&A - CapEx - ΔNWC
NOPAT = EBIT × (1 - Steuersatz)
```

**DSCR:**
```
DSCR = FCF / (Zinsen + Tilgung)
Covenant Floor: 1.20x
```

**IRR / MOIC:**
```
Cash Flows = [-Equity₀, 0, ..., Exit Equity]
MOIC = Exit Equity / Entry Equity
```

**Red Flags:**
- Entry Leverage > 6.0x
- DSCR Year 1 < 1.20x
- IRR < 15%
- MOIC < 2.0x
- Covenant Breach in Projektion
- EBITDA Margin < 8%

---

*For indicative purposes only – not investment advice*
