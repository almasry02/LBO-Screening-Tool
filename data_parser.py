"""
Data Parser – Excel (Bilanz/GuV) & Moodys/Nortdata Format
Handles column mapping, normalization, and validation
"""

import pandas as pd
import numpy as np
import sys, os
from typing import Optional

# Ensure project root is on path (needed when imported from Streamlit Cloud)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.finance_engine import CompanyInputs


# ─────────────────────────────────────────────
# FIELD MAPPINGS
# ─────────────────────────────────────────────

# Standard internal field names → display labels
REQUIRED_FIELDS = {
    "revenue":          "Umsatz / Revenue (€M)",
    "ebitda":           "EBITDA (€M)",
    "ebit":             "EBIT (€M)",
    "depreciation":     "D&A (€M)",
    "interest_expense": "Zinsaufwand / Interest Expense (€M)",
    "tax_rate":         "Steuerquote / Tax Rate (0–1)",
    "total_debt":       "Gesamtschulden / Total Debt (€M)",
    "cash":             "Kasse / Cash (€M)",
    "net_working_capital": "Net Working Capital (€M)",
    "capex":            "Investitionen / CapEx (€M)",
}

# Moodys / Nortdata typical column name variants
MOODYS_ALIASES = {
    "revenue":          ["Net Sales", "Revenue", "Umsatz", "Gesamtumsatz", "Total Revenue", "Sales"],
    "ebitda":           ["EBITDA", "Ebitda", "Operating EBITDA", "Reported EBITDA"],
    "ebit":             ["EBIT", "Ebit", "Operating Income", "Operating Profit"],
    "depreciation":     ["D&A", "Depreciation", "Depreciation & Amortization", "Abschreibungen"],
    "interest_expense": ["Interest Expense", "Zinsaufwand", "Net Interest Expense", "Finance Costs"],
    "tax_rate":         ["Tax Rate", "Effective Tax Rate", "Steuerquote"],
    "total_debt":       ["Total Debt", "Gesamtschulden", "Financial Debt", "Gross Debt"],
    "cash":             ["Cash", "Cash & Equivalents", "Kasse", "Liquide Mittel"],
    "net_working_capital": ["Net Working Capital", "NWC", "Working Capital"],
    "capex":            ["CapEx", "Capex", "Capital Expenditures", "Investitionen", "CAPEX"],
}

# Generic Excel aliases (user-uploaded Bilanz/GuV)
EXCEL_ALIASES = {
    "revenue":          ["Umsatz", "Umsatzerlöse", "Revenue", "Net Revenue", "Sales"],
    "ebitda":           ["EBITDA", "Betriebsergebnis vor Abschreibungen"],
    "ebit":             ["EBIT", "Betriebsergebnis", "Operating Result"],
    "depreciation":     ["AfA", "Abschreibungen", "D&A", "Depreciation"],
    "interest_expense": ["Zinsaufwand", "Zinsen", "Interest", "Finanzaufwand"],
    "tax_rate":         ["Steuersatz", "Tax Rate", "Ertragsteuerquote"],
    "total_debt":       ["Verbindlichkeiten ggü. KI", "Bankverbindlichkeiten", "Financial Debt", "Total Debt"],
    "cash":             ["Kasse", "Kassenbestand", "Flüssige Mittel", "Cash"],
    "net_working_capital": ["NWC", "Working Capital", "Umlaufvermögen (netto)"],
    "capex":            ["Investitionen", "CapEx", "Sachanlagen-Zugänge", "Capital Expenditure"],
}

ALL_ALIASES = {k: list(set(MOODYS_ALIASES[k] + EXCEL_ALIASES[k])) for k in REQUIRED_FIELDS}


# ─────────────────────────────────────────────
# PARSER CLASS
# ─────────────────────────────────────────────

class FinancialDataParser:

    def __init__(self):
        self.raw_df: Optional[pd.DataFrame] = None
        self.detected_format: str = "unknown"
        self.mapping: dict = {}          # internal_field → detected column
        self.unmapped: list = []         # fields that need manual mapping
        self.warnings: list = []

    # ── File Loading ───────────────────────────

    def load_excel(self, file) -> pd.DataFrame:
        """Load Excel file, try multiple sheet strategies"""
        xl = pd.ExcelFile(file)
        sheet_names = xl.sheet_names

        # Prefer sheets with financial keywords
        preferred = [s for s in sheet_names if any(
            kw in s.lower() for kw in ["guv", "p&l", "income", "bilanz", "balance", "financials", "summary"]
        )]
        sheet = preferred[0] if preferred else sheet_names[0]

        df = pd.read_excel(file, sheet_name=sheet, header=None)
        df = self._detect_header_row(df)
        self.raw_df = df
        self.detected_format = "excel"
        return df

    def load_moodys_csv(self, file) -> pd.DataFrame:
        """Load Moodys/Nortdata CSV export"""
        # Try different encodings common in German financial exports
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(file, encoding=enc, sep=None, engine="python")
                break
            except Exception:
                continue

        df.columns = [str(c).strip() for c in df.columns]
        self.raw_df = df
        self.detected_format = "moodys_csv"
        return df

    # ── Header Detection ───────────────────────

    def _detect_header_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find the row that contains column headers (not always row 0)"""
        for i, row in df.iterrows():
            values = [str(v).lower() for v in row if pd.notna(v)]
            if any(kw in " ".join(values) for kw in ["umsatz", "revenue", "ebitda", "sales"]):
                df.columns = df.iloc[i]
                df = df.iloc[i+1:].reset_index(drop=True)
                df.columns = [str(c).strip() for c in df.columns]
                return df
        # Fallback: assume first row
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    # ── Auto-Mapping ───────────────────────────

    def auto_map_columns(self) -> dict:
        """Try to auto-map raw columns to internal field names"""
        if self.raw_df is None:
            raise ValueError("No data loaded yet")

        available_cols = list(self.raw_df.columns)
        aliases = ALL_ALIASES

        mapped = {}
        unmapped = []

        for field, alias_list in aliases.items():
            found = None
            for alias in alias_list:
                # Exact match
                if alias in available_cols:
                    found = alias
                    break
                # Case-insensitive
                for col in available_cols:
                    if alias.lower() == col.lower():
                        found = col
                        break
                if found:
                    break

            if found:
                mapped[field] = found
            else:
                unmapped.append(field)

        self.mapping = mapped
        self.unmapped = unmapped
        return {"mapped": mapped, "unmapped": unmapped, "available_columns": available_cols}

    # ── Value Extraction ───────────────────────

    def extract_latest_year(self) -> dict:
        """
        Extract most recent year's values.
        For wide-format (years as columns): takes rightmost numeric column.
        For long-format (rows = line items): aggregates by row label.
        """
        df = self.raw_df
        values = {}

        # Detect if wide format (multiple year columns)
        year_cols = [c for c in df.columns if str(c).strip().isdigit() and 2015 <= int(str(c).strip()) <= 2030]

        if year_cols:
            latest_year_col = sorted(year_cols)[-1]
            for field, col_name in self.mapping.items():
                try:
                    mask = df.iloc[:, 0].astype(str).str.lower().str.contains(col_name.lower(), na=False)
                    val = df.loc[mask, latest_year_col].values
                    if len(val) > 0:
                        values[field] = self._parse_number(val[0])
                except Exception:
                    pass
        else:
            # Assume long format: row = year, col = metric
            try:
                last_row = df.dropna(how="all").iloc[-1]
                for field, col_name in self.mapping.items():
                    if col_name in df.columns:
                        values[field] = self._parse_number(last_row.get(col_name, 0))
            except Exception:
                pass

        return values

    def _parse_number(self, val) -> float:
        """Robust number parsing (handles German formats, thousands separators)"""
        if isinstance(val, (int, float)) and not np.isnan(val):
            return float(val)
        s = str(val).strip().replace(" ", "").replace(".", "").replace(",", ".")
        s = s.replace("€", "").replace("T€", "").replace("k", "000")
        try:
            return float(s)
        except Exception:
            return 0.0

    # ── Build CompanyInputs ────────────────────

    def build_company_inputs(
        self,
        extracted_values: dict,
        manual_overrides: dict = {},
        company_name: str = "Target",
        scale_factor: float = 1.0,   # e.g. if values are in €k → divide by 1000
    ) -> tuple[CompanyInputs, list]:
        """
        Build CompanyInputs from extracted + manually provided values.
        Returns (CompanyInputs, list_of_warnings)
        """
        vals = {**extracted_values, **manual_overrides}
        warnings = []

        def get(field, default=0.0):
            v = vals.get(field, default)
            if v == 0.0:
                warnings.append(f"⚠️ '{field}' not found – using 0. Please verify.")
            return v * scale_factor if field != "tax_rate" else v

        # Derive EBIT if missing
        ebitda = get("ebitda")
        dep    = get("depreciation")
        ebit   = vals.get("ebit", ebitda - dep) * scale_factor

        # Derive tax rate if given as percentage
        tax_rate = get("tax_rate", 0.25)
        if tax_rate > 1:
            tax_rate /= 100.0
            warnings.append("Tax rate was > 1 – converted from % to decimal.")

        inputs = CompanyInputs(
            revenue=get("revenue"),
            ebitda=ebitda,
            ebit=ebit,
            depreciation=dep,
            interest_expense=get("interest_expense"),
            tax_rate=tax_rate,
            total_debt=get("total_debt"),
            cash=get("cash"),
            net_working_capital=get("net_working_capital"),
            capex=get("capex"),
            company_name=company_name,
        )

        return inputs, warnings


# ─────────────────────────────────────────────
# SAMPLE DATA GENERATOR (for demo / testing)
# ─────────────────────────────────────────────

def generate_sample_excel(path: str = "data/sample/sample_company.xlsx"):
    """Generate a sample Excel file mimicking a German Bilanz/GuV export"""
    data = {
        "Kennzahl": [
            "Umsatz", "EBITDA", "EBIT", "Abschreibungen",
            "Zinsaufwand", "Steuersatz",
            "Bankverbindlichkeiten", "Kassenbestand",
            "Net Working Capital", "Investitionen"
        ],
        "2022": [48.5, 9.2, 6.8, 2.4, 1.1, 0.28, 22.0, 3.5, 6.0, 2.1],
        "2023": [52.1, 10.4, 7.7, 2.7, 1.3, 0.27, 21.0, 4.0, 6.3, 2.3],
        "2024": [56.8, 11.9, 8.9, 3.0, 1.5, 0.26, 20.0, 4.5, 6.8, 2.5],
    }
    df = pd.DataFrame(data)
    df.to_excel(path, index=False)
    print(f"Sample file saved to {path}")
    return path
