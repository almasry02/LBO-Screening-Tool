"""
Data Parser v2 – Moodys Orbis Excel Format + Generic Excel
"""

import pandas as pd
import numpy as np
import sys, os
from datetime import date
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from finance_engine import CompanyInputs

REQUIRED_FIELDS = {
    "revenue":              "Umsatz / Revenue",
    "ebitda":               "EBITDA",
    "ebit":                 "EBIT",
    "depreciation":         "D&A / Abschreibungen",
    "interest_expense":     "Zinsaufwand / Interest Expense",
    "tax_rate":             "Steuerquote / Tax Rate (0–1)",
    "total_debt":           "Gesamtschulden / Total Debt",
    "cash":                 "Cash / Zahlungsmittel",
    "net_working_capital":  "Net Working Capital",
    "capex":                "CapEx / Investitionen",
}

ROW_ALIASES = {
    "revenue": [
        "umsatz", "betriebsertrag (umsatz)", "umsatzerlöse",
        "net sales", "revenue", "total revenue", "net revenue", "sales", "operating revenue",
    ],
    "ebitda": [
        "ebitda", "ebitda spanne",
        "earnings before interest taxes depreciation amortization",
    ],
    "ebit": [
        "betriebsgewinn/-verlust [=ebit]", "ebit", "operating profit",
        "operating income", "operating result", "betriebsergebnis",
    ],
    "depreciation": [
        "wertminderungen & abschreibungen", "abschreibungen", "afa", "d&a",
        "depreciation", "depreciation & amortization", "amortization",
    ],
    "interest_expense": [
        "zinsaufwand", "finanzaufwendungen", "interest expense",
        "finance costs", "net interest expense", "zinsen", "financial expense",
    ],
    "tax_rate": [
        "∟ steuern",
        "tax expense", "income taxes", "steueraufwand",
    ],
    "pretax_profit": [
        "gewinn/verlust vor steuern",
        "∟ gewinn/verlust vor steuern",
        "profit before tax",
        "income before tax",
        "ebt",
    ],
    "total_debt": [
        "langfristige finanzschulden", "∟ langfristige finanzschulden",
        "kurzfristige finanzschulden", "∟ kurzfristige finanzschulden",
        "vergebene kredite", "∟ vergebene kredite",
        "total debt", "financial debt", "gross debt",
        "gesamtschulden", "bankverbindlichkeiten",
    ],
    "cash": [
        "zahlungsmittel & zahlungsmitteläquivalente",
        "cash", "cash & equivalents", "kasse", "kassenbestand", "flüssige mittel",
    ],
    "net_working_capital": [
        "∟ working capital",
        "working capital",
        "net working capital",
        "nwc",
    ],
    "capex": [
        "capex",
        "capital expenditure",
        "capital expenditures",
        "sachanlagen-zugänge",
        "zugänge sachanlagen",
        "anlagezugänge",
        "additions to property",
        "additions to fixed assets",
        "investitionen in sachanlagen",
        "purchase of property",
    ],
}

MOODYS_SHEET_PRIORITY = {
    "revenue":          ["GuV-Rechnung", "P&L", "Income Statement"],
    "ebitda":           ["GuV-Rechnung", "P&L", "Income Statement"],
    "ebit":             ["GuV-Rechnung", "P&L", "Income Statement"],
    "depreciation":     ["GuV-Rechnung", "P&L", "Income Statement"],
    "interest_expense": ["GuV-Rechnung", "P&L", "Income Statement"],
    "tax_rate":         ["GuV-Rechnung", "P&L", "Income Statement"],
    "pretax_profit":    ["GuV-Rechnung", "P&L", "Income Statement"],
    "total_debt":       ["Bilanz", "Balance Sheet"],
    "cash":             ["Bilanz", "Balance Sheet"],
    "net_working_capital": ["Bilanz", "Balance Sheet"],
    "capex":            ["GuV-Rechnung", "P&L", "Bilanz", "Balance Sheet"],
}


def detect_currency_and_unit(unit_string: str) -> dict:
    s = str(unit_string).lower().strip()
    currency = "USD" if "usd" in s else "EUR" if "eur" in s else "?"
    if any(x in s for x in ["tsd", "tausend", "thousand"]):
        unit_label, scale_to_millions, raw_unit = "Tausend", 0.001, "tsd"
    elif any(x in s for x in ["mio", "million"]):
        unit_label, scale_to_millions, raw_unit = "Millionen", 1.0, "mio"
    elif any(x in s for x in ["mrd", "milliard", "billion"]):
        unit_label, scale_to_millions, raw_unit = "Milliarden", 1000.0, "mrd"
    else:
        unit_label, scale_to_millions, raw_unit = "Einzelwert", 0.000001, "units"
    return {
        "currency": currency,
        "unit_label": unit_label,
        "raw_unit": raw_unit,
        "scale_to_millions": scale_to_millions,
        "display": f"{currency} ({unit_label})",
        "symbol": "€" if currency == "EUR" else "$" if currency == "USD" else "",
    }


def excel_serial_to_year(serial) -> Optional[int]:
    try:
        s = int(float(str(serial)))
        if 40000 < s < 50000:
            import datetime
            d = date(1899, 12, 30) + datetime.timedelta(days=s)
            return d.year
        if 2000 <= s <= 2035:
            return s
    except Exception:
        pass
    return None


class MoodysOrbisParser:
    def __init__(self):
        self.sheets: dict = {}
        self.company_name: str = "Unknown"
        self.currency_info: dict = {}
        self.years: list = []
        self.raw_data: dict = {}
        self.warnings: list = []
        self.is_moodys_format: bool = False

    def load(self, file) -> bool:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file, data_only=True, read_only=True)
            self.sheets = {sname: list(wb[sname].iter_rows(values_only=True)) for sname in wb.sheetnames}
        except Exception as e:
            self.warnings.append(f"Ladefehler: {e}")
            return False

        moodys_keys = {"Bilanz", "GuV-Rechnung", "Kennzahlen", "Cover", "Balance Sheet", "P&L", "Income Statement"}
        self.is_moodys_format = len(moodys_keys & set(self.sheets.keys())) >= 2
        if not self.is_moodys_format:
            return False

        self._extract_company_name()
        self._extract_years_and_currency()
        self._extract_all_fields()
        return True

    def _extract_company_name(self):
        for sheet_key in ["Cover"] + list(self.sheets.keys()):
            if sheet_key in self.sheets:
                for row in self.sheets[sheet_key][:3]:
                    val = row[0] if row else None
                    if val and isinstance(val, str) and len(val.strip()) > 3:
                        self.company_name = val.strip()
                        return

    def _extract_years_and_currency(self):
        for sheet_name, rows in self.sheets.items():
            if sheet_name == "Cover" or len(rows) < 7:
                continue
            date_row = rows[4] if len(rows) > 4 else []
            unit_row = rows[5] if len(rows) > 5 else []
            years = [yr for cell in date_row[1:] if (yr := excel_serial_to_year(cell))]
            if years:
                self.years = years
            for cell in unit_row[1:]:
                if cell and str(cell).strip():
                    self.currency_info = detect_currency_and_unit(str(cell))
                    return
        if not self.currency_info:
            self.currency_info = detect_currency_and_unit("EUR")
            self.warnings.append("⚠️ Währung nicht erkannt – EUR als Default")

    def _extract_all_fields(self):
        for field, sheet_priority in MOODYS_SHEET_PRIORITY.items():
            found = False
            for sheet_name in sheet_priority:
                if sheet_name in self.sheets:
                    result = self._find_field_in_sheet(field, self.sheets[sheet_name])
                    if result is not None:
                        self.raw_data[field] = result
                        found = True
                        break
            if not found:
                for sheet_name, rows in self.sheets.items():
                    if sheet_name == "Cover":
                        continue
                    result = self._find_field_in_sheet(field, rows)
                    if result is not None:
                        self.raw_data[field] = result
                        found = True
                        break
            if not found:
                self.warnings.append(f"⚠️ '{field}' nicht gefunden")

    def _find_field_in_sheet(self, field: str, rows: list) -> Optional[dict]:
        aliases = ROW_ALIASES.get(field, [])
        for row in rows:
            if not row or row[0] is None:
                continue
            label_clean = str(row[0]).strip().lower().lstrip(" \t\xa0∟").strip()
            for alias in aliases:
                if label_clean == alias or label_clean.endswith(" " + alias):
                    values = list(row[1:])
                    result = {}
                    for i, yr in enumerate(self.years):
                        if i < len(values):
                            v = self._safe_float(values[i])
                            if v is not None:
                                result[yr] = v
                    if result:
                        return result
        return None

    def _safe_float(self, val) -> Optional[float]:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return None if val != val else float(val)
        s = str(val).strip().lower()
        if s in ("n.v.", "n.s.", "n/a", "", "-", "—"):
            return None
        try:
            return float(s.replace(".", "").replace(",", "."))
        except Exception:
            return None

    def get_latest_year_values(self) -> dict:
        if not self.years:
            return {}
        latest = self.years[0]
        result = {}
        for field, year_data in self.raw_data.items():
            if latest in year_data:
                result[field] = year_data[latest]
            elif year_data:
                result[field] = year_data[sorted(year_data.keys(), reverse=True)[0]]
        return result

    def get_timeseries(self) -> pd.DataFrame:
        if not self.years:
            return pd.DataFrame()
        data = {}
        for field, year_data in self.raw_data.items():
            data[field] = {yr: year_data.get(yr) for yr in self.years}
        return pd.DataFrame(data, index=self.years).sort_index(ascending=False)

    def build_company_inputs(self, manual_overrides={}, use_year=None) -> tuple:
        if use_year:
            vals = {f: self.raw_data[f].get(use_year) for f in self.raw_data
                    if use_year in self.raw_data.get(f, {})}
        else:
            vals = self.get_latest_year_values()
        vals.update(manual_overrides)
        warnings = list(self.warnings)

        def get(f, d=0.0):
            v = vals.get(f)
            if v is None:
                warnings.append(f"⚠️ '{f}' fehlt – Default verwendet")
                return d
            return v

        ebitda = get("ebitda")
        dep = get("depreciation")

        # Derive tax rate: steuern / gewinn_vor_steuern
        tax_expense = vals.get("tax_rate")       # actually tax expense (from "Steuern" row)
        pretax      = vals.get("pretax_profit")  # "Gewinn/Verlust vor Steuern"
        if tax_expense and pretax and pretax > 0 and tax_expense < pretax:
            tax_rate = tax_expense / pretax
        elif tax_expense and pretax and pretax > 0:
            tax_rate = 0.25
            warnings.append(f"⚠️ Steuersatz-Ableitung unplausibel ({tax_expense:.0f}/{pretax:.0f}) – 25% Default")
        else:
            tax_rate = 0.25
            warnings.append("⚠️ Steuersatz nicht ableitbar – 25% Default")

        # CapEx fallback: Moodys Orbis rarely has explicit CapEx
        # Use D&A as maintenance CapEx proxy if CapEx == 0 (conservative)
        capex_val = vals.get("capex")
        if not capex_val or capex_val == 0:
            capex_val = dep * 0.75  # maintenance CapEx ~ 75% of D&A
            warnings.append("ℹ️ CapEx nicht gefunden – 75% der D&A als Maintenance-CapEx angesetzt")

        # Total debt: sum long-term + short-term if both available
        debt_val = get("total_debt")
        # If debt = 0 but we have both lt and st entries, it means the company is debt-free
        # This is valid for Glock

        inputs = CompanyInputs(
            revenue=get("revenue"), ebitda=ebitda,
            ebit=vals.get("ebit") or (ebitda - dep),
            depreciation=dep, interest_expense=get("interest_expense"),
            tax_rate=tax_rate, total_debt=debt_val,
            cash=get("cash"), net_working_capital=get("net_working_capital"),
            capex=capex_val, company_name=self.company_name,
        )
        return inputs, warnings, self.currency_info


class GenericExcelParser:
    def __init__(self):
        self.df = None
        self.currency_info = {}
        self.company_name = "Target"
        self.mapping = {}
        self.unmapped = []
        self.warnings = []

    def load(self, file) -> bool:
        try:
            xl = pd.ExcelFile(file, engine="openpyxl")
            preferred = [s for s in xl.sheet_names if any(
                kw in s.lower() for kw in ["guv", "p&l", "income", "bilanz", "balance"])]
            sheet = preferred[0] if preferred else xl.sheet_names[0]
            df = pd.read_excel(file, sheet_name=sheet, header=None, engine="openpyxl")
            self.df = self._set_header(df)
            self._detect_currency()
            return True
        except Exception as e:
            self.warnings.append(f"Fehler: {e}")
            return False

    def _set_header(self, df):
        for i, row in df.iterrows():
            vals = [str(v).lower() for v in row if pd.notna(v)]
            if any(kw in " ".join(vals) for kw in ["umsatz", "revenue", "ebitda"]):
                df.columns = df.iloc[i]
                return df.iloc[i+1:].reset_index(drop=True)
        df.columns = df.iloc[0]
        return df.iloc[1:].reset_index(drop=True)

    def _detect_currency(self):
        if self.df is None:
            self.currency_info = detect_currency_and_unit("EUR")
            return
        for col in self.df.columns:
            s = str(col).lower()
            if any(x in s for x in ["usd", "eur", "tsd", "mio"]):
                self.currency_info = detect_currency_and_unit(col)
                return
        self.currency_info = detect_currency_and_unit("EUR")

    def auto_map(self) -> dict:
        if self.df is None:
            return {"mapped": {}, "unmapped": list(REQUIRED_FIELDS.keys()), "available_columns": []}
        available = list(self.df.columns)
        mapped, unmapped = {}, []
        for field, aliases in ROW_ALIASES.items():
            found = None
            for alias in aliases:
                for col in available:
                    if alias in str(col).lower():
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
        return {"mapped": mapped, "unmapped": unmapped, "available_columns": available}

    def build_company_inputs(self, manual_overrides={}) -> tuple:
        vals = manual_overrides
        warnings = list(self.warnings)
        def get(f, d=0.0):
            v = vals.get(f)
            if v is None:
                warnings.append(f"⚠️ '{f}' fehlt")
                return d
            return v
        ebitda = get("ebitda")
        dep = get("depreciation")
        tax = get("tax_rate", 0.25)
        if tax > 1:
            tax /= 100.0
        inputs = CompanyInputs(
            revenue=get("revenue"), ebitda=ebitda,
            ebit=vals.get("ebit") or (ebitda - dep),
            depreciation=dep, interest_expense=get("interest_expense"),
            tax_rate=tax, total_debt=get("total_debt"), cash=get("cash"),
            net_working_capital=get("net_working_capital"), capex=get("capex"),
            company_name=self.company_name,
        )
        return inputs, warnings, self.currency_info


def parse_file(file) -> tuple:
    moodys = MoodysOrbisParser()
    if moodys.load(file):
        return moodys, True
    generic = GenericExcelParser()
    generic.load(file)
    return generic, False


def generate_sample_excel(path: str = "data/sample/sample_company.xlsx"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "Kennzahl": ["∟ Umsatz", "∟ EBITDA", "∟ Betriebsgewinn/-verlust [=EBIT]",
                     "∟ Wertminderungen & Abschreibungen", "∟ Zinsaufwand",
                     "∟ Steuern", "Langfristige Finanzschulden",
                     "Zahlungsmittel & Zahlungsmitteläquivalente", "Working Capital", "CapEx"],
        "2022 (tsd EUR)": [48500, 11900, 8900, 3000, 1500, 3100, 20000, 4500, 6800, 2500],
        "2024 (tsd EUR)": [56800, 13900, 10200, 3700, 1300, 3600, 17500, 6100, 7600, 2800],
    }
    pd.DataFrame(data).to_excel(path, index=False)
    return path