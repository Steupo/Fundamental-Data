import duckdb
import pandas as pd
from pathlib import Path
from openpyxl.styles import Font, PatternFill, Alignment

DB_PATH = "fundamental_universe.duckdb"
EXPORT_DIR = Path("exports_framework")

PCT_METRICS = {
    "Gross Margin", "R&D / Sales", "EBITDA Margin", "EBIT Margin",
    "Tax Rate", "Net Margin", "Return on Equity", "Return on Assets",
    "Return on Invested Capital"
}
PER_SHARE_METRICS = {"EPS", "FCF / Share", "Tangible Book Value per Share"}

ORDERED_CONSOLIDATED_ROWS = [
    ("Revenue", True),
    ("Cost of Good Sold", True),
    ("Gross Profit", True),
    ("Gross Margin", True),
    ("Opex", True),
    ("SG&A", False),
    ("Sales & Marketing", False),
    ("General & Administrative", False),
    ("R&D", False),
    ("R&D / Sales", False),
    ("In Process R&D", False),
    ("Fulfillment / Store operating", False),
    ("Restructuring & Impairment", False),
    ("Other Operating", False),
    ("EBITDA", True),
    ("EBITDA Margin", True),
    ("Depreciation & Amortization", True),
    ("EBIT", True),
    ("EBIT Margin", True),
    ("Cost of Interest", True),
    ("Pre-Tax Income", True),
    ("Tax", True),
    ("Tax Rate", True),
    ("Net Income", True),
    ("Net Margin", True),
    ("Total Share Outstanding", True),
    ("EPS", True),
    ("Cash flow from operations", True),
    ("CAPEX", True),
    ("Free Cash Flow", True),
    ("FCF / Share", True),
    ("Stock-Based Compensation", True),
    ("FCF ex-SBC", True),
    ("Cash Flow from Investing", True),
    ("Cash flow from financing", True),
    ("Dividends Paid", True),
    ("Share Buybacks", True),
    ("Cash & Short-Term Investments", True),
    ("Working Capital", True),
    ("Change in Working Capital", True),
    ("Short Term Debt", True),
    ("Long Term Debt", True),
    ("Total Debt", True),
    ("Net Debt", True),
    ("Net Debt / EBITDA", True),
    ("Net Debt / Free cash Flow", True),
    ("Total Debt / Free Cash Flow", True),
    ("Total Assets", True),
    ("Shareholders Equity", True),
    ("Goodwill", True),
    ("Intangible Assets", True),
    ("Tangible Book Value per Share", True),
    ("Number of Employee", True),
    ("Revenue per employee", True),
    ("Return on Equity", True),
    ("Return on Assets", True),
    ("Return on Invested Capital", True)
]

SEGMENT_PARENT_BLOCKS = {
    "Revenue": ("Revenue", False),
    "Gross Profit": ("Gross Profit", False),
    "Gross Margin": ("Gross Margin", True),
    "EBITDA": ("EBITDA", False),
    "EBITDA Margin": ("EBITDA Margin", True),
    "EBIT": ("EBIT", False),
    "EBIT Margin": ("EBIT Margin", True)
}


def build_company_sheets(con: duckdb.DuckDBPyConnection, ticker: str, form_type: str) -> tuple[pd.DataFrame, dict]:
    """Construit la matrice complète (Consolidé + Divisions dynamiques) pour 10-K ou 10-Q."""
    if form_type == "10-K":
        df_cons = con.execute("""
            SELECT * FROM vw_annual_framework
            WHERE ticker = ?
            ORDER BY fiscal_year DESC
        """, [ticker]).df()
        if df_cons.empty:
            return pd.DataFrame(), {}
        period_cols = df_cons["fiscal_year"].astype(str).tolist()
        df_cons = df_cons.set_index(df_cons["fiscal_year"].astype(str))
    else:
        df_cons = con.execute("""
            SELECT * FROM vw_quarterly_framework
            WHERE ticker = ?
            ORDER BY sort_key DESC
        """, [ticker]).df()
        if df_cons.empty:
            return pd.DataFrame(), {}
        period_cols = df_cons["period_label"].tolist()
        df_cons = df_cons.set_index("period_label")

    # Récupération des segments actifs (classés par CA le plus récent, max 10)
    df_seg = con.execute("""
        SELECT
            fiscal_year,
            fiscal_quarter,
            (CAST(fiscal_year AS VARCHAR) || '-' || fiscal_quarter) AS period_label,
            segment_code,
            segment_label,
            metric,
            value
        FROM segment_fundamentals
        WHERE ticker = ? AND form = ?
    """, [ticker, form_type]).df()

    active_segments = []
    seg_lookup = {}
    if not df_seg.empty:
        # Classement des segments par CA décroissant sur la dernière période disponible
        rev_seg = df_seg[df_seg["metric"] == "Revenue"]
        if not rev_seg.empty:
            ranked = (
                rev_seg.groupby(["segment_code", "segment_label"])["value"]
                .max()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            active_segments = list(zip(ranked["segment_code"], ranked["segment_label"]))
        else:
            uniq = df_seg[["segment_code", "segment_label"]].drop_duplicates().head(10)
            active_segments = list(zip(uniq["segment_code"], uniq["segment_label"]))

        p_col = "fiscal_year" if form_type == "10-K" else "period_label"
        for _, r in df_seg.iterrows():
            key = (str(r[p_col]), r["segment_code"], r["metric"])
            seg_lookup[key] = r["value"]

    rows_data = {}
    row_styles = {}  # label -> 'BOLD' | 'ITALIC' | 'NORMAL'

    for metric_label, is_bold in ORDERED_CONSOLIDATED_ROWS:
        if metric_label in df_cons.columns:
            rows_data[metric_label] = [df_cons.loc[p, metric_label] for p in period_cols]
        else:
            rows_data[metric_label] = [None] * len(period_cols)
        row_styles[metric_label] = "BOLD" if is_bold else "NORMAL"

        # Insertion dynamique des divisions actives juste sous le poste consolidé parent
        if metric_label in SEGMENT_PARENT_BLOCKS and active_segments:
            base_metric, is_margin = SEGMENT_PARENT_BLOCKS[metric_label]
            for seg_code, seg_label in active_segments:
                div_row_name = f"  {metric_label} - {seg_label}"
                vals = []
                for p in period_cols:
                    if is_margin:
                        num = seg_lookup.get((p, seg_code, base_metric.replace(" Margin", "")))
                        den = seg_lookup.get((p, seg_code, "Revenue"))
                        vals.append((num / den) if (num is not None and den not in (None, 0)) else None)
                    elif base_metric == "EBITDA":
                        ebit_s = seg_lookup.get((p, seg_code, "EBIT"))
                        da_s = seg_lookup.get((p, seg_code, "Depreciation & Amortization"))
                        vals.append((ebit_s + da_s) if (ebit_s is not None and da_s is not None) else None)
                    else:
                        vals.append(seg_lookup.get((p, seg_code, base_metric)))
                rows_data[div_row_name] = vals
                row_styles[div_row_name] = "ITALIC"

    df_final = pd.DataFrame.from_dict(rows_data, orient="index", columns=period_cols)
    return df_final, row_styles


def export_company_excel(ticker: str):
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DB_PATH, read_only=True)
    out_path = EXPORT_DIR / f"{ticker.upper()}_Full_Framework.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for form_type, sheet_name in [("10-K", "Annual Data 10K"), ("10-Q", "Quarterly Data 10Q")]:
            df_sheet, styles = build_company_sheets(con, ticker.upper(), form_type)
            if df_sheet.empty:
                continue
            df_sheet.to_excel(writer, sheet_name=sheet_name)
            ws = writer.sheets[sheet_name]

            header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
            header_font = Font(color="FFFFFF", bold=True)
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center")

            for r_idx, row_label in enumerate(df_sheet.index, start=2):
                style = styles.get(row_label, "NORMAL")
                label_cell = ws.cell(row=r_idx, column=1)
                label_cell.font = Font(bold=(style == "BOLD"), italic=(style == "ITALIC"))

                clean_metric = row_label.strip().split(" - ")[0]
                for c_idx in range(2, len(df_sheet.columns) + 2):
                    cell = ws.cell(row=r_idx, column=c_idx)
                    if clean_metric in PCT_METRICS:
                        cell.number_format = "0.0%"
                    elif clean_metric in PER_SHARE_METRICS:
                        cell.number_format = "#,##0.00"
                    else:
                        cell.number_format = "#,##0"

            ws.column_dimensions["A"].width = 42
            ws.freeze_panes = "B2"

    con.close()
    print(f"[OK] Classeur complet généré : {out_path}")


if __name__ == "__main__":
    export_company_excel("AAPL")
    export_company_excel("MSFT")
    export_company_excel("AMZN")