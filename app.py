import io
import numpy as np
import pandas as pd
import duckdb
import streamlit as st
from openpyxl.styles import Font, PatternFill, Alignment
from Screener import build_company_sheets, PCT_METRICS, PER_SHARE_METRICS

st.set_page_config(
    page_title="SEC Fundamental & DCF Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    @media (max-width: 768px) {
        .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
        div[data-testid="stMetricValue"] { font-size: 1.1rem; }
    }
</style>
""", unsafe_allow_html=True)

# Repasser à "fundamental_universe.duckdb" une fois Dataextractor.py terminé
DB_PATH = "preview.duckdb"

@st.cache_data(ttl=300)
def load_universe():
    con = duckdb.connect(DB_PATH, read_only=True)
    tickers = con.execute(
        "SELECT DISTINCT ticker FROM raw_fundamentals ORDER BY ticker"
    ).df()["ticker"].tolist()
    names = dict(
        con.execute("SELECT ticker, company_name FROM universe_metadata").fetchall()
    )
    con.close()
    return tickers, names

@st.cache_data(ttl=300)
def load_company_data(ticker: str):
    con = duckdb.connect(DB_PATH, read_only=True)
    df_10k, styles_10k = build_company_sheets(con, ticker, "10-K")
    df_10q, styles_10q = build_company_sheets(con, ticker, "10-Q")
    con.close()
    return (df_10k, styles_10k), (df_10q, styles_10q)

extracted_tickers, company_names = load_universe()

if not extracted_tickers:
    st.warning("Aucune donnée en base. Exécute d'abord : python3 Dataextractor.py")
    st.stop()

# ==============================================================================
# NAVIGATION & PARAMÈTRES GLOBAUX (SIDEBAR)
# ==============================================================================
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Vue active",
    ["Analyse Fondamentale & Synthèse", "Valorisation DCF (Consolidé & Divisions)"],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.header("Société")
selected_ticker = st.sidebar.selectbox(
    "Sélectionner un émetteur",
    extracted_tickers,
    format_func=lambda t: f"{t} — {company_names.get(t, '')}"
)

unit_choice = st.sidebar.radio(
    "Échelle d'affichage (montants)",
    ["Millions ($M)", "Milliards ($B)", "Brut ($)"],
    index=0
)
UNIT_DIVISOR = {"Millions ($M)": 1e6, "Milliards ($B)": 1e9, "Brut ($)": 1.0}[unit_choice]

(df_10k, styles_10k), (df_10q, styles_10q) = load_company_data(selected_ticker)

# ==============================================================================
# UTILITAIRES EXTRACTION DERNIÈRE VALEUR (10-Q PRIORITAIRE, REPLI 10-K)
# ==============================================================================
def get_latest_metric(df_q: pd.DataFrame, df_k: pd.DataFrame, row_candidates: list):
    """Cherche la valeur non-nulle la plus récente dans le 10-Q en priorité, puis dans le 10-K."""
    for source_label, df in [("10-Q", df_q), ("10-K", df_k)]:
        if df.empty:
            continue
        for row_name in row_candidates:
            if row_name in df.index:
                s = pd.to_numeric(df.loc[row_name], errors="coerce").dropna()
                if not s.empty:
                    latest_col = s.index[0]
                    return float(s.iloc[0]), f"{source_label} ({latest_col})"
    return 0.0, "Indisponible"

# ==============================================================================
# FONCTIONS DE CALCUL DES TABLEAUX SYNTHÉTIQUES (CROISSANCE & MARGES)
# ==============================================================================
def get_series(df: pd.DataFrame, row_name: str) -> pd.Series:
    if df.empty or row_name not in df.index:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df.loc[row_name], errors="coerce")
    return s.sort_index()

def compute_cagr(series: pd.Series, years: int) -> float:
    if len(series) <= years:
        return np.nan
    end_val = series.iloc[-1]
    start_val = series.iloc[-(years + 1)]
    if pd.isna(end_val) or pd.isna(start_val) or start_val <= 0 or end_val <= 0:
        return np.nan
    return (end_val / start_val) ** (1.0 / years) - 1.0

def compute_avg_ratio(num_series: pd.Series, den_series: pd.Series, window: int) -> float:
    aligned = pd.concat([num_series, den_series], axis=1).dropna()
    if len(aligned) == 0:
        return np.nan
    sub = aligned.iloc[-window:]
    if len(sub) < window:
        return np.nan
    ratios = sub.iloc[:, 0] / sub.iloc[:, 1].replace(0, np.nan)
    return ratios.mean()

def build_annual_summary_tables(df_10k: pd.DataFrame):
    horizons = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    cols = [f"{h}Y" for h in horizons]

    rev = get_series(df_10k, "Revenue")
    opex = get_series(df_10k, "Operating Expenses")
    ebitda = get_series(df_10k, "EBITDA")
    ebit = get_series(df_10k, "EBIT")
    ni = get_series(df_10k, "Net Income")
    eps = get_series(df_10k, "EPS Diluted")
    if eps.dropna().empty:
        eps = get_series(df_10k, "EPS Basic")
    capex = get_series(df_10k, "Capex").abs()
    fcf = get_series(df_10k, "Free Cash Flow")
    tot_debt = get_series(df_10k, "Total Debt")
    net_debt = get_series(df_10k, "Net Debt")

    growth_map = {
        "Revenue Growth CAGR": rev,
        "OPEX Growth CAGR": opex,
        "EBITDA Growth CAGR": ebitda,
        "EBIT Growth CAGR": ebit,
        "Net Income Growth CAGR": ni,
        "EPS Growth CAGR": eps,
        "CAPEX Growth CAGR": capex,
        "FCF Growth CAGR": fcf,
        "Total Debt Growth CAGR": tot_debt,
        "Net Debt Growth CAGR": net_debt,
    }

    df_growth = pd.DataFrame(index=list(growth_map.keys()), columns=cols, dtype=float)
    for label, s in growth_map.items():
        for h in horizons:
            df_growth.loc[label, f"{h}Y"] = compute_cagr(s, h)

    margin_rows = [
        "EBITDA Margin", "EBIT Margin", "Net Income Margin",
        "FCF Margin", "Total Debt / EBITDA", "Net Debt / FCF"
    ]
    df_margins = pd.DataFrame(index=margin_rows, columns=cols, dtype=float)
    for h in horizons:
        col = f"{h}Y"
        df_margins.loc["EBITDA Margin", col] = compute_avg_ratio(ebitda, rev, h)
        df_margins.loc["EBIT Margin", col] = compute_avg_ratio(ebit, rev, h)
        df_margins.loc["Net Income Margin", col] = compute_avg_ratio(ni, rev, h)
        df_margins.loc["FCF Margin", col] = compute_avg_ratio(fcf, rev, h)
        df_margins.loc["Total Debt / EBITDA", col] = compute_avg_ratio(tot_debt, ebitda, h)
        df_margins.loc["Net Debt / FCF", col] = compute_avg_ratio(net_debt, fcf, h)

    return df_growth, df_margins

def build_quarterly_summary_tables(df_10q: pd.DataFrame):
    q_labels = ["Q", "Q-1", "Q-2", "Q-3", "Q-4", "Q-5", "Q-6", "Q-7", "Q-8"]

    rev = get_series(df_10q, "Revenue")
    opex = get_series(df_10q, "Operating Expenses")
    ebitda = get_series(df_10q, "EBITDA")
    ebit = get_series(df_10q, "EBIT")
    ni = get_series(df_10q, "Net Income")
    eps = get_series(df_10q, "EPS Diluted")
    if eps.dropna().empty:
        eps = get_series(df_10q, "EPS Basic")
    capex = get_series(df_10q, "Capex").abs()
    fcf = get_series(df_10q, "Free Cash Flow")
    tot_debt = get_series(df_10q, "Total Debt")
    net_debt = get_series(df_10q, "Net Debt")

    def yoy_q(s: pd.Series) -> pd.Series:
        res = pd.Series(index=s.index, dtype=float)
        for i in range(4, len(s)):
            curr, prev = s.iloc[i], s.iloc[i - 4]
            if pd.notna(curr) and pd.notna(prev) and prev > 0:
                res.iloc[i] = (curr / prev) - 1.0
        return res

    growth_series = {
        "Revenue Growth (YoY)": yoy_q(rev),
        "OPEX Growth (YoY)": yoy_q(opex),
        "EBITDA Growth (YoY)": yoy_q(ebitda),
        "EBIT Growth (YoY)": yoy_q(ebit),
        "Net Income Growth (YoY)": yoy_q(ni),
        "EPS Growth (YoY)": yoy_q(eps),
        "CAPEX Growth (YoY)": yoy_q(capex),
        "FCF Growth (YoY)": yoy_q(fcf),
        "Total Debt Growth (YoY)": yoy_q(tot_debt),
        "Net Debt Growth (YoY)": yoy_q(net_debt),
    }

    df_q_growth = pd.DataFrame(index=list(growth_series.keys()), columns=q_labels, dtype=float)
    for label, s in growth_series.items():
        for idx_offset, q_col in enumerate(q_labels):
            pos = len(s) - 1 - idx_offset
            if pos >= 0:
                df_q_growth.loc[label, q_col] = s.iloc[pos]

    ebitda_ltm = ebitda.rolling(4).sum()
    fcf_ltm = fcf.rolling(4).sum()

    margin_series = {
        "EBITDA Margin": ebitda / rev.replace(0, np.nan),
        "EBIT Margin": ebit / rev.replace(0, np.nan),
        "Net Income Margin": ni / rev.replace(0, np.nan),
        "FCF Margin": fcf / rev.replace(0, np.nan),
        "Total Debt / EBITDA (LTM)": tot_debt / ebitda_ltm.replace(0, np.nan),
        "Net Debt / FCF (LTM)": net_debt / fcf_ltm.replace(0, np.nan),
    }

    df_q_margins = pd.DataFrame(index=list(margin_series.keys()), columns=q_labels, dtype=float)
    for label, s in margin_series.items():
        for idx_offset, q_col in enumerate(q_labels):
            pos = len(s) - 1 - idx_offset
            if pos >= 0:
                df_q_margins.loc[label, q_col] = s.iloc[pos]

    return df_q_growth, df_q_margins

# ==============================================================================
# FORMATAGE & STYLER CONDITIONNEL
# ==============================================================================
def style_summary_table(df: pd.DataFrame, is_margin_table: bool = False):
    def fmt_cell(val, row_name):
        if pd.isna(val):
            return "—"
        if is_margin_table and ("/ EBITDA" in row_name or "/ FCF" in row_name):
            return f"{val:.2f}x"
        return f"{val * 100:.1f}%"

    formatted = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)
    for r in df.index:
        for c in df.columns:
            formatted.loc[r, c] = fmt_cell(df.loc[r, c], r)

    def color_cells(val_str):
        if val_str == "—" or "x" in val_str:
            return ""
        try:
            num = float(val_str.replace("%", ""))
            if num > 0:
                return "color: #16a34a; font-weight: 500;"
            elif num < 0:
                return "color: #dc2626; font-weight: 500;"
        except ValueError:
            pass
        return ""

    return formatted.style.map(color_cells)

def format_statement_styler(df_sheet: pd.DataFrame, styles: dict, divisor: float):
    df_disp = df_sheet.copy().astype(object)
    for row_label in df_disp.index:
        clean_metric = row_label.strip().split(" - ")[0]
        for col in df_disp.columns:
            val = df_disp.loc[row_label, col]
            if pd.isna(val):
                df_disp.loc[row_label, col] = ""
            elif clean_metric in PCT_METRICS:
                df_disp.loc[row_label, col] = f"{val * 100:.1f}%"
            elif clean_metric in PER_SHARE_METRICS:
                df_disp.loc[row_label, col] = f"{val:,.2f}"
            else:
                df_disp.loc[row_label, col] = f"{val / divisor:,.1f}" if divisor > 1 else f"{val:,.0f}"

    def highlight_rows(row):
        style_type = styles.get(row.name, "NORMAL")
        if style_type == "BOLD":
            return ["font-weight: 700; background-color: rgba(31, 78, 120, 0.12);"] * len(row)
        elif style_type == "ITALIC":
            return ["font-style: italic; color: #94a3b8;"] * len(row)
        return [""] * len(row)

    return df_disp.style.apply(highlight_rows, axis=1)

def generate_excel_buffer(ticker: str, sheets: dict) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, (df_sheet, styles) in sheets.items():
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
                ws.cell(row=r_idx, column=1).font = Font(
                    bold=(style == "BOLD"), italic=(style == "ITALIC")
                )
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
    return buffer.getvalue()

# ==============================================================================
# VUE 1 : ANALYSE FONDAMENTALE & TABLEAUX SYNTHÉTIQUES
# ==============================================================================
if page == "Analyse Fondamentale & Synthèse":
    col_title, col_action = st.columns([3, 1])
    with col_title:
        st.title(f"{selected_ticker} — {company_names.get(selected_ticker, '')}")
    with col_action:
        st.write("")
        excel_bytes = generate_excel_buffer(
            selected_ticker,
            {"Annual Data 10K": (df_10k, styles_10k), "Quarterly Data 10Q": (df_10q, styles_10q)}
        )
        st.download_button(
            label="📥 Export Excel Complet",
            data=excel_bytes,
            file_name=f"{selected_ticker}_Full_Framework.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch"
        )

    if not df_10k.empty:
        latest_col = df_10k.columns[0]
        rev_last = df_10k.loc["Revenue", latest_col] if "Revenue" in df_10k.index else np.nan
        ebit_last = df_10k.loc["EBIT", latest_col] if "EBIT" in df_10k.index else np.nan
        fcf_last = df_10k.loc["Free Cash Flow", latest_col] if "Free Cash Flow" in df_10k.index else np.nan
        nd_last, nd_src = get_latest_metric(df_10q, df_10k, ["Net Debt"])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Revenue ({latest_col})", f"{rev_last / UNIT_DIVISOR:,.1f}" if pd.notna(rev_last) else "N/A")
        k2.metric(f"EBIT ({latest_col})", f"{ebit_last / UNIT_DIVISOR:,.1f}" if pd.notna(ebit_last) else "N/A")
        k3.metric(f"Free Cash Flow ({latest_col})", f"{fcf_last / UNIT_DIVISOR:,.1f}" if pd.notna(fcf_last) else "N/A")
        k4.metric(f"Net Debt ({nd_src})", f"{nd_last / UNIT_DIVISOR:,.1f}")

    st.divider()

    st.subheader("Synthèse Opérationnelle : Croissance & Marges")
    tab_ann_sum, tab_q_sum = st.tabs(["Synthèse Annuelle (1Y à 25Y)", "Synthèse Trimestrielle (Q à Q-8)"])

    with tab_ann_sum:
        if not df_10k.empty:
            df_g_ann, df_m_ann = build_annual_summary_tables(df_10k)
            st.markdown("**Historical Growth (CAGR)**")
            st.dataframe(style_summary_table(df_g_ann, is_margin_table=False), width="stretch")
            st.markdown("**Historical Margins & Leverage (Moyennes sur la période)**")
            st.dataframe(style_summary_table(df_m_ann, is_margin_table=True), width="stretch")
        else:
            st.info("Données annuelles 10-K indisponibles pour ce ticker.")

    with tab_q_sum:
        if not df_10q.empty:
            df_g_q, df_m_q = build_quarterly_summary_tables(df_10q)
            st.markdown("**Historical Growth Quarterly Data (YoY vs Q-4)**")
            st.dataframe(style_summary_table(df_g_q, is_margin_table=False), width="stretch")
            st.markdown("**Historical Margins Quarterly Data (Leviers sur EBITDA/FCF LTM)**")
            st.dataframe(style_summary_table(df_m_q, is_margin_table=True), width="stretch")
        else:
            st.info("Données trimestrielles 10-Q indisponibles pour ce ticker.")

    st.divider()

    st.subheader("États Financiers & Divisions (Filtrables)")
    freq_choice = st.radio("Périodicité des états détaillés", ["Annuel (10-K)", "Trimestriel (10-Q)"], horizontal=True)
    active_df, active_styles = (df_10k, styles_10k) if freq_choice == "Annuel (10-K)" else (df_10q, styles_10q)

    if not active_df.empty:
        with st.expander("Filtres d'affichage (Lignes, Divisions & Périodes)", expanded=True):
            f_col1, f_col2, f_col3 = st.columns([1.2, 1.5, 1.3])
            with f_col1:
                view_preset = st.selectbox(
                    "Préréglage de lignes",
                    [
                        "Toutes les lignes",
                        "Aggregats Clés Uniquement (gras)",
                        "Compte de Résultat & Divisions",
                        "Flux de Trésorerie & Bilan",
                        "Uniquement Divisions / Segments"
                    ]
                )
            with f_col2:
                all_rows = list(active_df.index)
                if view_preset == "Aggregats Clés Uniquement (gras)":
                    default_rows = [r for r in all_rows if active_styles.get(r) == "BOLD"]
                elif view_preset == "Compte de Résultat & Divisions":
                    default_rows = [r for r in all_rows if any(k in r for k in ["Revenue", "Gross Profit", "Operating", "EBIT", "EBITDA", "Net Income", "EPS"])]
                elif view_preset == "Flux de Trésorerie & Bilan":
                    default_rows = [r for r in all_rows if any(k in r for k in ["Cash", "Capex", "Free Cash Flow", "Debt", "Equity", "Assets", "Liabilities"])]
                elif view_preset == "Uniquement Divisions / Segments":
                    default_rows = [r for r in all_rows if active_styles.get(r) == "ITALIC" or r.strip() in ["Revenue", "Operating Income", "EBIT"]]
                else:
                    default_rows = all_rows

                selected_rows = st.multiselect(
                    "Lignes affichées",
                    options=all_rows,
                    default=default_rows
                )
            with f_col3:
                all_cols = list(active_df.columns)
                default_cols = all_cols[:min(8, len(all_cols))]
                selected_cols = st.multiselect(
                    "Périodes affichées (mobile-friendly)",
                    options=all_cols,
                    default=default_cols
                )

        if selected_rows and selected_cols:
            filtered_df = active_df.loc[selected_rows, selected_cols]
            st.dataframe(
                format_statement_styler(filtered_df, active_styles, UNIT_DIVISOR),
                width="stretch",
                height=650
            )
        else:
            st.warning("Sélectionne au moins une ligne et une période dans les filtres.")

# ==============================================================================
# VUE 2 : VALORISATION DCF (CONSOLIDÉ & SUM-OF-THE-PARTS PAR DIVISION)
# ==============================================================================
elif page == "Valorisation DCF (Consolidé & Divisions)":
    st.title(f"Modèle DCF Multi-Périodes — {selected_ticker}")

    if df_10k.empty:
        st.error("Données 10-K requises pour initialiser la base opérationnelle du DCF.")
        st.stop()

    latest_fy = df_10k.columns[0]
    base_rev = float(df_10k.loc["Revenue", latest_fy]) if "Revenue" in df_10k.index and pd.notna(df_10k.loc["Revenue", latest_fy]) else 0.0
    base_fcf = float(df_10k.loc["Free Cash Flow", latest_fy]) if "Free Cash Flow" in df_10k.index and pd.notna(df_10k.loc["Free Cash Flow", latest_fy]) else 0.0

    # Priorité 10-Q puis repli 10-K pour la Dette Nette et le Nombre d'Actions (en unités)
    latest_nd, nd_source = get_latest_metric(df_10q, df_10k, ["Net Debt"])
    latest_shares_units, shares_source = get_latest_metric(
        df_10q,
        df_10k,
        [
            "Diluted Shares",
            "Basic Shares",
            "Shares Outstanding",
            "Weighted Average Shares Diluted",
            "Weighted Average Shares Basic",
            "Common Stock Shares Outstanding"
        ]
    )

    default_fcf_margin = round((base_fcf / base_rev) * 100, 2) if base_rev > 0 else 15.0

    dcf_mode = st.radio(
        "Architecture de valorisation",
        ["Société Consolidée", "Par Divisions (Sum-of-the-Parts / SOTP)"],
        horizontal=True
    )

    st.info(
        f"Base opérationnelle ({latest_fy}) : Revenue **{base_rev / 1e6:,.1f} M$** | "
        f"FCF **{base_fcf / 1e6:,.1f} M$** ({default_fcf_margin:.2f}%)  —  "
        f"Bilan & Actions : Dette Nette **{latest_nd / 1e6:,.1f} M$** *[{nd_source}]* | "
        f"Actions **{int(round(latest_shares_units)):,}** *[{shares_source}]*"
    )

    # Bloc Equity Bridge (100 % modifiable à la main, actions en unités, buybacks actualisés)
    st.subheader("Pont Valeur d'Entreprise → Valeur Intrinsèque / Action")
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        remaining_buybacks_m = st.number_input(
            "Enveloppe Share Buybacks (M$)",
            value=0.0,
            step=1000.0,
            format="%.2f",
            help="Enveloppe totale de rachats d'actions prévue (en M$), étalée linéairement et actualisée au WACC."
        )
    with b_col2:
        buyback_years = st.number_input(
            "Horizon Buybacks (Années)",
            value=10,
            min_value=1,
            max_value=30,
            step=1,
            format="%d",
            help="Nombre d'années sur lequel l'enveloppe de rachats est exécutée et actualisée au WACC."
        )
    with b_col3:
        net_debt_input_m = st.number_input(
            f"Dette Nette soustraite (M$) — {nd_source}",
            value=round(latest_nd / 1e6, 2),
            step=100.0,
            format="%.2f",
            help="Positif = dette nette soustraite, Négatif = trésorerie nette ajoutée."
        )
    with b_col4:
        shares_input_units = st.number_input(
            f"Nombre d'actions (Unités) — {shares_source}",
            value=int(round(latest_shares_units)) if latest_shares_units > 0 else 1,
            min_value=1,
            step=100000,
            format="%d",
            help="Nombre total d'actions diluées en unités exactes."
        )

    st.divider()

    def discount_buybacks_m(total_buybacks_m: float, years: int, wacc_pct: float) -> float:
        """Actualise au WACC une enveloppe de rachats d'actions répartie linéairement sur `years` années."""
        if total_buybacks_m == 0 or years <= 0:
            return 0.0
        wacc = wacc_pct / 100.0
        annual_bb = total_buybacks_m / float(years)
        if wacc == 0:
            return total_buybacks_m
        return sum(annual_bb / ((1.0 + wacc) ** t) for t in range(1, years + 1))

    def run_dcf_from_fcf(fcf_0_m: float, cagrs_pct: list, lt_growth_pct: float, wacc_pct: float):
        """
        Projette le FCF initial (en M$) sur les périodes renseignées (5 ans chacune).
        Dès qu'une période est laissée vide (None / NaN), l'actualisation explicite s'arrête
        et la Valeur Terminale prend immédiatement le relais sur le dernier FCF projeté (FCF_N).
        """
        wacc = wacc_pct / 100.0
        g_lt = lt_growth_pct / 100.0

        if wacc <= g_lt:
            return None, "Erreur : le WACC doit être strictement supérieur au taux de croissance à long terme (LT Growth Rate)."

        # Conserver uniquement les périodes renseignées consécutivement à partir de P1
        active_cagrs = []
        for idx, c in enumerate(cagrs_pct):
            if c is None or (isinstance(c, float) and np.isnan(c)):
                # Vérifier qu'une période ultérieure n'a pas été remplie après un trou
                later_filled = [x for x in cagrs_pct[idx + 1:] if x is not None and not (isinstance(x, float) and np.isnan(x))]
                if later_filled:
                    return None, f"Erreur : la période P{idx + 1} est vide alors qu'une période suivante est renseignée."
                break
            active_cagrs.append(float(c))

        if not active_cagrs:
            return None, "Erreur : renseigne au moins le taux de croissance de la Période 1 (P1)."

        records = []
        curr_fcf = fcf_0_m
        pv_sum = 0.0

        for period_idx, cagr_pct in enumerate(active_cagrs, start=1):
            g = cagr_pct / 100.0
            for yr_in_period in range(1, 6):
                t = (period_idx - 1) * 5 + yr_in_period
                curr_fcf = curr_fcf * (1.0 + g)
                df_factor = (1.0 + wacc) ** t
                pv_fcf = curr_fcf / df_factor
                pv_sum += pv_fcf
                records.append({
                    "Année": f"Y{t}",
                    "Période": f"P{period_idx}",
                    "CAGR": f"{cagr_pct:.2f}%",
                    "FCF (M$)": curr_fcf,
                    "PV FCF (M$)": pv_fcf
                })

        horizon_years = len(records)
        fcf_terminal_base = records[-1]["FCF (M$)"]
        tv_undiscounted = (fcf_terminal_base * (1.0 + g_lt)) / (wacc - g_lt)
        pv_tv = tv_undiscounted / ((1.0 + wacc) ** horizon_years)
        ev_m = pv_sum + pv_tv

        return {
            "schedule": pd.DataFrame(records),
            "horizon_years": horizon_years,
            "pv_fcf_m": pv_sum,
            "tv_undiscounted_m": tv_undiscounted,
            "pv_tv_m": pv_tv,
            "enterprise_value_m": ev_m
        }, None

    # --- MODE A : SOCIÉTÉ CONSOLIDÉE (GRILLE COMPACTE SANS SCROLL HORIZONTAL) ---
    if dcf_mode == "Société Consolidée":
        st.subheader("Paramètres DCF Consolidé")
        st.caption("Laisse vides les périodes au-delà de ton horizon de visibilité : l'actualisation s'arrêtera automatiquement et la Valeur Terminale prendra le relais.")

        state_prefix = f"dcf_cons_{selected_ticker}"
        if f"{state_prefix}_rev" not in st.session_state:
            st.session_state[f"{state_prefix}_rev"] = round(base_rev / 1e6, 2)
            st.session_state[f"{state_prefix}_margin"] = float(default_fcf_margin)
            st.session_state[f"{state_prefix}_fcf"] = round(base_fcf / 1e6, 2)

        def on_rev_or_margin_change():
            r_val = st.session_state[f"{state_prefix}_rev"]
            m_val = st.session_state[f"{state_prefix}_margin"]
            st.session_state[f"{state_prefix}_fcf"] = round(r_val * (m_val / 100.0), 2)

        def on_fcf_manual_change():
            r_val = st.session_state[f"{state_prefix}_rev"]
            f_val = st.session_state[f"{state_prefix}_fcf"]
            if r_val > 0:
                st.session_state[f"{state_prefix}_margin"] = round((f_val / r_val) * 100.0, 2)

        # Ligne 1 : 5 paramètres de base & taux
        r1_c1, r1_c2, r1_c3, r1_c4, r1_c5 = st.columns(5)
        with r1_c1:
            st.number_input(
                "Revenue Base (M$)",
                key=f"{state_prefix}_rev",
                step=100.0,
                format="%.2f",
                on_change=on_rev_or_margin_change
            )
        with r1_c2:
            st.number_input(
                "FCF Margin (%)",
                key=f"{state_prefix}_margin",
                step=0.5,
                format="%.2f",
                on_change=on_rev_or_margin_change
            )
        with r1_c3:
            fcf_base_m = st.number_input(
                "FCF Base (M$)",
                key=f"{state_prefix}_fcf",
                step=50.0,
                format="%.2f",
                on_change=on_fcf_manual_change
            )
        with r1_c4:
            lt_growth_pct = st.number_input("LT Growth Rate (%)", value=3.0, step=0.25, format="%.2f")
        with r1_c5:
            wacc_pct = st.number_input("WACC (%)", value=8.58, step=0.10, format="%.2f")

        # Ligne 2 : Les 6 périodes de 5 ans (P3 à P6 vides par défaut => arrêt à Y10)
        r2_c1, r2_c2, r2_c3, r2_c4, r2_c5, r2_c6 = st.columns(6)
        with r2_c1:
            cagr_p1 = st.number_input("CAGR P1 (Y1-5 %)", value=7.27, step=0.25, format="%.2f")
        with r2_c2:
            cagr_p2 = st.number_input("CAGR P2 (Y6-10 %)", value=5.00, step=0.25, format="%.2f")
        with r2_c3:
            cagr_p3 = st.number_input("CAGR P3 (Y11-15 %)", value=None, placeholder="Vide = TV en Y10", step=0.25, format="%.2f")
        with r2_c4:
            cagr_p4 = st.number_input("CAGR P4 (Y16-20 %)", value=None, placeholder="Vide", step=0.25, format="%.2f")
        with r2_c5:
            cagr_p5 = st.number_input("CAGR P5 (Y21-25 %)", value=None, placeholder="Vide", step=0.25, format="%.2f")
        with r2_c6:
            cagr_p6 = st.number_input("CAGR P6 (Y26-30 %)", value=None, placeholder="Vide", step=0.25, format="%.2f")

        res, err = run_dcf_from_fcf(
            fcf_0_m=float(fcf_base_m),
            cagrs_pct=[cagr_p1, cagr_p2, cagr_p3, cagr_p4, cagr_p5, cagr_p6],
            lt_growth_pct=float(lt_growth_pct),
            wacc_pct=float(wacc_pct)
        )

        if err:
            st.error(err)
        else:
            horizon_y = res["horizon_years"]
            ev_m = res["enterprise_value_m"]
            pv_bb_m = discount_buybacks_m(remaining_buybacks_m, int(buyback_years), float(wacc_pct))
            eq_val_m = ev_m + pv_bb_m - net_debt_input_m
            implied_price = (eq_val_m * 1e6) / shares_input_units if shares_input_units > 0 else np.nan

            st.divider()
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric(f"PV des FCF ({horizon_y} ans)", f"{res['pv_fcf_m']:,.0f} M$")
            m2.metric(f"PV TV (relais Y{horizon_y})", f"{res['pv_tv_m']:,.0f} M$", f"{res['pv_tv_m']/ev_m*100:.1f}% EV" if ev_m != 0 else None)
            m3.metric("Enterprise Value (EV)", f"{ev_m:,.0f} M$")
            m4.metric("PV Share Buybacks", f"+{pv_bb_m:,.0f} M$")
            m5.metric("Equity Value", f"{eq_val_m:,.0f} M$")
            m6.metric("Valeur Intrinsèque / Action", f"{implied_price:,.2f} $" if pd.notna(implied_price) else "N/A")

            with st.expander(f"Détail des flux actualisés sur {horizon_y} ans (relais Valeur Terminale en Y{horizon_y})"):
                st.dataframe(res["schedule"].style.format({
                    "FCF (M$)": "{:,.1f}",
                    "PV FCF (M$)": "{:,.1f}"
                }), width="stretch")

    # --- MODE B : PAR DIVISIONS (SUM-OF-THE-PARTS COMPACT SANS SCROLL) ---
    else:
        st.subheader("Paramètres DCF par Division (Sum-of-the-Parts)")
        st.caption("Laisse les colonnes P3 % à P6 % vides pour déclencher la Valeur Terminale dès la fin de P2 (Année 10).")

        seg_rows = [
            r for r in df_10k.index
            if r.strip().startswith("Revenue - ") and pd.notna(df_10k.loc[r, latest_fy]) and df_10k.loc[r, latest_fy] > 0
        ]

        if not seg_rows:
            init_divs = [{
                "Division": "Core Business",
                "Rev (M$)": round(base_rev / 1e6, 1),
                "FCF %": default_fcf_margin,
                "FCF (M$)": round(base_fcf / 1e6, 1),
                "P1 %": 7.27,
                "P2 %": 5.00,
                "P3 %": np.nan,
                "P4 %": np.nan,
                "P5 %": np.nan,
                "P6 %": np.nan,
                "LT %": 3.00,
                "WACC %": 8.58,
            }]
        else:
            init_divs = []
            for r in seg_rows:
                seg_name = r.strip().replace("Revenue - ", "")
                seg_rev_m = round(float(df_10k.loc[r, latest_fy]) / 1e6, 1)
                init_divs.append({
                    "Division": seg_name,
                    "Rev (M$)": seg_rev_m,
                    "FCF %": default_fcf_margin,
                    "FCF (M$)": round(seg_rev_m * (default_fcf_margin / 100.0), 1),
                    "P1 %": 7.27,
                    "P2 %": 5.00,
                    "P3 %": np.nan,
                    "P4 %": np.nan,
                    "P5 %": np.nan,
                    "P6 %": np.nan,
                    "LT %": 3.00,
                    "WACC %": 8.58,
                })

        df_divs_input = pd.DataFrame(init_divs)

        compact_col_config = {
            "Division": st.column_config.TextColumn("Division", width="medium"),
            "Rev (M$)": st.column_config.NumberColumn("Rev (M$)", format="%.1f", width="small"),
            "FCF %": st.column_config.NumberColumn("FCF %", format="%.2f", width="small"),
            "FCF (M$)": st.column_config.NumberColumn("FCF (M$)", format="%.1f", width="small"),
            "P1 %": st.column_config.NumberColumn("P1 %", format="%.2f", width="small"),
            "P2 %": st.column_config.NumberColumn("P2 %", format="%.2f", width="small"),
            "P3 %": st.column_config.NumberColumn("P3 %", format="%.2f", width="small"),
            "P4 %": st.column_config.NumberColumn("P4 %", format="%.2f", width="small"),
            "P5 %": st.column_config.NumberColumn("P5 %", format="%.2f", width="small"),
            "P6 %": st.column_config.NumberColumn("P6 %", format="%.2f", width="small"),
            "LT %": st.column_config.NumberColumn("LT %", format="%.2f", width="small"),
            "WACC %": st.column_config.NumberColumn("WACC %", format="%.2f", width="small"),
        }

        edited_divs = st.data_editor(
            df_divs_input,
            num_rows="dynamic",
            hide_index=True,
            column_config=compact_col_config,
            width="stretch"
        )

        sotp_results = []
        total_ev_m = 0.0
        weighted_wacc_num = 0.0
        has_error = False

        for idx, drow in edited_divs.iterrows():
            div_name = str(drow["Division"])
            rev_val = float(drow["Rev (M$)"]) if pd.notna(drow["Rev (M$)"]) else 0.0
            margin_val = float(drow["FCF %"]) if pd.notna(drow["FCF %"]) else 0.0
            fcf_val = float(drow["FCF (M$)"]) if pd.notna(drow["FCF (M$)"]) else 0.0

            if idx < len(df_divs_input):
                orig = df_divs_input.iloc[idx]
                if fcf_val == orig["FCF (M$)"] and (rev_val != orig["Rev (M$)"] or margin_val != orig["FCF %"]):
                    effective_fcf_0_m = rev_val * (margin_val / 100.0)
                else:
                    effective_fcf_0_m = fcf_val
            else:
                effective_fcf_0_m = fcf_val if fcf_val != 0 else rev_val * (margin_val / 100.0)

            div_wacc = float(drow["WACC %"]) if pd.notna(drow["WACC %"]) else 9.0
            div_lt = float(drow["LT %"]) if pd.notna(drow["LT %"]) else 2.5

            res, err = run_dcf_from_fcf(
                fcf_0_m=effective_fcf_0_m,
                cagrs_pct=[
                    drow["P1 %"],
                    drow["P2 %"],
                    drow["P3 %"],
                    drow["P4 %"],
                    drow["P5 %"],
                    drow["P6 %"],
                ],
                lt_growth_pct=div_lt,
                wacc_pct=div_wacc
            )
            if err:
                st.error(f"[{div_name}] {err}")
                has_error = True
                break

            ev_div_m = res["enterprise_value_m"]
            total_ev_m += ev_div_m
            weighted_wacc_num += ev_div_m * div_wacc
            sotp_results.append({
                "Division": div_name,
                "Horizon": f"Y{res['horizon_years']}",
                "FCF Base (M$)": effective_fcf_0_m,
                "PV FCF (M$)": res["pv_fcf_m"],
                "PV TV (M$)": res["pv_tv_m"],
                "Enterprise Value (M$)": ev_div_m,
            })

        if not has_error and sotp_results:
            df_sotp = pd.DataFrame(sotp_results)
            df_sotp["Poids EV (%)"] = (df_sotp["Enterprise Value (M$)"] / total_ev_m) * 100 if total_ev_m != 0 else 0.0

            avg_wacc_sotp = (weighted_wacc_num / total_ev_m) if total_ev_m > 0 else 9.0
            pv_bb_sotp_m = discount_buybacks_m(remaining_buybacks_m, int(buyback_years), avg_wacc_sotp)
            eq_val_sotp_m = total_ev_m + pv_bb_sotp_m - net_debt_input_m
            implied_price_sotp = (eq_val_sotp_m * 1e6) / shares_input_units if shares_input_units > 0 else np.nan

            st.subheader("Synthèse Sum-of-the-Parts (SOTP)")
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Somme des EV Divisions", f"{total_ev_m:,.0f} M$")
            s2.metric("PV Share Buybacks", f"+{pv_bb_sotp_m:,.0f} M$")
            s3.metric("Dette Nette Consolidée", f"-{net_debt_input_m:,.0f} M$")
            s4.metric("Equity Value Totale", f"{eq_val_sotp_m:,.0f} M$")
            s5.metric("Valeur Intrinsèque / Action", f"{implied_price_sotp:,.2f} $" if pd.notna(implied_price_sotp) else "N/A")

            st.dataframe(df_sotp.style.format({
                "FCF Base (M$)": "{:,.1f}",
                "PV FCF (M$)": "{:,.1f}",
                "PV TV (M$)": "{:,.1f}",
                "Enterprise Value (M$)": "{:,.1f}",
                "Poids EV (%)": "{:.1f}%"
            }), width="stretch")