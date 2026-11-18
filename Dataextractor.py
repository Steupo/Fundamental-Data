import io
import time
import requests
import duckdb
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

# ==============================================================================
# 1. CONFIGURATION & MAPPING EXHAUSTIF US-GAAP / DEI
# ==============================================================================
SEC_HEADERS = {
    "User-Agent": "Antoine Post antoine.post.research@gmail.com",
    "Accept-Encoding": "gzip, deflate"
}

DB_PATH = "fundamental_universe.duckdb"
UNIVERSE_CSV = Path("universe.csv")

# Nombre d'années de dépôts 10-K / 10-Q dont les instances XML segments sont parsées
XML_SEGMENT_LOOKBACK_YEARS = 12

GAAP_TAG_MAPPING = {
    # --- P&L ---
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet"
    ],
    "Cost of Good Sold": [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold"
    ],
    "Gross Profit": ["GrossProfit"],
    "SG&A": ["SellingGeneralAndAdministrativeExpense"],
    "Sales & Marketing": [
        "SellingAndMarketingExpense",
        "MarketingAndAdvertisingExpense"
    ],
    "General & Administrative": ["GeneralAndAdministrativeExpense"],
    "R&D": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"
    ],
    "In Process R&D": ["ResearchAndDevelopmentExpenseAcquiredInProcess"],
    "Fulfillment / Store operating": ["FulfillmentExpense"],
    "Restructuring & Impairment": [
        "RestructuringCharges",
        "RestructuringSettlementAndImpairmentProvisions",
        "GoodwillImpairmentLoss",
        "AssetImpairmentCharges"
    ],
    "EBIT": ["OperatingIncomeLoss"],
    "Depreciation & Amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation"
    ],
    "Cost of Interest": [
        "InterestExpense",
        "InterestExpenseDebt"
    ],
    "Pre-Tax Income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"
    ],
    "Tax": ["IncomeTaxExpenseBenefit"],
    "Net Income": [
        "NetIncomeLoss",
        "ProfitLoss"
    ],
    "EPS": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasic"
    ],
    "Total Share Outstanding": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "CommonStockSharesOutstanding"
    ],
    # --- CASH FLOW ---
    "Cash flow from operations": ["NetCashProvidedByUsedInOperatingActivities"],
    "CAPEX": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpendituresIncurredButNotYetPaid"
    ],
    "Stock-Based Compensation": ["ShareBasedCompensation"],
    "Cash Flow from Investing": ["NetCashProvidedByUsedInInvestingActivities"],
    "Cash flow from financing": ["NetCashProvidedByUsedInFinancingActivities"],
    "Dividends Paid": [
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividends"
    ],
    "Share Buybacks": ["PaymentsForRepurchaseOfCommonStock"],
    # --- BILAN & STRUCTURE FINANCIÈRE ---
    "Cash & Short-Term Investments": [
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
    ],
    "Accounts Receivable": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent"
    ],
    "Inventory": [
        "InventoryNet",
        "InventoryFinishedGoodsNetOfReserves"
    ],
    "Accounts Payable": ["AccountsPayableCurrent"],
    "Accrued Liabilities": [
        "AccruedLiabilitiesCurrent",
        "EmployeeRelatedLiabilitiesCurrent"
    ],
    "Short Term Debt": [
        "ShortTermBorrowings",
        "DebtCurrent",
        "LongTermDebtCurrent"
    ],
    "Long Term Debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt"
    ],
    "Total Assets": ["Assets"],
    "Shareholders Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
    ],
    "Goodwill": ["Goodwill"],
    "Intangible Assets": ["IntangibleAssetsNetExcludingGoodwill"]
}

DEI_TAG_MAPPING = {
    "Number of Employee": ["EntityNumberOfEmployees"]
}

SEGMENT_TAG_TO_METRIC = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue",
    "RevenueFromContractWithCustomerIncludingAssessedTax": "Revenue",
    "Revenues": "Revenue",
    "SalesRevenueNet": "Revenue",
    "GrossProfit": "Gross Profit",
    "DepreciationDepletionAndAmortization": "Depreciation & Amortization",
    "DepreciationAndAmortization": "Depreciation & Amortization",
    "OperatingIncomeLoss": "EBIT"
}

VALID_SEGMENT_AXES = {
    "StatementBusinessSegmentsAxis",
    "ProductOrServiceAxis"
}

EXCLUDED_MEMBER_KEYWORDS = (
    "elimination", "intersegment", "reconciling", "consolidation"
)


# ==============================================================================
# 2. UTILITAIRES DE CALENDRIER FISCAL (CORRECTION DÉCALAGE COMPARATIF SEC)
# ==============================================================================
def _normalize_period_end_date(end_date_str: str) -> pd.Timestamp:
    """Ramène les clôtures 52/53 semaines débordant début du mois suivant (ex: 1er oct -> sept)."""
    dt = pd.to_datetime(end_date_str)
    if dt.day <= 5:
        dt = dt - pd.Timedelta(days=7)
    return dt


def _detect_fy_end_month(us_gaap: dict) -> int:
    """Déduit le mois de clôture fiscale réel de la société (ex: 9=Sept, 6=Juin, 12=Déc)."""
    months = []
    probe_tags = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "NetIncomeLoss",
        "Assets"
    ]
    for tag in probe_tags:
        if tag not in us_gaap:
            continue
        for unit_entries in us_gaap[tag].get("units", {}).values():
            for entry in unit_entries:
                if entry.get("form") == "10-K" and entry.get("end"):
                    start = entry.get("start")
                    end = entry.get("end")
                    if start:
                        days = (pd.to_datetime(end) - pd.to_datetime(start)).days
                        if days < 330:
                            continue
                    norm_dt = _normalize_period_end_date(end)
                    months.append(norm_dt.month)
        if len(months) >= 3:
            break

    if not months:
        return 12
    return Counter(months).most_common(1)[0][0]


def infer_fiscal_period(end_date_str: str, fy_end_month: int) -> tuple[int, str]:
    """Calcule l'exercice fiscal (FY) et le trimestre (Q1..Q4) réels à partir de period_end."""
    dt = _normalize_period_end_date(end_date_str)
    m = dt.month
    y = dt.year

    if fy_end_month == 12:
        fy = y
        q_num = (m - 1) // 3 + 1
    else:
        fy = y + 1 if m > fy_end_month else y
        months_into_fy = (m - fy_end_month - 1) % 12 + 1
        q_num = (months_into_fy - 1) // 3 + 1

    return int(fy), f"Q{q_num}"


# ==============================================================================
# 3. SCHÉMA DUCKDB & VUES ANALYTIQUES (10-K & 10-Q)
# ==============================================================================
def init_database(con: duckdb.DuckDBPyConnection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS universe_metadata (
            ticker VARCHAR PRIMARY KEY,
            company_name VARCHAR,
            exchange VARCHAR,
            entity_type VARCHAR,
            is_secondary_class INTEGER DEFAULT 0,
            bloomberg_id VARCHAR,
            active INTEGER DEFAULT 1,
            cik VARCHAR,
            sec_status VARCHAR DEFAULT 'PENDING',
            last_checked_at TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_fundamentals (
            ticker VARCHAR,
            cik VARCHAR,
            form VARCHAR,
            fiscal_year INTEGER,
            fiscal_quarter VARCHAR,
            period_end VARCHAR,
            duration_days INTEGER,
            metric VARCHAR,
            value DOUBLE,
            unit VARCHAR,
            filed_date VARCHAR,
            accn VARCHAR,
            PRIMARY KEY (ticker, form, fiscal_year, fiscal_quarter, metric)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS segment_fundamentals (
            ticker VARCHAR,
            cik VARCHAR,
            form VARCHAR,
            fiscal_year INTEGER,
            fiscal_quarter VARCHAR,
            period_end VARCHAR,
            axis_name VARCHAR,
            segment_code VARCHAR,
            segment_label VARCHAR,
            metric VARCHAR,
            value DOUBLE,
            filed_date VARCHAR,
            accn VARCHAR,
            PRIMARY KEY (ticker, form, fiscal_year, fiscal_quarter, segment_code, metric)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS processed_xml_filings (
            accn VARCHAR PRIMARY KEY,
            ticker VARCHAR,
            form VARCHAR,
            filed_date VARCHAR,
            parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # --------------------------------------------------------------------------
    # VUE 1 : ANNUAL DATA 10-K (CONSOLIDÉ COMPLET)
    # --------------------------------------------------------------------------
    con.execute("""
        CREATE OR REPLACE VIEW vw_annual_framework AS
        WITH pivoted AS (
            SELECT
                ticker,
                fiscal_year,
                MAX(period_end) AS period_end,
                MAX(filed_date) AS last_filed_date,
                MAX(CASE WHEN metric = 'Revenue' THEN value END) AS rev,
                MAX(CASE WHEN metric = 'Cost of Good Sold' THEN value END) AS cogs,
                MAX(CASE WHEN metric = 'Gross Profit' THEN value END) AS gp_raw,
                MAX(CASE WHEN metric = 'SG&A' THEN value END) AS sga_raw,
                MAX(CASE WHEN metric = 'Sales & Marketing' THEN value END) AS sm,
                MAX(CASE WHEN metric = 'General & Administrative' THEN value END) AS ga,
                MAX(CASE WHEN metric = 'R&D' THEN value END) AS rd,
                MAX(CASE WHEN metric = 'In Process R&D' THEN value END) AS iprd,
                MAX(CASE WHEN metric = 'Fulfillment / Store operating' THEN value END) AS fulfillment,
                MAX(CASE WHEN metric = 'Restructuring & Impairment' THEN value END) AS restruct,
                MAX(CASE WHEN metric = 'EBIT' THEN value END) AS ebit,
                MAX(CASE WHEN metric = 'Depreciation & Amortization' THEN value END) AS da,
                MAX(CASE WHEN metric = 'Cost of Interest' THEN value END) AS int_exp,
                MAX(CASE WHEN metric = 'Pre-Tax Income' THEN value END) AS ebt,
                MAX(CASE WHEN metric = 'Tax' THEN value END) AS tax,
                MAX(CASE WHEN metric = 'Net Income' THEN value END) AS ni,
                MAX(CASE WHEN metric = 'EPS' THEN value END) AS eps,
                MAX(CASE WHEN metric = 'Total Share Outstanding' THEN value END) AS shares,
                MAX(CASE WHEN metric = 'Cash flow from operations' THEN value END) AS cfo,
                MAX(CASE WHEN metric = 'CAPEX' THEN ABS(value) END) AS capex,
                MAX(CASE WHEN metric = 'Stock-Based Compensation' THEN value END) AS sbc,
                MAX(CASE WHEN metric = 'Cash Flow from Investing' THEN value END) AS cfi,
                MAX(CASE WHEN metric = 'Cash flow from financing' THEN value END) AS cff,
                MAX(CASE WHEN metric = 'Dividends Paid' THEN ABS(value) END) AS divs,
                MAX(CASE WHEN metric = 'Share Buybacks' THEN ABS(value) END) AS buybacks,
                MAX(CASE WHEN metric = 'Cash & Short-Term Investments' THEN value END) AS cash,
                MAX(CASE WHEN metric = 'Accounts Receivable' THEN value END) AS ar,
                MAX(CASE WHEN metric = 'Inventory' THEN value END) AS inv,
                MAX(CASE WHEN metric = 'Accounts Payable' THEN value END) AS ap,
                MAX(CASE WHEN metric = 'Accrued Liabilities' THEN value END) AS acc_liab,
                MAX(CASE WHEN metric = 'Short Term Debt' THEN value END) AS st_debt,
                MAX(CASE WHEN metric = 'Long Term Debt' THEN value END) AS lt_debt,
                MAX(CASE WHEN metric = 'Total Assets' THEN value END) AS assets,
                MAX(CASE WHEN metric = 'Shareholders Equity' THEN value END) AS equity,
                MAX(CASE WHEN metric = 'Goodwill' THEN value END) AS goodwill,
                MAX(CASE WHEN metric = 'Intangible Assets' THEN value END) AS intangibles,
                MAX(CASE WHEN metric = 'Number of Employee' THEN value END) AS employees
            FROM raw_fundamentals
            WHERE form = '10-K'
            GROUP BY ticker, fiscal_year
        ),
        calc AS (
            SELECT
                *,
                COALESCE(gp_raw, rev - cogs) AS gp,
                COALESCE(sga_raw, COALESCE(sm, 0) + COALESCE(ga, 0)) AS sga,
                (COALESCE(gp_raw, rev - cogs) - ebit) AS total_opex,
                (ebit + COALESCE(da, 0)) AS ebitda,
                (cfo - COALESCE(capex, 0)) AS fcf,
                (COALESCE(ar, 0) + COALESCE(inv, 0)) - (COALESCE(ap, 0) + COALESCE(acc_liab, 0)) AS wc,
                (COALESCE(st_debt, 0) + COALESCE(lt_debt, 0)) AS total_debt,
                (COALESCE(st_debt, 0) + COALESCE(lt_debt, 0) - COALESCE(cash, 0)) AS net_debt,
                GREATEST(0.0, LEAST(0.50, COALESCE(tax / NULLIF(ebt, 0), 0.21))) AS eff_tax_bounded
            FROM pivoted
        ),
        with_lags AS (
            SELECT
                *,
                wc - LAG(wc) OVER (PARTITION BY ticker ORDER BY fiscal_year ASC) AS chg_wc,
                (total_debt + equity - COALESCE(cash, 0)) AS invested_capital
            FROM calc
        )
        SELECT
            ticker,
            fiscal_year,
            period_end,
            last_filed_date,
            rev AS "Revenue",
            cogs AS "Cost of Good Sold",
            gp AS "Gross Profit",
            gp / NULLIF(rev, 0) AS "Gross Margin",
            total_opex AS "Opex",
            sga AS "SG&A",
            sm AS "Sales & Marketing",
            ga AS "General & Administrative",
            rd AS "R&D",
            rd / NULLIF(rev, 0) AS "R&D / Sales",
            iprd AS "In Process R&D",
            fulfillment AS "Fulfillment / Store operating",
            restruct AS "Restructuring & Impairment",
            total_opex - (COALESCE(sga, 0) + COALESCE(rd, 0) + COALESCE(iprd, 0) + COALESCE(fulfillment, 0) + COALESCE(restruct, 0)) AS "Other Operating",
            ebitda AS "EBITDA",
            ebitda / NULLIF(rev, 0) AS "EBITDA Margin",
            da AS "Depreciation & Amortization",
            ebit AS "EBIT",
            ebit / NULLIF(rev, 0) AS "EBIT Margin",
            int_exp AS "Cost of Interest",
            ebt AS "Pre-Tax Income",
            tax AS "Tax",
            tax / NULLIF(ebt, 0) AS "Tax Rate",
            ni AS "Net Income",
            ni / NULLIF(rev, 0) AS "Net Margin",
            shares AS "Total Share Outstanding",
            eps AS "EPS",
            cfo AS "Cash flow from operations",
            capex AS "CAPEX",
            fcf AS "Free Cash Flow",
            fcf / NULLIF(shares, 0) AS "FCF / Share",
            sbc AS "Stock-Based Compensation",
            (fcf - COALESCE(sbc, 0)) AS "FCF ex-SBC",
            cfi AS "Cash Flow from Investing",
            cff AS "Cash flow from financing",
            divs AS "Dividends Paid",
            buybacks AS "Share Buybacks",
            cash AS "Cash & Short-Term Investments",
            wc AS "Working Capital",
            chg_wc AS "Change in Working Capital",
            st_debt AS "Short Term Debt",
            lt_debt AS "Long Term Debt",
            total_debt AS "Total Debt",
            net_debt AS "Net Debt",
            net_debt / NULLIF(ebitda, 0) AS "Net Debt / EBITDA",
            net_debt / NULLIF(fcf, 0) AS "Net Debt / Free cash Flow",
            total_debt / NULLIF(fcf, 0) AS "Total Debt / Free Cash Flow",
            assets AS "Total Assets",
            equity AS "Shareholders Equity",
            goodwill AS "Goodwill",
            intangibles AS "Intangible Assets",
            (equity - COALESCE(goodwill, 0) - COALESCE(intangibles, 0)) / NULLIF(shares, 0) AS "Tangible Book Value per Share",
            employees AS "Number of Employee",
            rev / NULLIF(employees, 0) AS "Revenue per employee",
            ni / NULLIF(equity, 0) AS "Return on Equity",
            ni / NULLIF(assets, 0) AS "Return on Assets",
            (ebit * (1.0 - eff_tax_bounded)) / NULLIF(invested_capital, 0) AS "Return on Invested Capital"
        FROM with_lags
    """)

    # --------------------------------------------------------------------------
    # VUE 2 : QUARTERLY DATA 10-Q + Q4 RECONSTRUIT + DÉ-CUMUL YTD + RATIOS TTM
    # --------------------------------------------------------------------------
    con.execute("""
        CREATE OR REPLACE VIEW vw_quarterly_framework AS
        WITH q_raw AS (
            SELECT
                ticker,
                fiscal_year,
                fiscal_quarter,
                CASE fiscal_quarter WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4 END AS q_num,
                MAX(period_end) AS period_end,
                MAX(filed_date) AS last_filed_date,
                MAX(CASE WHEN metric = 'Revenue' THEN value END) AS rev_raw,
                MAX(CASE WHEN metric = 'Cost of Good Sold' THEN value END) AS cogs_raw,
                MAX(CASE WHEN metric = 'Gross Profit' THEN value END) AS gp_raw,
                MAX(CASE WHEN metric = 'SG&A' THEN value END) AS sga_raw,
                MAX(CASE WHEN metric = 'Sales & Marketing' THEN value END) AS sm_raw,
                MAX(CASE WHEN metric = 'General & Administrative' THEN value END) AS ga_raw,
                MAX(CASE WHEN metric = 'R&D' THEN value END) AS rd_raw,
                MAX(CASE WHEN metric = 'In Process R&D' THEN value END) AS iprd_raw,
                MAX(CASE WHEN metric = 'Fulfillment / Store operating' THEN value END) AS ful_raw,
                MAX(CASE WHEN metric = 'Restructuring & Impairment' THEN value END) AS rest_raw,
                MAX(CASE WHEN metric = 'EBIT' THEN value END) AS ebit_raw,
                MAX(CASE WHEN metric = 'Cost of Interest' THEN value END) AS int_raw,
                MAX(CASE WHEN metric = 'Pre-Tax Income' THEN value END) AS ebt_raw,
                MAX(CASE WHEN metric = 'Tax' THEN value END) AS tax_raw,
                MAX(CASE WHEN metric = 'Net Income' THEN value END) AS ni_raw,
                MAX(CASE WHEN metric = 'EPS' THEN value END) AS eps_raw,
                MAX(CASE WHEN metric = 'Depreciation & Amortization' THEN value END) AS da_ytd,
                MAX(CASE WHEN metric = 'Cash flow from operations' THEN value END) AS cfo_ytd,
                MAX(CASE WHEN metric = 'CAPEX' THEN ABS(value) END) AS capex_ytd,
                MAX(CASE WHEN metric = 'Stock-Based Compensation' THEN value END) AS sbc_ytd,
                MAX(CASE WHEN metric = 'Cash Flow from Investing' THEN value END) AS cfi_ytd,
                MAX(CASE WHEN metric = 'Cash flow from financing' THEN value END) AS cff_ytd,
                MAX(CASE WHEN metric = 'Dividends Paid' THEN ABS(value) END) AS divs_ytd,
                MAX(CASE WHEN metric = 'Share Buybacks' THEN ABS(value) END) AS buybacks_ytd,
                MAX(CASE WHEN metric = 'Total Share Outstanding' THEN value END) AS shares,
                MAX(CASE WHEN metric = 'Cash & Short-Term Investments' THEN value END) AS cash,
                MAX(CASE WHEN metric = 'Accounts Receivable' THEN value END) AS ar,
                MAX(CASE WHEN metric = 'Inventory' THEN value END) AS inv,
                MAX(CASE WHEN metric = 'Accounts Payable' THEN value END) AS ap,
                MAX(CASE WHEN metric = 'Accrued Liabilities' THEN value END) AS acc_liab,
                MAX(CASE WHEN metric = 'Short Term Debt' THEN value END) AS st_debt,
                MAX(CASE WHEN metric = 'Long Term Debt' THEN value END) AS lt_debt,
                MAX(CASE WHEN metric = 'Total Assets' THEN value END) AS assets,
                MAX(CASE WHEN metric = 'Shareholders Equity' THEN value END) AS equity,
                MAX(CASE WHEN metric = 'Goodwill' THEN value END) AS goodwill,
                MAX(CASE WHEN metric = 'Intangible Assets' THEN value END) AS intangibles,
                MAX(CASE WHEN metric = 'Number of Employee' THEN value END) AS employees
            FROM raw_fundamentals
            GROUP BY ticker, fiscal_year, fiscal_quarter
        ),
        decumulated AS (
            SELECT
                *,
                -- Reconstruction Q4 uniquement si Q1, Q2 et Q3 sont tous trois présents
                CASE WHEN q_num < 4 THEN rev_raw
                     WHEN COUNT(CASE WHEN q_num < 4 THEN rev_raw END) OVER (PARTITION BY ticker, fiscal_year) = 3
                          THEN rev_raw - SUM(CASE WHEN q_num < 4 THEN rev_raw END) OVER (PARTITION BY ticker, fiscal_year)
                END AS rev,
                CASE WHEN q_num < 4 THEN cogs_raw
                     WHEN COUNT(CASE WHEN q_num < 4 THEN cogs_raw END) OVER (PARTITION BY ticker, fiscal_year) = 3
                          THEN cogs_raw - SUM(CASE WHEN q_num < 4 THEN cogs_raw END) OVER (PARTITION BY ticker, fiscal_year)
                END AS cogs,
                CASE WHEN q_num < 4 THEN gp_raw
                     WHEN COUNT(CASE WHEN q_num < 4 THEN gp_raw END) OVER (PARTITION BY ticker, fiscal_year) = 3
                          THEN gp_raw - SUM(CASE WHEN q_num < 4 THEN gp_raw END) OVER (PARTITION BY ticker, fiscal_year)
                END AS gp_direct,
                CASE WHEN q_num < 4 THEN sga_raw
                     WHEN COUNT(CASE WHEN q_num < 4 THEN sga_raw END) OVER (PARTITION BY ticker, fiscal_year) = 3
                          THEN sga_raw - SUM(CASE WHEN q_num < 4 THEN sga_raw END) OVER (PARTITION BY ticker, fiscal_year)
                END AS sga_direct,
                CASE WHEN q_num < 4 THEN sm_raw
                     WHEN COUNT(CASE WHEN q_num < 4 THEN sm_raw END) OVER (PARTITION BY ticker, fiscal_year) = 3
                          THEN sm_raw - SUM(CASE WHEN q_num < 4 THEN sm_raw END) OVER (PARTITION BY ticker, fiscal_year)
                END AS sm,
                CASE WHEN q_num < 4 THEN ga_raw
                     WHEN COUNT(CASE WHEN q_num < 4 THEN ga_raw END) OVER (PARTITION BY ticker, fiscal_year) = 3
                          THEN ga_raw - SUM(CASE WHEN q_num < 4 THEN ga_raw END) OVER (PARTITION BY ticker, fiscal_year)
                END AS ga,
                CASE WHEN q_num < 4 THEN rd_raw
                     WHEN COUNT(CASE WHEN q_num < 4 THEN rd_raw END) OVER (PARTITION BY ticker, fiscal_year) = 3
                          THEN rd_raw - SUM(CASE WHEN q_num < 4 THEN rd_raw END) OVER (PARTITION BY ticker, fiscal_year)
                END AS rd,
                CASE WHEN q_num < 4 THEN iprd_raw
                     WHEN COUNT(CASE WHEN q_num < 4 THEN iprd_raw END) OVER (PARTITION BY ticker, fiscal_year) = 3
                          THEN iprd_raw - SUM(CASE WHEN q_num < 4 THEN iprd_raw END) OVER (PARTITION BY ticker, fiscal_year)
                END AS iprd,
                CASE WHEN q_num < 4 THEN ful_raw
                     WHEN COUNT(CASE WHEN q_num < 4 THEN ful_raw END) OVER (PARTITION BY ticker, fiscal_year) = 3
                          THEN ful_raw - SUM(CASE WHEN q_num < 4 THEN ful_raw END) OVER (PARTITION BY ticker, fiscal_year)
                END AS fulfillment,
                CASE WHEN q_num < 4 THEN rest_raw
                     WHEN COUNT(CASE WHEN q_num < 4 THEN rest_raw END) OVER (PARTITION BY ticker, fiscal_year) = 3
                          THEN rest_raw - SUM(CASE WHEN q_num < 4 THEN rest_raw END) OVER (PARTITION BY ticker, fiscal_year)
                END AS restruct,
                CASE WHEN q_num < 4 THEN ebit_raw
                     WHEN COUNT(CASE WHEN q_num < 4 THEN ebit_raw END) OVER (PARTITION BY ticker, fiscal_year) = 3
                          THEN ebit_raw - SUM(CASE WHEN q_num < 4 THEN ebit_raw END) OVER (PARTITION BY ticker, fiscal_year)
                END AS ebit,
                CASE WHEN q_num < 4 THEN int_raw
                     WHEN COUNT(CASE WHEN q_num < 4 THEN int_raw END) OVER (PARTITION BY ticker, fiscal_year) = 3
                          THEN int_raw - SUM(CASE WHEN q_num < 4 THEN int_raw END) OVER (PARTITION BY ticker, fiscal_year)
                END AS int_exp,
                CASE WHEN q_num < 4 THEN ebt_raw
                     WHEN COUNT(CASE WHEN q_num < 4 THEN ebt_raw END) OVER (PARTITION BY ticker, fiscal_year) = 3
                          THEN ebt_raw - SUM(CASE WHEN q_num < 4 THEN ebt_raw END) OVER (PARTITION BY ticker, fiscal_year)
                END AS ebt,
                CASE WHEN q_num < 4 THEN tax_raw
                     WHEN COUNT(CASE WHEN q_num < 4 THEN tax_raw END) OVER (PARTITION BY ticker, fiscal_year) = 3
                          THEN tax_raw - SUM(CASE WHEN q_num < 4 THEN tax_raw END) OVER (PARTITION BY ticker, fiscal_year)
                END AS tax,
                CASE WHEN q_num < 4 THEN ni_raw
                     WHEN COUNT(CASE WHEN q_num < 4 THEN ni_raw END) OVER (PARTITION BY ticker, fiscal_year) = 3
                          THEN ni_raw - SUM(CASE WHEN q_num < 4 THEN ni_raw END) OVER (PARTITION BY ticker, fiscal_year)
                END AS ni,
                CASE WHEN q_num < 4 THEN eps_raw
                     WHEN COUNT(CASE WHEN q_num < 4 THEN eps_raw END) OVER (PARTITION BY ticker, fiscal_year) = 3
                          THEN eps_raw - SUM(CASE WHEN q_num < 4 THEN eps_raw END) OVER (PARTITION BY ticker, fiscal_year)
                END AS eps,
                -- Dé-cumul YTD Cash Flow (Q1 = YTD1, Qk = YTDk - YTD(k-1))
                CASE WHEN q_num = 1 THEN da_ytd
                     ELSE da_ytd - LAG(da_ytd) OVER (PARTITION BY ticker, fiscal_year ORDER BY q_num)
                END AS da,
                CASE WHEN q_num = 1 THEN cfo_ytd
                     ELSE cfo_ytd - LAG(cfo_ytd) OVER (PARTITION BY ticker, fiscal_year ORDER BY q_num)
                END AS cfo,
                CASE WHEN q_num = 1 THEN capex_ytd
                     ELSE capex_ytd - LAG(capex_ytd) OVER (PARTITION BY ticker, fiscal_year ORDER BY q_num)
                END AS capex,
                CASE WHEN q_num = 1 THEN sbc_ytd
                     ELSE sbc_ytd - LAG(sbc_ytd) OVER (PARTITION BY ticker, fiscal_year ORDER BY q_num)
                END AS sbc,
                CASE WHEN q_num = 1 THEN cfi_ytd
                     ELSE cfi_ytd - LAG(cfi_ytd) OVER (PARTITION BY ticker, fiscal_year ORDER BY q_num)
                END AS cfi,
                CASE WHEN q_num = 1 THEN cff_ytd
                     ELSE cff_ytd - LAG(cff_ytd) OVER (PARTITION BY ticker, fiscal_year ORDER BY q_num)
                END AS cff,
                CASE WHEN q_num = 1 THEN divs_ytd
                     ELSE divs_ytd - LAG(divs_ytd) OVER (PARTITION BY ticker, fiscal_year ORDER BY q_num)
                END AS divs,
                CASE WHEN q_num = 1 THEN buybacks_ytd
                     ELSE buybacks_ytd - LAG(buybacks_ytd) OVER (PARTITION BY ticker, fiscal_year ORDER BY q_num)
                END AS buybacks
            FROM q_raw
        ),
        calc AS (
            SELECT
                *,
                (fiscal_year * 10 + q_num) AS sort_key,
                (CAST(fiscal_year AS VARCHAR) || '-' || fiscal_quarter) AS period_label,
                COALESCE(gp_direct, rev - cogs) AS gp,
                COALESCE(sga_direct, COALESCE(sm, 0) + COALESCE(ga, 0)) AS sga,
                (COALESCE(gp_direct, rev - cogs) - ebit) AS total_opex,
                (ebit + COALESCE(da, 0)) AS ebitda,
                (cfo - COALESCE(capex, 0)) AS fcf,
                (COALESCE(ar, 0) + COALESCE(inv, 0)) - (COALESCE(ap, 0) + COALESCE(acc_liab, 0)) AS wc,
                (COALESCE(st_debt, 0) + COALESCE(lt_debt, 0)) AS total_debt,
                (COALESCE(st_debt, 0) + COALESCE(lt_debt, 0) - COALESCE(cash, 0)) AS net_debt,
                GREATEST(0.0, LEAST(0.50, COALESCE(tax / NULLIF(ebt, 0), 0.21))) AS eff_tax_bounded
            FROM decumulated
        ),
        ttm AS (
            SELECT
                *,
                wc - LAG(wc) OVER (PARTITION BY ticker ORDER BY sort_key) AS chg_wc,
                (total_debt + equity - COALESCE(cash, 0)) AS invested_capital,
                CASE WHEN COUNT(ebitda) OVER (PARTITION BY ticker ORDER BY sort_key ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) = 4
                     THEN SUM(ebitda) OVER (PARTITION BY ticker ORDER BY sort_key ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
                END AS ttm_ebitda,
                CASE WHEN COUNT(fcf) OVER (PARTITION BY ticker ORDER BY sort_key ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) = 4
                     THEN SUM(fcf) OVER (PARTITION BY ticker ORDER BY sort_key ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
                END AS ttm_fcf,
                CASE WHEN COUNT(ni) OVER (PARTITION BY ticker ORDER BY sort_key ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) = 4
                     THEN SUM(ni) OVER (PARTITION BY ticker ORDER BY sort_key ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
                END AS ttm_ni,
                CASE WHEN COUNT(ebit) OVER (PARTITION BY ticker ORDER BY sort_key ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) = 4
                     THEN SUM(ebit * (1.0 - eff_tax_bounded)) OVER (PARTITION BY ticker ORDER BY sort_key ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
                END AS ttm_nopat
            FROM calc
        )
        SELECT
            ticker,
            fiscal_year,
            fiscal_quarter,
            period_label,
            sort_key,
            period_end,
            last_filed_date,
            rev AS "Revenue",
            cogs AS "Cost of Good Sold",
            gp AS "Gross Profit",
            gp / NULLIF(rev, 0) AS "Gross Margin",
            total_opex AS "Opex",
            sga AS "SG&A",
            sm AS "Sales & Marketing",
            ga AS "General & Administrative",
            rd AS "R&D",
            rd / NULLIF(rev, 0) AS "R&D / Sales",
            iprd AS "In Process R&D",
            fulfillment AS "Fulfillment / Store operating",
            restruct AS "Restructuring & Impairment",
            total_opex - (COALESCE(sga, 0) + COALESCE(rd, 0) + COALESCE(iprd, 0) + COALESCE(fulfillment, 0) + COALESCE(restruct, 0)) AS "Other Operating",
            ebitda AS "EBITDA",
            ebitda / NULLIF(rev, 0) AS "EBITDA Margin",
            da AS "Depreciation & Amortization",
            ebit AS "EBIT",
            ebit / NULLIF(rev, 0) AS "EBIT Margin",
            int_exp AS "Cost of Interest",
            ebt AS "Pre-Tax Income",
            tax AS "Tax",
            tax / NULLIF(ebt, 0) AS "Tax Rate",
            ni AS "Net Income",
            ni / NULLIF(rev, 0) AS "Net Margin",
            shares AS "Total Share Outstanding",
            eps AS "EPS",
            cfo AS "Cash flow from operations",
            capex AS "CAPEX",
            fcf AS "Free Cash Flow",
            fcf / NULLIF(shares, 0) AS "FCF / Share",
            sbc AS "Stock-Based Compensation",
            (fcf - COALESCE(sbc, 0)) AS "FCF ex-SBC",
            cfi AS "Cash Flow from Investing",
            cff AS "Cash flow from financing",
            divs AS "Dividends Paid",
            buybacks AS "Share Buybacks",
            cash AS "Cash & Short-Term Investments",
            wc AS "Working Capital",
            chg_wc AS "Change in Working Capital",
            st_debt AS "Short Term Debt",
            lt_debt AS "Long Term Debt",
            total_debt AS "Total Debt",
            net_debt AS "Net Debt",
            net_debt / NULLIF(ttm_ebitda, 0) AS "Net Debt / EBITDA",
            net_debt / NULLIF(ttm_fcf, 0) AS "Net Debt / Free cash Flow",
            total_debt / NULLIF(ttm_fcf, 0) AS "Total Debt / Free Cash Flow",
            assets AS "Total Assets",
            equity AS "Shareholders Equity",
            goodwill AS "Goodwill",
            intangibles AS "Intangible Assets",
            (equity - COALESCE(goodwill, 0) - COALESCE(intangibles, 0)) / NULLIF(shares, 0) AS "Tangible Book Value per Share",
            employees AS "Number of Employee",
            (rev * 4.0) / NULLIF(employees, 0) AS "Revenue per employee",
            ttm_ni / NULLIF(equity, 0) AS "Return on Equity",
            ttm_ni / NULLIF(assets, 0) AS "Return on Assets",
            ttm_nopat / NULLIF(invested_capital, 0) AS "Return on Invested Capital"
        FROM ttm
    """)


def sync_universe_from_csv(con: duckdb.DuckDBPyConnection, csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier {csv_path} introuvable.")
    df_csv = pd.read_csv(csv_path, keep_default_na=False)
    df_csv["ticker"] = (
        df_csv["ticker"].astype(str).str.strip().str.upper()
        .str.replace("/", "-", regex=False)
        .str.replace(".", "-", regex=False)
    )
    df_csv = df_csv[df_csv["ticker"] != ""].copy()
    df_csv["is_secondary_class"] = pd.to_numeric(df_csv["is_secondary_class"], errors="coerce").fillna(0).astype(int)
    df_csv["active"] = pd.to_numeric(df_csv["active"], errors="coerce").fillna(1).astype(int)

    con.register("tmp_universe_csv", df_csv)
    con.execute("""
        INSERT INTO universe_metadata (
            ticker, company_name, exchange, entity_type, is_secondary_class, bloomberg_id, active
        )
        SELECT
            ticker, company_name, exchange, entity_type, is_secondary_class, bloomberg_id, active
        FROM tmp_universe_csv
        ON CONFLICT (ticker) DO UPDATE SET
            company_name = EXCLUDED.company_name,
            exchange = EXCLUDED.exchange,
            entity_type = EXCLUDED.entity_type,
            is_secondary_class = EXCLUDED.is_secondary_class,
            bloomberg_id = EXCLUDED.bloomberg_id,
            active = EXCLUDED.active
    """)
    con.unregister("tmp_universe_csv")


# ==============================================================================
# 4. EXTRACTION CONSOLIDÉE (COMPANYFACTS : 10-K + 10-Q + DEI)
# ==============================================================================
def get_sec_ticker_to_cik_map() -> dict[str, str]:
    url = "https://www.sec.gov/files/company_tickers.json"
    time.sleep(0.15)
    resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    return {
        item["ticker"].upper(): str(item["cik_str"]).zfill(10)
        for item in resp.json().values()
    }


def _parse_taxonomy_facts(
    ticker: str,
    cik: str,
    taxonomy_dict: dict,
    tag_mapping: dict,
    is_cash_flow_ytd: set[str],
    fy_end_month: int
) -> list[dict]:
    rows = []
    for metric_name, candidate_tags in tag_mapping.items():
        tag_records = []
        for priority_idx, tag in enumerate(candidate_tags):
            if tag not in taxonomy_dict:
                continue
            units_dict = taxonomy_dict[tag].get("units", {})
            unit_key = next(
                (u for u in ("USD/shares", "USD", "shares", "pure") if u in units_dict),
                None
            )
            if not unit_key:
                continue

            for entry in units_dict[unit_key]:
                form = entry.get("form")
                if form not in {"10-K", "10-Q"}:
                    continue
                fp = entry.get("fp")
                val = entry.get("val")
                end = entry.get("end")
                start = entry.get("start")
                filed = entry.get("filed")
                if val is None or not end:
                    continue

                days = (pd.to_datetime(end) - pd.to_datetime(start)).days if start else 0

                if form == "10-K":
                    if start and days < 330:
                        continue
                    true_fy, _ = infer_fiscal_period(end, fy_end_month)
                    fq = "Q4"
                else:
                    if fp not in {"Q1", "Q2", "Q3"}:
                        continue
                    true_fy, fq = infer_fiscal_period(end, fy_end_month)
                    if fq == "Q4":
                        continue
                    if start:
                        if metric_name in is_cash_flow_ytd:
                            expected_max = 110 if fq == "Q1" else (200 if fq == "Q2" else 290)
                            expected_min = 70 if fq == "Q1" else (160 if fq == "Q2" else 250)
                            if not (expected_min <= days <= expected_max):
                                continue
                        else:
                            if not (75 <= days <= 105):
                                continue

                tag_records.append({
                    "ticker": ticker,
                    "cik": cik,
                    "form": form,
                    "fiscal_year": int(true_fy),
                    "fiscal_quarter": fq,
                    "period_end": str(end),
                    "duration_days": int(days),
                    "metric": metric_name,
                    "value": float(val),
                    "unit": unit_key,
                    "filed_date": str(filed),
                    "accn": str(entry.get("accn", "")),
                    "priority": priority_idx
                })

        if tag_records:
            df_m = pd.DataFrame(tag_records)
            # Tri par priorité de tag, puis dépôt le plus récent (retraitements inclus) et clôture la plus récente
            df_m = df_m.sort_values(
                by=["form", "fiscal_year", "fiscal_quarter", "priority", "filed_date", "period_end"],
                ascending=[True, True, True, True, False, False]
            ).drop_duplicates(subset=["form", "fiscal_year", "fiscal_quarter"], keep="first")
            rows.append(df_m.drop(columns=["priority"]))

    return rows


def extract_consolidated_company_facts(ticker: str, cik: str) -> tuple[pd.DataFrame, int]:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    time.sleep(0.14)
    resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
    if resp.status_code == 404:
        return pd.DataFrame(), 12
    resp.raise_for_status()

    facts_root = resp.json().get("facts", {})
    us_gaap = facts_root.get("us-gaap", {})
    dei = facts_root.get("dei", {})
    if not us_gaap:
        return pd.DataFrame(), 12

    fy_end_month = _detect_fy_end_month(us_gaap)

    ytd_metrics = {
        "Depreciation & Amortization",
        "Cash flow from operations",
        "CAPEX",
        "Stock-Based Compensation",
        "Cash Flow from Investing",
        "Cash flow from financing",
        "Dividends Paid",
        "Share Buybacks"
    }

    rows = _parse_taxonomy_facts(ticker, cik, us_gaap, GAAP_TAG_MAPPING, ytd_metrics, fy_end_month)
    if dei:
        rows.extend(_parse_taxonomy_facts(ticker, cik, dei, DEI_TAG_MAPPING, set(), fy_end_month))

    df_out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return df_out, fy_end_month


# ==============================================================================
# 5. PARSEUR XML INSTANCE XBRL (DIVISIONS / SEGMENTS EN STREAMING RAM)
# ==============================================================================
def _clean_segment_label(member_qname: str) -> tuple[str, str]:
    raw_code = member_qname.split(":")[-1]
    label = raw_code
    for suffix in ("Member", "Segment", "Product", "Service"):
        if label.endswith(suffix) and len(label) > len(suffix):
            label = label[: -len(suffix)]
    readable = "".join(
        (" " + ch if (i > 0 and ch.isupper() and not label[i - 1].isupper()) else ch)
        for i, ch in enumerate(label)
    ).strip()
    return raw_code, readable


def _parse_single_xbrl_instance(
    xml_bytes: bytes,
    ticker: str,
    cik: str,
    form: str,
    filed_date: str,
    accn: str,
    fy_end_month: int
) -> list[dict]:
    contexts = {}
    raw_facts = []

    try:
        context_iter = ET.iterparse(io.BytesIO(xml_bytes), events=("end",))
        for _, elem in context_iter:
            local_name = elem.tag.split("}")[-1]

            if local_name == "context":
                ctx_id = elem.attrib.get("id")
                members = []
                start_date = None
                end_date = None

                for sub in elem.iter():
                    sub_name = sub.tag.split("}")[-1]
                    if sub_name == "explicitMember":
                        dim = sub.attrib.get("dimension", "").split(":")[-1]
                        val = (sub.text or "").strip()
                        members.append((dim, val))
                    elif sub_name == "startDate":
                        start_date = (sub.text or "").strip()
                    elif sub_name == "endDate":
                        end_date = (sub.text or "").strip()

                if len(members) == 1 and start_date and end_date:
                    dim_name, member_val = members[0]
                    if dim_name in VALID_SEGMENT_AXES and not any(
                        kw in member_val.lower() for kw in EXCLUDED_MEMBER_KEYWORDS
                    ):
                        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
                        contexts[ctx_id] = {
                            "axis": dim_name,
                            "member": member_val,
                            "start": start_date,
                            "end": end_date,
                            "days": days
                        }
                elem.clear()

            elif local_name in SEGMENT_TAG_TO_METRIC and "contextRef" in elem.attrib:
                text_val = (elem.text or "").strip()
                if text_val:
                    try:
                        num_val = float(text_val)
                        raw_facts.append({
                            "ctx_id": elem.attrib["contextRef"],
                            "metric": SEGMENT_TAG_TO_METRIC[local_name],
                            "value": num_val
                        })
                    except ValueError:
                        pass
                elem.clear()
    except ET.ParseError:
        return []

    if not contexts or not raw_facts:
        return []

    records = []
    for f in raw_facts:
        ctx = contexts.get(f["ctx_id"])
        if not ctx:
            continue
        days = ctx["days"]
        if form == "10-K" and days < 330:
            continue
        if form == "10-Q" and not (75 <= days <= 105):
            continue

        fy, fq = infer_fiscal_period(ctx["end"], fy_end_month)
        if form == "10-K":
            fq = "Q4"

        seg_code, seg_label = _clean_segment_label(ctx["member"])

        records.append({
            "ticker": ticker,
            "cik": cik,
            "form": form,
            "fiscal_year": fy,
            "fiscal_quarter": fq,
            "period_end": ctx["end"],
            "axis_name": ctx["axis"],
            "segment_code": seg_code,
            "segment_label": seg_label,
            "metric": f["metric"],
            "value": f["value"],
            "filed_date": filed_date,
            "accn": accn
        })

    return records


def extract_xml_segments_for_company(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    cik: str,
    fy_end_month: int,
    lookback_years: int = XML_SEGMENT_LOOKBACK_YEARS
):
    sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    time.sleep(0.14)
    resp = requests.get(sub_url, headers=SEC_HEADERS, timeout=30)
    if resp.status_code != 200:
        return

    recent = resp.json().get("filings", {}).get("recent", {})
    cik_int = str(int(cik))
    min_year = pd.Timestamp.now().year - lookback_years

    already_parsed = set(
        con.execute("SELECT accn FROM processed_xml_filings WHERE ticker = ?", [ticker]).df()["accn"].tolist()
    )

    all_segment_rows = []
    for form, accn, f_date, primary_doc in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("filingDate", []),
        recent.get("primaryDocument", [])
    ):
        if form not in {"10-K", "10-Q"}:
            continue
        if int(str(f_date)[:4]) < min_year:
            continue
        if accn in already_parsed:
            continue

        accn_clean = accn.replace("-", "")
        base_dir_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_clean}"

        stem = Path(primary_doc).stem
        xml_url = f"{base_dir_url}/{stem}_htm.xml"
        time.sleep(0.14)
        r_xml = requests.get(xml_url, headers=SEC_HEADERS, timeout=30)

        if r_xml.status_code != 200:
            sum_url = f"{base_dir_url}/FilingSummary.xml"
            time.sleep(0.14)
            r_sum = requests.get(sum_url, headers=SEC_HEADERS, timeout=20)
            if r_sum.status_code == 200:
                try:
                    sum_root = ET.fromstring(r_sum.content)
                    inst_name = None
                    for rep in sum_root.iter("Report"):
                        if rep.attrib.get("instance"):
                            inst_name = rep.attrib["instance"]
                            break
                    if inst_name:
                        time.sleep(0.14)
                        r_xml = requests.get(f"{base_dir_url}/{inst_name}", headers=SEC_HEADERS, timeout=30)
                except ET.ParseError:
                    pass

        if r_xml.status_code == 200:
            seg_rows = _parse_single_xbrl_instance(
                r_xml.content, ticker, cik, form, str(f_date), accn, fy_end_month
            )
            all_segment_rows.extend(seg_rows)

        con.execute(
            "INSERT OR IGNORE INTO processed_xml_filings (accn, ticker, form, filed_date) VALUES (?, ?, ?, ?)",
            [accn, ticker, form, str(f_date)]
        )

    if all_segment_rows:
        df_seg = pd.DataFrame(all_segment_rows)
        df_seg = df_seg.sort_values(
            by=["form", "fiscal_year", "fiscal_quarter", "segment_code", "metric", "filed_date", "period_end"],
            ascending=[True, True, True, True, True, False, False]
        ).drop_duplicates(subset=["form", "fiscal_year", "fiscal_quarter", "segment_code", "metric"], keep="first")

        con.register("tmp_seg_df", df_seg)
        con.execute("INSERT OR REPLACE INTO segment_fundamentals SELECT * FROM tmp_seg_df")
        con.unregister("tmp_seg_df")


# ==============================================================================
# 6. ORCHESTRATEUR PRINCIPAL INCRÉMENTAL
# ==============================================================================
def run_full_pipeline(
    target_tickers: list[str] | None = None,
    skip_existing_consolidated: bool = True,
    parse_xml_segments: bool = True
):
    con = duckdb.connect(DB_PATH)
    init_database(con)
    sync_universe_from_csv(con, UNIVERSE_CSV)

    existing_tickers = set(
        con.execute("SELECT DISTINCT ticker FROM raw_fundamentals").df()["ticker"].tolist()
    )
    existing_ciks = set(
        con.execute("SELECT DISTINCT cik FROM raw_fundamentals WHERE cik IS NOT NULL").df()["cik"].tolist()
    )

    if target_tickers:
        placeholders = ",".join(["?"] * len(target_tickers))
        candidates = con.execute(
            f"SELECT ticker, company_name, sec_status FROM universe_metadata WHERE ticker IN ({placeholders}) ORDER BY ticker",
            [t.upper() for t in target_tickers]
        ).fetchall()
    else:
        candidates = con.execute(
            "SELECT ticker, company_name, sec_status FROM universe_metadata WHERE active = 1 ORDER BY ticker"
        ).fetchall()

    print("Chargement du référentiel CIK SEC...")
    cik_map = get_sec_ticker_to_cik_map()
    total = len(candidates)

    for idx, (ticker, company_name, sec_status) in enumerate(candidates, start=1):
        cik = cik_map.get(ticker)
        if not cik:
            con.execute(
                "UPDATE universe_metadata SET sec_status = 'NO_CIK', last_checked_at = CURRENT_TIMESTAMP WHERE ticker = ?",
                [ticker]
            )
            continue

        if skip_existing_consolidated and sec_status in {"NO_10K_GAAP", "DUPLICATE_CIK"}:
            continue

        fy_end_month = 12

        if skip_existing_consolidated and ticker in existing_tickers:
            continue

        # 1. Extraction Consolidée (10-K + 10-Q + DEI)
        if not (skip_existing_consolidated and ticker in existing_tickers):
            if skip_existing_consolidated and cik in existing_ciks:
                con.execute(
                    "UPDATE universe_metadata SET cik = ?, sec_status = 'DUPLICATE_CIK', last_checked_at = CURRENT_TIMESTAMP WHERE ticker = ?",
                    [cik, ticker]
                )
                continue

            print(f"[{idx}/{total}] Consolidé 10-K/10-Q : {ticker} ({company_name})...")
            try:
                df_cons, fy_end_month = extract_consolidated_company_facts(ticker, cik)
                if df_cons.empty:
                    con.execute(
                        "UPDATE universe_metadata SET cik = ?, sec_status = 'NO_10K_GAAP', last_checked_at = CURRENT_TIMESTAMP WHERE ticker = ?",
                        [cik, ticker]
                    )
                    continue

                con.register("tmp_cons_df", df_cons)
                con.execute("INSERT OR REPLACE INTO raw_fundamentals SELECT * FROM tmp_cons_df")
                con.unregister("tmp_cons_df")
                con.execute(
                    "UPDATE universe_metadata SET cik = ?, sec_status = 'OK', last_checked_at = CURRENT_TIMESTAMP WHERE ticker = ?",
                    [cik, ticker]
                )
                existing_tickers.add(ticker)
                existing_ciks.add(cik)
            except Exception as e:
                print(f"   -> [ERREUR CONSOLIDÉ] {ticker} : {e}")
                continue
        else:
            # Déduit le mois de clôture depuis les 10-K déjà en base si le consolidé a été sauté
            res_m = con.execute(
                "SELECT period_end FROM raw_fundamentals WHERE ticker = ? AND form = '10-K' LIMIT 5",
                [ticker]
            ).fetchall()
            if res_m:
                fy_end_month = Counter(_normalize_period_end_date(r[0]).month for r in res_m).most_common(1)[0][0]

        # 2. Extraction Granulaire XML Segments / Divisions (incrémental par accn)
        if parse_xml_segments and ticker in existing_tickers:
            print(f"[{idx}/{total}] Parseur XML Segments : {ticker} (clôture mois {fy_end_month})...")
            try:
                extract_xml_segments_for_company(con, ticker, cik, fy_end_month)
            except Exception as e:
                print(f"   -> [ERREUR XML] {ticker} : {e}")

    con.close()
    print("\nPipeline terminé.")


if __name__ == "__main__":
    run_full_pipeline(
        target_tickers=None,
        skip_existing_consolidated=True,
        parse_xml_segments=True
    )