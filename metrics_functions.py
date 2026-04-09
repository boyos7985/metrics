import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

_fx_cache = {}

############################################
# Utilitaires de récupération de données
############################################

def safe_get(df, keys, period=0):
    for k in keys:
        if k in df.index:
            try:
                return df.loc[k].iloc[period]
            except:
                return 0
    return 0

def get_exchange_rate(from_currency, to_currency):
    """Récupère le taux de change entre deux devises avec cache"""
    if from_currency == to_currency:
        return 1.0
    pair = (from_currency, to_currency)
    if pair in _fx_cache:
        return _fx_cache[pair]
    try:
        ticker = yf.Ticker(f"{from_currency}{to_currency}=X")
        rate = ticker.info.get('regularMarketPrice')
        if rate and rate > 0:
            _fx_cache[pair] = rate
            _fx_cache[(to_currency, from_currency)] = 1.0 / rate
            return rate
    except:
        pass
    rates_to_usd = {
        'USD': 1.0, 'EUR': 1.08, 'GBP': 1.27, 'JPY': 0.0067, 'CNY': 0.138,
        'HKD': 0.128, 'AUD': 0.65, 'CAD': 0.74, 'CHF': 1.13, 'SEK': 0.095,
        'NOK': 0.093, 'DKK': 0.14, 'NZD': 0.60, 'SGD': 0.75, 'KRW': 0.00073,
        'TWD': 0.031, 'INR': 0.012, 'BRL': 0.17, 'ZAR': 0.055, 'MXN': 0.058,
        'PLN': 0.25, 'TRY': 0.029, 'IDR': 0.000062, 'MYR': 0.22, 'PHP': 0.017,
        'ILS': 0.28, 'CZK': 0.043, 'HUF': 0.0027, 'SAR': 0.267, 'AED': 0.272,
        'THB': 0.028, 'CLP': 0.0010, 'COP': 0.00024, 'PEN': 0.27,
        'QAR': 0.275, 'KWD': 3.26, 'BHD': 2.65, 'OMR': 2.60, 'EGP': 0.020,
        'VND': 0.000040, 'PKR': 0.0036, 'NGN': 0.00063, 'KES': 0.0077,
    }
    from_to_usd = rates_to_usd.get(from_currency)
    to_to_usd = rates_to_usd.get(to_currency)
    if from_to_usd and to_to_usd:
        rate = from_to_usd / to_to_usd
        _fx_cache[pair] = rate
        return rate
    return 1.0


def get_rate_free(fallback=0.04):
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="1d")
        if hist.empty:
            return fallback
        return hist["Close"].iloc[-1] / 100
    except:
        return fallback

#########################################
# BASIC RATIOS (info direct)
#########################################

def check_pe_ratio(info, seuil=20.0):
    pe = info.get('trailingPE')
    return True, pe

def check_pb_ratio(info, seuil=1.5):
    pb = info.get('priceToBook')
    return True, pb

def check_current_ratio(info, seuil=1.5):
    current_ratio = info.get('currentRatio')
    return True, current_ratio

#########################################
# DEBT / EQUITY
#########################################

def check_get_debt_to_equity(ticker,seuil = 0.5, period=0, debug=False):
    t = yf.Ticker(ticker)
    bs = t.balance_sheet

    total_debt = safe_get(bs, ["Total Debt"], period)
    total_equity = safe_get(bs, ["Stockholders Equity", "Common Stock Equity"], period)

    if total_debt == 0:
        long_term = safe_get(bs, ["Long Term Debt"], period)
        current_debt = safe_get(bs, ["Current Debt"], period)
        total_debt = long_term + current_debt

    if total_equity == 0:
        total_equity = safe_get(bs, ["Total Equity Gross Minority Interest"], period)

    if total_equity == 0:
        return True, None

    return True, total_debt / total_equity

#########################################
# INVESTED CAPITAL + ROIC
#########################################

def invested_capital_precise(balance_sheet, period=0, debug=False):
    total_assets = safe_get(balance_sheet, ["Total Assets"], period)
    total_current_assets = safe_get(balance_sheet, ["Current Assets"], period)
    total_current_liabilities = safe_get(balance_sheet, ["Current Liabilities"], period)
    accounts_payable = safe_get(balance_sheet, ["Payables And Accrued Expenses"], period)
    cash = safe_get(balance_sheet, ["Cash Cash Equivalents And Short Term Investments"], period)

    excess_cash = cash - max(0, total_current_liabilities - total_current_assets + cash)
    invested_capital = total_assets - accounts_payable - excess_cash
    return invested_capital

def check_roic_precise(ticker="AAPL", debug=False, seuil = 0.05):
    t = yf.Ticker(ticker)
    income_stmt = t.financials
    balance_sheet = t.balance_sheet

    operating_income = safe_get(income_stmt, ["Operating Income", "EBIT"], 0)
    tax_rate = safe_get(income_stmt, ["Tax Rate For Calcs"], 0)
    if tax_rate == 0:
        tax_rate = 0.21
    nopat = operating_income * (1 - tax_rate)

    ic0 = invested_capital_precise(balance_sheet, period=0)
    ic1 = invested_capital_precise(balance_sheet, period=1)
    avg_ic = (ic0 + ic1) / 2 if (ic0 and ic1) else None

    roc = nopat / avg_ic if avg_ic else None
    return True, roc, avg_ic

#########################################
# FCF YIELD
#########################################

def calculate_fcf_yield(ticker=None, info=None, cashflow=None, market_cap_local=None):
    if info is None or cashflow is None:
        t = yf.Ticker(ticker)
        if info is None: info = t.info
        if cashflow is None: cashflow = t.cashflow
    market_cap = market_cap_local if market_cap_local is not None else (info.get('marketCap') or 0)
    free_cash_flow = safe_get(cashflow, ["Free Cash Flow"], 0)
    if free_cash_flow and market_cap > 0:
        return free_cash_flow / market_cap
    return 0

#########################################
# SHAREHOLDER YIELD
#########################################

def calculate_shareholder_yield(ticker=None, debug=False, info=None, cashflow=None, market_cap_local=None):
    try:
        if info is None or cashflow is None:
            t = yf.Ticker(ticker)
            if info is None: info = t.info
            if cashflow is None: cashflow = t.cashflow

        dividend_yield = info.get('dividendYield', 0) or 0

        buybacks_method1 = abs(safe_get(cashflow, [
            "Repurchase Of Capital Stock",
            "Common Stock Payments",
            "Purchase of Stock"
        ], 0))

        net_issuance = safe_get(cashflow, ["Net Common Stock Issuance"], 0)
        buybacks_method2 = abs(net_issuance) if net_issuance < 0 else 0

        buybacks = max(buybacks_method1, buybacks_method2)
        market_cap = market_cap_local if market_cap_local is not None else (info.get('marketCap') or 0)
        if not market_cap or market_cap <= 0:
            return dividend_yield, 1

        buyback_yield = buybacks / market_cap
        shareholder_yield = (dividend_yield / 100) + buyback_yield

        net_income = info.get('netIncomeToCommon', 0) or 0
        dividends_paid = (dividend_yield / 100) * market_cap
        if net_income <= 0:
            total_payout_ratio = 1
        else:
            total_payout_ratio = (dividends_paid + buybacks) / net_income

        return shareholder_yield, total_payout_ratio

    except:
        return 0, 1

#########################################
# WACC
#########################################

def calculate_wacc(ticker: str, rf: float = 0.04, market_premium: float = 0.06) -> float:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        E = info.get("marketCap") or 0
        balance = stock.balance_sheet
        total_debt = safe_get(balance, ["Total Debt"], 0)
        cash = safe_get(balance, ["Cash Cash Equivalents And Short Term Investments"], 0)
        D = max(total_debt - cash, 0.0)
        V = E + D if (E + D) > 0 else 1.0

        beta = info.get("beta") or 1.0
        Re = rf + float(beta) * market_premium

        financials = stock.financials
        interest_expense = safe_get(financials, ["Interest Expense", "Interest Expense Non Operating"], 0)
        Rd = abs(float(interest_expense))/total_debt if total_debt>0 else 0.03

        tax_rate = safe_get(financials, ["Tax Rate For Calcs"],0) or 0.21
        WACC = (E / V) * Re + (D / V) * Rd * (1 - tax_rate)
        return float(WACC)

    except:
        return math.nan

def check_wacc(ticker="AAPL", debug=False, seuil=0.08, rf=0.04, market_premium=0.06):
    wacc_value = calculate_wacc(ticker, rf=rf, market_premium=market_premium)
    return True, wacc_value

#########################################
# OTHER METRICS
#########################################

def check_ev_ebitda(info, seuil=15):
    try:
        enterprise_value = info.get('enterpriseValue')
        ebitda = info.get('ebitda')
        ev_ebitda = enterprise_value / ebitda if enterprise_value and ebitda else None
        return True, ev_ebitda
    except:
        return True, None

def check_net_margin(info, seuil_min=0.1):
    try:
        net_income = info.get('netIncomeToCommon')
        total_revenue = info.get('totalRevenue')
        net_margin = net_income / total_revenue if net_income and total_revenue else None
        return True, net_margin
    except:
        return True, None

def get_beta(info):
    """Retourne le beta depuis yfinance info, fallback 1.0."""
    beta = info.get("beta") if info.get("beta") is not None else 1.0
    try:
        return float(beta)
    except Exception:
        return 1.0


def get_gross_margin(info):
    gm = info.get("grossMargins") or 0
    try:
        return float(gm)
    except:
        return 0

def check_asset_turnover_financials(ticker, seuil_min=0.3):
    try:
        Ticker = yf.Ticker(ticker)
        financials =Ticker.financials
        total_revenue = safe_get(financials, ["Total Revenue"], 0)
        bs = Ticker.balance_sheet
        total_assets = safe_get(bs, ["Total Assets"], 0)
        at = total_revenue / total_assets if total_assets else None
        return True, at
    except:
        return True, None
        
def check_last_three_revenues(revs, min_increase=0.05):
    """
    Vérifie que les 3 derniers résultats respectent une croissance minimale
    - L'avant-dernier > (1 + min_increase) * le résultat d'avant
    - Le dernier > (1 + min_increase) * l'avant-dernier
    """
    if len(revs) < 3:
        return False  # Pas assez de données
    
    # Prendre les 3 derniers résultats
    last_three = revs.iloc[-3:]
    
    # Vérifier les augmentations
    for i in range(len(last_three) - 1):
        if last_three.iloc[i + 1] <= (1 + min_increase) * last_three.iloc[i]:
            return False
    
    return True

def score_total_assets_growth(ticker, years=4):
    bs = yf.Ticker(ticker).balance_sheet
    assets = [safe_get(bs, ["Total Assets"], i) for i in range(years-1, -1, -1)]
    growth_years = sum(assets[i] <= assets[i+1] for i in range(len(assets)-1))
    score = growth_years / (len(assets)-1)
    return score, assets

# --- 2️⃣ Sales growth ---
def score_sales_growth(ticker, years=4):
    is_ = yf.Ticker(ticker).financials
    sales = [safe_get(is_, ["Total Revenue"], i) for i in range(years-1, -1, -1)]
    growth_years = sum(sales[i] <= sales[i+1] for i in range(len(sales)-1))
    score = growth_years / (len(sales)-1)
    return score, sales

# --- 3️⃣ Debt-to-Equity ratio ---
def score_debt_to_equity(ticker, years=4):
    bs = yf.Ticker(ticker).balance_sheet
    ratios = []
    for i in range(years-1, -1, -1):
        total_debt = safe_get(bs, ["Total Debt", "Short Long Term Debt Total", "Total Liabilities"], i)
        equity = safe_get(bs, ["Stockholder Equity", "Common Stock Equity"], i)
        ratios.append(total_debt / equity if equity != 0 else np.nan)

    decreasing_years = sum(ratios[i] >= ratios[i+1] for i in range(len(ratios)-1))
    score = decreasing_years / (len(ratios)-1)
    return score, ratios

def get_debt_to_equity(ticker, period=0, debug=False):
    """
    Calcule le Debt-to-Equity Ratio à partir du bilan.
    period=0 : dernier exercice
    """
    t = yf.Ticker(ticker)
    bs = t.balance_sheet

    total_debt = safe_get(bs, ["Total Debt"], period)
    total_equity = safe_get(bs, ["Stockholders Equity", "Common Stock Equity"], period)

    # Fallback si Total Debt est vide : somme long + court terme
    if total_debt == 0:
        long_term = safe_get(bs, ["Long Term Debt"], period)
        current_debt = safe_get(bs, ["Current Debt"], period)
        total_debt = long_term + current_debt

    # Fallback si equity est vide
    if total_equity == 0:
        total_equity = safe_get(bs, ["Total Equity Gross Minority Interest"], period)

    if total_equity == 0:
        return None  # impossible à calculer

    de_ratio = total_debt / total_equity

    if debug:
        print(f"--- Debt-to-Equity Debug ({ticker}) ---")
        print("Total Debt:", total_debt)
        print("Total Equity:", total_equity)
        print("Debt-to-Equity Ratio:", de_ratio)

    return de_ratio


def score_roic_stability(ticker, years=4, tolerance=0.03, debug=False):
    """
    Scoring de la stabilité ou amélioration du ROIC sur 'years' années.
    - Utilise EXACTEMENT la même logique que check_roic_precise (NOPAT / avg(IC sur 2 ans)).
    - Score = 1 si le ROIC est stable (+/- tolerance) ou en hausse.
    """
    t = yf.Ticker(ticker)
    is_ = t.financials
    bs = t.balance_sheet
    roics = []

    # Parcours des années de plus ancien -> plus récent
    for i in range(years - 1, -1, -1):
        operating_income = safe_get(is_, ["Operating Income", "EBIT"], i)
        pretax_income = safe_get(is_, ["Pretax Income", "Income Before Tax", "Earnings Before Tax"], i)
        income_tax = safe_get(is_, ["Income Tax Expense", "Provision for Income Taxes"], i)

        # Même logique que check_roic_precise pour le tax rate
        tax_rate = safe_get(is_, ["Tax Rate For Calcs"], i)
        if tax_rate == 0:
            tax_rate = 0.21

        nopat = operating_income * (1 - tax_rate)

        # 🧠 On calcule l'invested capital sur deux années (i et i+1) pour faire la moyenne
        ic_current = invested_capital_precise(bs, period=i, debug=debug)
        ic_prev = invested_capital_precise(bs, period=i + 1, debug=debug) if i + 1 < bs.shape[1] else ic_current
        avg_ic = (ic_current + ic_prev) / 2 if (ic_current and ic_prev) else np.nan

        roic = nopat / avg_ic if avg_ic else np.nan
        roics.append(roic)

        if debug:
            print(f"\n--- ROIC Debug (period {i}) ---")
            print("Operating Income:", operating_income)
            print("Pretax Income:", pretax_income)
            print("Income Tax:", income_tax)
            print("Tax Rate:", tax_rate)
            print("NOPAT:", nopat)
            print("IC current:", ic_current)
            print("IC previous:", ic_prev)
            print("Average IC:", avg_ic)
            print("ROIC:", roic)

    # Nettoyage des NaN
    roics = [r for r in roics if not np.isnan(r)]
    if len(roics) < 2:
        return 0, roics

    # 📊 Stabilité ou amélioration
    last = roics[-1]
    first = roics[0]
    stable = all(abs(roics[i] - roics[i - 1]) <= tolerance for i in range(1, len(roics)))
    improving = last >= first

    score = 1 if (stable or improving) else 0
    return score, roics
        
def get_annual_revenues(ticker):
    try:
        t = yf.Ticker(ticker)
        fin = t.financials
        if fin is None or fin.empty:
            return None

        rev_idx = None
        for possible in ['Total Revenue','TotalRevenue','Revenue','Revenues','totalRevenue']:
            if possible in fin.index:
                rev_idx = possible
                break
        if rev_idx is None:
            for idx in fin.index:
                if 'Revenue' in str(idx):
                    rev_idx = idx
                    break
        if rev_idx is None:
            return None

        rev_series = fin.loc[rev_idx].dropna().astype(float)
        if rev_series.empty:
            return None
        return rev_series.sort_index()
    except Exception:
        return None


def get_annual_ebitda(ticker=None, financials=None):
    """Retourne la série annuelle d'EBITDA depuis les financials yfinance."""
    try:
        if financials is None:
            financials = yf.Ticker(ticker).financials
        fin = financials
        if fin is None or fin.empty:
            return None
        for possible in ['EBITDA', 'Ebitda']:
            if possible in fin.index:
                series = fin.loc[possible].dropna().astype(float)
                if not series.empty:
                    return series.sort_index()
        ebit_idx = None
        for possible in ['EBIT', 'Operating Income', 'OperatingIncome']:
            if possible in fin.index:
                ebit_idx = possible
                break
        da_idx = None
        for possible in ['Reconciled Depreciation', 'Depreciation And Amortization', 'DepreciationAndAmortization']:
            if possible in fin.index:
                da_idx = possible
                break
        if ebit_idx is None or da_idx is None:
            return None
        ebit_series = fin.loc[ebit_idx].dropna().astype(float)
        da_series = fin.loc[da_idx].dropna().astype(float)
        common_idx = ebit_series.index.intersection(da_series.index)
        if len(common_idx) < 2:
            return None
        return (ebit_series.loc[common_idx] + da_series.loc[common_idx]).sort_index()
    except Exception:
        return None


def check_margin_gap_trend(ticker, years=3):
    """
    Analyse si l'écart entre marge brute et marge nette a augmenté consécutivement sur les X dernières années
    Retourne True si tendance à la hausse persistante + métriques détaillées
    """
    try:
        Ticker = yf.Ticker(ticker)
        # Obtenir les données historiques
        hist_financials = Ticker.financials
        
        if hist_financials.empty or len(hist_financials.columns) < years:
            return False, "Données insuffisantes"
        
        # Prendre les X dernières années
        recent_years = hist_financials.columns[:years]
        
        margin_gaps = []
        margin_data = []
        
        for year in recent_years:
            try:
                gross_profit = hist_financials[year].get('Gross Profit')
                net_income = hist_financials[year].get('Net Income')
                total_revenue = hist_financials[year].get('Total Revenue')
                
                if None in [gross_profit, net_income, total_revenue] or total_revenue == 0:
                    continue
                
                gross_margin = gross_profit / total_revenue
                net_margin = net_income / total_revenue
                margin_gap = gross_margin - net_margin
                
                margin_data.append({
                    'year': year.year,
                    'gross_margin': gross_margin,
                    'net_margin': net_margin,
                    'margin_gap': margin_gap,
                    'gap_percentage': margin_gap / gross_margin if gross_margin != 0 else 0
                })
                margin_gaps.append(margin_gap)
                
            except (TypeError, ZeroDivisionError):
                continue
        
        # Vérifier si on a assez de données
        if len(margin_gaps) < years:
            return False, f"Données incomplètes sur {years} ans"
        
        # Vérifier la tendance haussière consécutive
        is_increasing = all(margin_gaps[i] < margin_gaps[i+1] for i in range(len(margin_gaps)-1))
        
        # Calculer des métriques supplémentaires
        gap_growth = ((margin_gaps[-1] - margin_gaps[0]) / margin_gaps[0]) if margin_gaps[0] != 0 else 0
        avg_gap = sum(margin_gaps) / len(margin_gaps)
        
        metrics = {
            'trend_increasing': is_increasing,
            'gap_growth_percentage': gap_growth,
            'current_gap': margin_gaps[-1],
            'average_gap': avg_gap,
            'yearly_data': margin_data,
            'gap_increase_consistency': sum(1 for i in range(len(margin_gaps)-1) 
                                          if margin_gaps[i] < margin_gaps[i+1]) / (len(margin_gaps)-1)
        }
        
        return is_increasing, metrics
        
    except Exception as e:
        return False, f"Erreur d'analyse: {str(e)}"

#analyse quali marges
def evaluer_qualite_marges(metrics):
    """Évaluation qualitative des marges basée sur les metrics"""
    
    positives = []
    attentions = []
    
    # Analyse marge brute
    if metrics['current_gap'] < 0.25:  # Écart inférieur à 25%
        positives.append(f"Écart marge brute/nette raisonnable: {metrics['current_gap']}")
    else:
        attentions.append(f"Écart assez important entre marge brute et nette {metrics['current_gap']}")
    
    # Analyse stabilité
    if metrics['average_gap'] > 0.15:
        positives.append("Marges nettes robustes (>15%)")
    
    # Analyse tendance
    if metrics['gap_growth_percentage'] > -0.2:  # Baisse modérée
        positives.append(f"Stabilité relative de la structure de coûts, gap growth % {metrics['gap_growth_percentage']}")
    
    return positives, attentions

def calculate_deep_value_growth_metric(n_ticker=None, roic=0, FCF_yield=0, total_payout_ratio=1, target_min=0.30):
    """
    Calcule le métrique: ROIC * (1 - Payout Ratio) + FCF Yield
    Target: ≥ 30-40% pour du 'deep value with growth'
    """
    try:
        if None in [roic, total_payout_ratio, FCF_yield]:
            return None, None
        expected_growth = roic * (1 - total_payout_ratio)
        deep_value_metric = expected_growth + FCF_yield
        meets_criteria = deep_value_metric >= target_min
        return meets_criteria, deep_value_metric
    except Exception as e:
        return None, None
#########################################
# ASSEMBLEUR FINAL POUR STREAMLIT
#########################################

def compute_all_metrics(ticker):
    import numpy as np
    import yfinance as yf

    t = yf.Ticker(ticker)
    info = t.info
    t_cashflow = t.cashflow

    # ------------------- FX-adjusted market cap (même logique que script_base.py) -------------------
    trading_ccy = info.get('currency', 'USD')
    fin_ccy = info.get('financialCurrency', trading_ccy)
    fx_rate = get_exchange_rate(trading_ccy, fin_ccy) if trading_ccy != fin_ccy else 1.0
    raw_market_cap = info.get('marketCap', 0) or 0
    market_cap_local = raw_market_cap * fx_rate

    # ------------------- prix + ATH -------------------
    hist = t.history(period="max")
    last_price = hist["Close"].iloc[-1] if not hist.empty else None
    ath_value = hist["Close"].max() if not hist.empty else None
    discount_from_ath = (last_price/ath_value - 1)*100 if last_price and ath_value else None

    # 1) basic ratios
    _, pe_value = check_pe_ratio(info)
    points_pe = (15 - pe_value)*3 if pe_value not in (None,0) else None

    _, pb_value = check_pb_ratio(info)
    points_pb = (7 - pb_value)*5 if pb_value not in (None,0) else None

    _, cr_value = check_current_ratio(info)
    points_cr = cr_value*4 if cr_value else None

    _, de_value = check_get_debt_to_equity(ticker)
    points_de = (1 - de_value)*25 if de_value not in (None,0) else None

    # ROIC
    _, roc_value, avg_ic = check_roic_precise(ticker, debug=False)

    # FCF yield (avec market_cap_local en devise financière)
    fcf_yield = calculate_fcf_yield(ticker, info=info, cashflow=t_cashflow, market_cap_local=market_cap_local)
    points_fcf = fcf_yield*300 if fcf_yield else None

    # shareholder yield (retourne aussi total_payout_ratio)
    shareholder_yield, total_payout_ratio = calculate_shareholder_yield(ticker, info=info, cashflow=t_cashflow, market_cap_local=market_cap_local)
    points_shr_y = shareholder_yield*100 if shareholder_yield else None

    # WACC
    rf = get_rate_free()
    _, wacc_value = check_wacc(ticker, False, 0.08, rf, 0.06)

    # EcoValAdded
    Eco_Val_Added = (roc_value - wacc_value) if (roc_value not in (None,0) and wacc_value not in (None,0)) else None
    points_eva = Eco_Val_Added*200 if Eco_Val_Added else None

    # EV/EBITDA
    _, ev_ebit_res = check_ev_ebitda(info)
    points_ev_eb = (3 - ev_ebit_res)*9 if ev_ebit_res else None

    # net margin
    _, net_margin_res = check_net_margin(info)
    points_net_margin = net_margin_res*150 if net_margin_res else None

    # gross margin
    gross_margin = get_gross_margin(info)

    # turnover
    _, turn_res = check_asset_turnover_financials(ticker)
    points_asset_t = turn_res*15 if turn_res else None

    # qualitative margin
    qm_bool, qm_metrics = check_margin_gap_trend(ticker)
    if isinstance(qm_metrics, dict):
        positives, attentions = evaluer_qualite_marges(qm_metrics)
        points_qual_margins = len(positives)*20
        quality_margin_bool = qm_bool
        positive_qual_margins = len(positives)
    else:
        quality_margin_bool = None
        positive_qual_margins = None
        points_qual_margins = None

    # deep value growth metric (utilise total_payout_ratio issu de shareholder_yield)
    _, deep_value_growth = calculate_deep_value_growth_metric(ticker, roic=roc_value, FCF_yield=fcf_yield, total_payout_ratio=total_payout_ratio)
    dvg_points = deep_value_growth*200 if deep_value_growth else None

    # revenue last 3
    is_ = t.financials
    if not is_.empty and "Total Revenue" in is_.index:
        revs = is_.loc["Total Revenue"]
        rev_up = check_last_three_revenues(revs)
        points_rev = 35 if rev_up else 0
    else:
        points_rev = None

    # growth assets
    score_assets, _ = score_total_assets_growth(ticker)
    points_growth_assets = score_assets*35

    # growth sales
    score_sales, _ = score_sales_growth(ticker)
    points_growth_sales = score_sales*35

    # improve DE
    score_de, _ = score_debt_to_equity(ticker)
    points_improve_debtEqRatio = score_de*35

    # growth ROIC
    score_roic_s, _ = score_roic_stability(ticker)
    points_growth_roic = score_roic_s*35

    # ---- NOUVELLES MÉTRIQUES (même méthodologie que script_base.py) ----

    # beta
    beta = get_beta(info)
    points_beta = (1 - abs(beta)) * 25

    # ebitda series (réutilisée par plusieurs métriques)
    ebitda_series = get_annual_ebitda(ticker, financials=is_)
    revs_series = get_annual_revenues(ticker)

    # incr_ebitda_margin = delta_EBITDA / delta_Revenue (2 dernières années)
    incr_ebitda_margin = None
    points_incr_ebitda_margin = 0
    if ebitda_series is not None and len(ebitda_series) >= 2 and revs_series is not None and len(revs_series) >= 2:
        common_dates = sorted(ebitda_series.index.intersection(revs_series.index))
        if len(common_dates) >= 2:
            delta_ebitda = ebitda_series[common_dates[-1]] - ebitda_series[common_dates[-2]]
            delta_rev = revs_series[common_dates[-1]] - revs_series[common_dates[-2]]
            rev_prev = revs_series[common_dates[-2]]
            if delta_rev > 0 and rev_prev != 0 and (delta_rev / abs(rev_prev)) >= 0.02:
                incr_ebitda_margin = (delta_ebitda / delta_rev) * 100
                if incr_ebitda_margin > 90:
                    incr_ebitda_margin = None
                else:
                    points_incr_ebitda_margin = min(100, max(0, incr_ebitda_margin))

    # distance au 52w low
    distance_52w = None
    points_52w_low = 0
    low_52w = info.get('fiftyTwoWeekLow')
    if low_52w is not None and low_52w > 0 and last_price and last_price > 0:
        distance_52w = (last_price - low_52w) / low_52w * 100
        points_52w_low = max(0, 50 - distance_52w)

    # market cap en USD (market_cap_local déjà calculé en haut)
    market_cap_usd = None
    points_mktcap = 0
    if market_cap_local > 0:
        fx_to_usd = get_exchange_rate(fin_ccy, 'USD')
        market_cap_usd = market_cap_local * fx_to_usd
        if market_cap_usd < 50e6:
            points_mktcap = 50
        elif market_cap_usd < 300e6:
            points_mktcap = 40
        elif market_cap_usd < 2e9:
            points_mktcap = 25
        elif market_cap_usd < 10e9:
            points_mktcap = 10

    # ROA change (amélioration du ROA sur 2 ans)
    roa_change = None
    points_roa_change = 0
    try:
        bs = t.balance_sheet
        ni_idx = None
        for possible in ['Net Income', 'NetIncome', 'Net Income Common Stockholders']:
            if possible in is_.index:
                ni_idx = possible
                break
        ta_idx = None
        for possible in ['Total Assets', 'TotalAssets']:
            if possible in bs.index:
                ta_idx = possible
                break
        if ni_idx and ta_idx:
            ni_ser = is_.loc[ni_idx].dropna().astype(float)
            ta_ser = bs.loc[ta_idx].dropna().astype(float)
            common = sorted(ni_ser.index.intersection(ta_ser.index))
            if len(common) >= 2:
                roa_prev = ni_ser[common[-2]] / ta_ser[common[-2]] if ta_ser[common[-2]] != 0 else None
                roa_curr = ni_ser[common[-1]] / ta_ser[common[-1]] if ta_ser[common[-1]] != 0 else None
                if roa_prev is not None and roa_curr is not None:
                    roa_change = roa_curr - roa_prev
                    points_roa_change = roa_change * 200
    except Exception:
        pass

    # EBITDA growth vs assets growth
    ebitda_gt_assets = False
    ebitda_growth = None
    points_ebitda_gt_assets = 0
    try:
        bs = t.balance_sheet
        ta_idx = None
        for possible in ['Total Assets', 'TotalAssets']:
            if possible in bs.index:
                ta_idx = possible
                break
        if ebitda_series is not None and len(ebitda_series) >= 2 and ta_idx:
            ta_ser_e = bs.loc[ta_idx].dropna().astype(float).sort_index()
            common = sorted(ebitda_series.index.intersection(ta_ser_e.index))
            if len(common) >= 2:
                ebitda_g = (ebitda_series[common[-1]] - ebitda_series[common[-2]]) / abs(ebitda_series[common[-2]]) if ebitda_series[common[-2]] != 0 else None
                assets_g = (ta_ser_e[common[-1]] - ta_ser_e[common[-2]]) / abs(ta_ser_e[common[-2]]) if ta_ser_e[common[-2]] != 0 else None
                ebitda_growth = ebitda_g
                if ebitda_g is not None and assets_g is not None and ebitda_g > 0 and assets_g > 0 and ebitda_g > assets_g:
                    ebitda_gt_assets = True
                    points_ebitda_gt_assets = 30
    except Exception:
        pass

    # EBITDA margin (dernière année)
    ebitda_margin_val = None
    points_ebitda_margin = 0
    if ebitda_series is not None and revs_series is not None:
        common = sorted(ebitda_series.index.intersection(revs_series.index))
        if common and revs_series[common[-1]] != 0:
            ebitda_margin_val = ebitda_series[common[-1]] / revs_series[common[-1]]
            if ebitda_margin_val <= 0.95:
                points_ebitda_margin = max(0, ebitda_margin_val * 100)

    # forward PE
    forward_pe = info.get('forwardPE')
    points_forward_pe = 0
    if forward_pe is not None and 0 < forward_pe < 100:
        points_forward_pe = max(0, (20 - forward_pe) * 1.5)

    # PEG ratio (avec fallback CAGR EPS)
    peg = info.get('pegRatio')
    points_peg = 0
    if (peg is None or peg == 0) and pe_value not in (None, 0) and pe_value > 0:
        try:
            eps_idx = None
            for possible in ['Diluted EPS', 'Basic EPS', 'EPS']:
                if possible in is_.index:
                    eps_idx = possible
                    break
            if eps_idx is not None:
                eps_ser = is_.loc[eps_idx].dropna().astype(float).sort_index()
                if len(eps_ser) >= 2:
                    eps_first = eps_ser.iloc[0]
                    eps_last = eps_ser.iloc[-1]
                    n_years = len(eps_ser) - 1
                    if eps_first > 0 and eps_last > 0:
                        eps_cagr = (eps_last / eps_first) ** (1 / n_years) - 1
                        if eps_cagr > 0:
                            peg = pe_value / (eps_cagr * 100)
        except Exception:
            pass
    if peg is not None and peg > 0:
        points_peg = max(0, (1 - peg) * 20) if peg < 1 else 0

    # earnings surprise
    earnings_surprise = None
    points_earnings_surprise = 0
    try:
        earnings_dates = t.earnings_dates
        if earnings_dates is not None and not earnings_dates.empty:
            recent = earnings_dates.dropna(subset=['Surprise(%)'])
            if not recent.empty:
                earnings_surprise = recent.iloc[0]['Surprise(%)'] / 100
                if earnings_surprise > 0:
                    points_earnings_surprise = min(15, earnings_surprise * 30)
    except Exception:
        pass

    # total points
    all_points = [
        points_pe, points_pb, points_cr, points_de,
        points_fcf, points_shr_y, points_eva, points_ev_eb,
        points_net_margin, points_asset_t, points_qual_margins, dvg_points,
        points_rev, points_growth_assets, points_growth_sales,
        points_improve_debtEqRatio, points_growth_roic,
        points_beta, points_incr_ebitda_margin, points_52w_low, points_mktcap,
        points_roa_change, points_ebitda_gt_assets, points_ebitda_margin,
        points_forward_pe, points_peg, points_earnings_surprise
    ]
    total_points = sum([p for p in all_points if p not in (None, np.nan)])

    final = {
        'Ticker': ticker,
        'LastPrice': last_price,
        'ATH_Value': ath_value,
        'Discount_From_ATH': discount_from_ath,
        'PE_Ratio': pe_value,
        'points_pe' : points_pe,
        'PB_Ratio': pb_value,
        'points_pb' : points_pb,
        'Current_Ratio': cr_value,
        'points_cr' : points_cr,
        'Debt_Equity_Ratio': de_value,
        'points_de' : points_de,
        'Return_on_capital': roc_value,
        'FCF_Yield': fcf_yield,
        'points_fcf' : points_fcf,
        'Shareholder_Yield': shareholder_yield,
        'points_shr_y' : points_shr_y,
        'Weighted_avg_K' : wacc_value,
        'Eco_Val_Added' : Eco_Val_Added,
        'points_eva' : points_eva,
        'EV_to_EBITDA' : ev_ebit_res,
        'points_ev_eb' : points_ev_eb,
        'net_margin' : net_margin_res,
        'points_net_margin' : points_net_margin,
        'gross_margin' : gross_margin,
        'asset_turnover' : turn_res,
        'points_turnover' : points_asset_t,
        'quality_margin' : quality_margin_bool,
        'positive_qual_margins' : positive_qual_margins,
        'points_qual_margin' : points_qual_margins,
        'deep_value_growth': deep_value_growth,
        'dvg_points' : dvg_points,
        'incr_3revs_points' : points_rev,
        'points_growth_assets' : points_growth_assets,
        'points_growth_sales' : points_growth_sales,
        'points_improve_debtEqRatio' : points_improve_debtEqRatio,
        'points_growth_roic' : points_growth_roic,
        'beta' : beta,
        'points_beta' : points_beta,
        'incr_ebitda_margin' : incr_ebitda_margin,
        'points_incr_ebitda_margin' : points_incr_ebitda_margin,
        'distance_52w_low_pct' : distance_52w,
        'points_52w_low' : points_52w_low,
        'market_cap_usd' : market_cap_usd,
        'points_mktcap' : points_mktcap,
        'roa_change' : roa_change,
        'points_roa_change' : points_roa_change,
        'ebitda_growth' : ebitda_growth,
        'ebitda_gt_assets' : ebitda_gt_assets,
        'points_ebitda_gt_assets' : points_ebitda_gt_assets,
        'ebitda_margin' : ebitda_margin_val,
        'points_ebitda_margin' : points_ebitda_margin,
        'forward_pe' : forward_pe,
        'points_forward_pe' : points_forward_pe,
        'peg_ratio' : peg,
        'points_peg' : points_peg,
        'earnings_surprise' : earnings_surprise,
        'points_earnings_surprise' : points_earnings_surprise,
        'Total_points' : total_points
    }

    # arrondir à 2 decimales si float
    for k,v in final.items():
        if isinstance(v,(int,float)) and v is not None:
            final[k] = round(v,2)

    return final
