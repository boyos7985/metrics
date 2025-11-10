import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

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

def calculate_fcf_yield(ticker):
    t = yf.Ticker(ticker)
    cash_flow = t.cashflow
    market_cap = t.info.get('marketCap')
    free_cash_flow = safe_get(cash_flow, ["Free Cash Flow"], 0)
    if market_cap and market_cap > 0:
        return free_cash_flow / market_cap
    return 0

#########################################
# SHAREHOLDER YIELD
#########################################

def calculate_shareholder_yield(ticker, debug=False):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        dividend_yield = info.get('dividendYield', 0) or 0
        cash_flow = t.cashflow

        buybacks_method1 = abs(safe_get(cash_flow, [
            "Repurchase Of Capital Stock", 
            "Common Stock Payments",
            "Purchase of Stock"
        ], 0))

        net_issuance = safe_get(cash_flow, ["Net Common Stock Issuance"], 0)
        buybacks_method2 = abs(net_issuance) if net_issuance < 0 else 0

        buybacks = max(buybacks_method1, buybacks_method2)
        market_cap = info.get('marketCap')
        if not market_cap or market_cap <= 0:
            return dividend_yield

        buyback_yield = buybacks / market_cap
        shareholder_yield = (dividend_yield/100) + buyback_yield
        return shareholder_yield

    except:
        return 0

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

def calculate_deep_value_growth_metric(n_ticker, roic = 0, FCF_yield = 0, target_min=0.30):
    """
    Calcule le métrique: ROIC * (1 - Payout Ratio) + Earnings Yield
    Target: ≥ 30-40% pour du 'deep value with growth'
    """
    try:
        ticker = yf.Ticker(n_ticker)
        info = ticker.info
        # 1. Payout Ratio (Dividende / Bénéfice net)
        payout_ratio = info.get('payoutRatio')
        
        net_income = safe_get(ticker.financials, ["Net Income", "Net Income Applicable To Common Shares"], 0)
        
        # Get dividend information
        dividend_yield = info.get('dividendYield', 0) or 0
        
        # Get buybacks from cash flow statement - CORRECTION ICI
        cash_flow = ticker.cashflow
        
        # Méthode 1: Direct buyback items (POSITIFS dans le cash flow)
        buybacks_method1 = abs(safe_get(cash_flow, [
            "Repurchase Of Capital Stock", 
            "Common Stock Payments",
            "Purchase of Stock"
        ], 0))
        
        # Méthode 2: Net Common Stock Issuance NÉGATIF = buybacks
        net_issuance = safe_get(cash_flow, ["Net Common Stock Issuance"], 0)
        if net_issuance < 0:
            buybacks_method2 = abs(net_issuance)  # Negative = buybacks
        else:
            buybacks_method2 = 0
        
        # Prendre la valeur la plus cohérente
        buybacks = max(buybacks_method1, buybacks_method2)
        
        # Get market capitalization
        market_cap = info.get('marketCap')
        if market_cap is None or market_cap <= 0:
            return dividend_yield
        
        
        dividends_paid = (dividend_yield or 0) * market_cap
        print(f"dividends paid of {n_ticker} is {dividends_paid}")
        print(f"buybacks paid of {n_ticker} is {buybacks}")
        print(f"net income of {n_ticker} is {net_income}")
        
        payout_ratio = (dividends_paid + buybacks) / net_income
        
        print(f"payout ratio of {n_ticker} is {payout_ratio}")
        
        # Vérification des données manquantes
        if None in [roic, payout_ratio, FCF_yield]:
            missing = []
            if roic is None: missing.append('roic')
            if payout_ratio is None: missing.append('Payout Ratio')
            if FCF_yield is None: missing.append('FCF Yield')
            print(f"Données manquantes: {missing}")
            return None, None
        
        # Calcul du métrique
        expected_growth = roic * (1 - payout_ratio)
        deep_value_metric = expected_growth + FCF_yield
        
        # Vérification si le critère est satisfait
        meets_criteria = deep_value_metric >= target_min
        
        details = {
            'roic': roic,
            'payout_ratio': payout_ratio,
            'FCF_yield': FCF_yield,
            'expected_growth': expected_growth,
            'deep_value_metric': deep_value_metric,
            'target_min': target_min
        }
        
        return meets_criteria, deep_value_metric
        
    except Exception as e:
        print(f"Erreur dans le calcul: {e}")
        return None, None
#########################################
# ASSEMBLEUR FINAL POUR STREAMLIT
#########################################

def compute_all_metrics(ticker):
    import numpy as np
    import yfinance as yf

    t = yf.Ticker(ticker)
    info = t.info

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

    # FCF yield
    fcf_yield = calculate_fcf_yield(ticker)
    points_fcf = fcf_yield*300 if fcf_yield else None

    # shareholder yield
    shareholder_yield = calculate_shareholder_yield(ticker)
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
    # 1) calcul trend
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

    # deep value growth metric
    _, deep_value_growth = calculate_deep_value_growth_metric(ticker, roic=roc_value, FCF_yield=fcf_yield)
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

    # total points
    all_points = [
        points_pe, points_pb, points_cr, points_de,
        points_fcf, points_shr_y, points_eva, points_ev_eb,
        points_net_margin, points_asset_t, points_qual_margins, dvg_points, 
        points_rev, points_growth_assets, points_growth_sales,
        points_improve_debtEqRatio, points_growth_roic
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
        'Total_points' : total_points
    }

    # arrondir à 2 decimales si float
    for k,v in final.items():
        if isinstance(v,(int,float)) and v is not None:
            final[k] = round(v,2)

    return final
