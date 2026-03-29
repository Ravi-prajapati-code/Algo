"""
NSE watchlist — 60 liquid Nifty 500 stocks across diverse sectors.
Each entry: (yfinance_symbol, sector, company_name)
yfinance uses .NS suffix for NSE stocks.
"""

WATCHLIST = [
    # ── BANKING & FINANCIALS ───────────────────────────────────────────
    ("HDFCBANK.NS",    "Banking",       "HDFC Bank"),
    ("ICICIBANK.NS",   "Banking",       "ICICI Bank"),
    ("KOTAKBANK.NS",   "Banking",       "Kotak Mahindra Bank"),
    ("AXISBANK.NS",    "Banking",       "Axis Bank"),
    ("SBIN.NS",        "Banking",       "State Bank of India"),
    ("BAJFINANCE.NS",  "Finance",       "Bajaj Finance"),
    ("BAJAJFINSV.NS",  "Finance",       "Bajaj Finserv"),
    ("HDFCLIFE.NS",    "Insurance",     "HDFC Life Insurance"),
    ("SBILIFE.NS",     "Insurance",     "SBI Life Insurance"),

    # ── IT / TECHNOLOGY ────────────────────────────────────────────────
    ("TCS.NS",         "IT",            "Tata Consultancy Services"),
    ("INFY.NS",        "IT",            "Infosys"),
    ("WIPRO.NS",       "IT",            "Wipro"),
    ("HCLTECH.NS",     "IT",            "HCL Technologies"),
    ("TECHM.NS",       "IT",            "Tech Mahindra"),
    ("LTIM.NS",        "IT",            "LTIMindtree"),

    # ── CONSUMER STAPLES ───────────────────────────────────────────────
    ("HINDUNILVR.NS",  "FMCG",         "Hindustan Unilever"),
    ("ITC.NS",         "FMCG",         "ITC Ltd"),
    ("NESTLEIND.NS",   "FMCG",         "Nestle India"),
    ("BRITANNIA.NS",   "FMCG",         "Britannia Industries"),
    ("DABUR.NS",       "FMCG",         "Dabur India"),
    ("MARICO.NS",      "FMCG",         "Marico"),

    # ── PHARMACEUTICALS ────────────────────────────────────────────────
    ("SUNPHARMA.NS",   "Pharma",        "Sun Pharmaceutical"),
    ("DRREDDY.NS",     "Pharma",        "Dr Reddy's Laboratories"),
    ("CIPLA.NS",       "Pharma",        "Cipla"),
    ("DIVISLAB.NS",    "Pharma",        "Divi's Laboratories"),
    ("APOLLOHOSP.NS",  "Healthcare",    "Apollo Hospitals"),

    # ── AUTOMOBILES ────────────────────────────────────────────────────
    ("MARUTI.NS",      "Auto",          "Maruti Suzuki"),
    ("TATAMOTORS.NS",  "Auto",          "Tata Motors"),
    ("M&M.NS",         "Auto",          "Mahindra & Mahindra"),
    ("BAJAJ-AUTO.NS",  "Auto",          "Bajaj Auto"),
    ("EICHERMOT.NS",   "Auto",          "Eicher Motors"),
    ("HEROMOTOCO.NS",  "Auto",          "Hero MotoCorp"),

    # ── ENERGY / OIL & GAS ────────────────────────────────────────────
    ("RELIANCE.NS",    "Energy",        "Reliance Industries"),
    ("ONGC.NS",        "Energy",        "Oil & Natural Gas Corp"),
    ("NTPC.NS",        "Energy",        "NTPC Ltd"),
    ("POWERGRID.NS",   "Energy",        "Power Grid Corporation"),
    ("COALINDIA.NS",   "Energy",        "Coal India"),

    # ── INFRASTRUCTURE / INDUSTRIALS ──────────────────────────────────
    ("LT.NS",          "Infra",         "Larsen & Toubro"),
    ("ADANIPORTS.NS",  "Infra",         "Adani Ports"),
    ("ULTRACEMCO.NS",  "Infra",         "UltraTech Cement"),
    ("GRASIM.NS",      "Infra",         "Grasim Industries"),
    ("SHREECEM.NS",    "Infra",         "Shree Cement"),

    # ── METALS & MINING ────────────────────────────────────────────────
    ("TATASTEEL.NS",   "Metals",        "Tata Steel"),
    ("HINDALCO.NS",    "Metals",        "Hindalco Industries"),
    ("JSWSTEEL.NS",    "Metals",        "JSW Steel"),
    ("VEDL.NS",        "Metals",        "Vedanta Ltd"),

    # ── TELECOM ────────────────────────────────────────────────────────
    ("BHARTIARTL.NS",  "Telecom",       "Bharti Airtel"),

    # ── CONSUMER DISCRETIONARY ────────────────────────────────────────
    ("TITAN.NS",       "Consumer",      "Titan Company"),
    ("ASIANPAINT.NS",  "Consumer",      "Asian Paints"),
    ("PIDILITIND.NS",  "Consumer",      "Pidilite Industries"),
    ("HAVELLS.NS",     "Consumer",      "Havells India"),
    ("TRENT.NS",       "Consumer",      "Trent Ltd"),

    # ── CHEMICALS ──────────────────────────────────────────────────────
    ("SRF.NS",         "Chemicals",     "SRF Limited"),
    ("AARTIIND.NS",    "Chemicals",     "Aarti Industries"),

    # ── REAL ESTATE ────────────────────────────────────────────────────
    ("DLF.NS",         "RealEstate",    "DLF Ltd"),
    ("GODREJPROP.NS",  "RealEstate",    "Godrej Properties"),

    # ── AVIATION / HOSPITALITY ────────────────────────────────────────
    ("INDIGO.NS",      "Aviation",      "IndiGo (InterGlobe Aviation)"),

    # ── RETAIL / E-COMMERCE ───────────────────────────────────────────
    ("DMART.NS",       "Retail",        "Avenue Supermarts (DMart)"),
]

# Quick lookup maps
SYMBOL_TO_SECTOR = {sym: sector for sym, sector, _ in WATCHLIST}
SYMBOL_TO_NAME   = {sym: name   for sym, _, name  in WATCHLIST}
ALL_SYMBOLS      = [sym for sym, _, _ in WATCHLIST]
ALL_SECTORS      = sorted(set(sector for _, sector, _ in WATCHLIST))
