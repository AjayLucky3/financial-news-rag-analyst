"""
Security master universe for Indian listed equities (NSE/BSE).
Provides metadata, sector categorization, search terms, and alias resolution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import re


@dataclass
class Company:
    symbol: str
    name: str
    short_name: str
    exchange: str = "NSE"
    sector: str = "General"
    aliases: List[str] = field(default_factory=list)
    query_string: str = ""
    is_default: bool = False

    def __post_init__(self):
        if not self.query_string:
            # Generate default query string from name and short name
            if self.name != self.short_name:
                self.query_string = f'"{self.name}" OR "{self.short_name}"'
            else:
                self.query_string = f'"{self.name}"'
        if not self.aliases:
            self.aliases = [self.symbol, self.short_name, self.name]


# Comprehensive security master containing 50+ major Indian listed companies
SECURITY_MASTER: List[Company] = [
    # --- Information Technology ---
    Company(
        symbol="TCS.NS",
        name="Tata Consultancy Services",
        short_name="TCS",
        sector="Information Technology",
        aliases=["TCS", "Tata Consultancy Services", "Tata Consultancy", "TCS.NS"],
        query_string='"Tata Consultancy Services" OR "TCS"',
        is_default=True,
    ),
    Company(
        symbol="INFY.NS",
        name="Infosys",
        short_name="Infosys",
        sector="Information Technology",
        aliases=["Infosys", "INFY", "INFY.NS"],
        query_string='"Infosys" OR "INFY"',
        is_default=True,
    ),
    Company(
        symbol="WIPRO.NS",
        name="Wipro",
        short_name="Wipro",
        sector="Information Technology",
        aliases=["Wipro", "Wipro Limited", "WIPRO.NS"],
        query_string='"Wipro"',
        is_default=True,
    ),
    Company(
        symbol="HCLTECH.NS",
        name="HCL Technologies",
        short_name="HCL Tech",
        sector="Information Technology",
        aliases=["HCL Tech", "HCL Technologies", "HCLTECH", "HCLTECH.NS"],
        query_string='"HCL Technologies" OR "HCL Tech"',
    ),
    Company(
        symbol="TECHM.NS",
        name="Tech Mahindra",
        short_name="Tech Mahindra",
        sector="Information Technology",
        aliases=["Tech Mahindra", "TECHM", "TECHM.NS"],
        query_string='"Tech Mahindra"',
    ),
    Company(
        symbol="LTIM.NS",
        name="LTIMindtree",
        short_name="LTIMindtree",
        sector="Information Technology",
        aliases=["LTIMindtree", "LTI Mindtree", "LTIM", "LTIM.NS"],
        query_string='"LTIMindtree" OR "LTI Mindtree"',
    ),

    # --- Banking & Financial Services ---
    Company(
        symbol="HDFCBANK.NS",
        name="HDFC Bank",
        short_name="HDFC Bank",
        sector="Banking & Financial Services",
        aliases=["HDFC Bank", "HDFCBANK", "HDFCBANK.NS", "HDFC"],
        query_string='"HDFC Bank" OR "HDFCBANK"',
        is_default=True,
    ),
    Company(
        symbol="ICICIBANK.NS",
        name="ICICI Bank",
        short_name="ICICI Bank",
        sector="Banking & Financial Services",
        aliases=["ICICI Bank", "ICICIBANK", "ICICIBANK.NS", "ICICI"],
        query_string='"ICICI Bank" OR "ICICIBANK"',
    ),
    Company(
        symbol="SBIN.NS",
        name="State Bank of India",
        short_name="SBI",
        sector="Banking & Financial Services",
        aliases=["State Bank of India", "SBI", "SBIN", "SBIN.NS"],
        query_string='"State Bank of India" OR "SBI"',
    ),
    Company(
        symbol="KOTAKBANK.NS",
        name="Kotak Mahindra Bank",
        short_name="Kotak Bank",
        sector="Banking & Financial Services",
        aliases=["Kotak Mahindra Bank", "Kotak Bank", "KOTAKBANK", "KOTAKBANK.NS"],
        query_string='"Kotak Mahindra Bank" OR "Kotak Bank"',
    ),
    Company(
        symbol="AXISBANK.NS",
        name="Axis Bank",
        short_name="Axis Bank",
        sector="Banking & Financial Services",
        aliases=["Axis Bank", "AXISBANK", "AXISBANK.NS"],
        query_string='"Axis Bank"',
    ),
    Company(
        symbol="BAJFINANCE.NS",
        name="Bajaj Finance",
        short_name="Bajaj Finance",
        sector="Banking & Financial Services",
        aliases=["Bajaj Finance", "BAJFINANCE", "BAJFINANCE.NS"],
        query_string='"Bajaj Finance"',
    ),
    Company(
        symbol="BAJAJFINSV.NS",
        name="Bajaj Finserv",
        short_name="Bajaj Finserv",
        sector="Banking & Financial Services",
        aliases=["Bajaj Finserv", "BAJAJFINSV", "BAJAJFINSV.NS"],
        query_string='"Bajaj Finserv"',
    ),
    Company(
        symbol="SBILIFE.NS",
        name="SBI Life Insurance",
        short_name="SBI Life",
        sector="Banking & Financial Services",
        aliases=["SBI Life Insurance", "SBI Life", "SBILIFE", "SBILIFE.NS"],
        query_string='"SBI Life Insurance" OR "SBI Life"',
    ),
    Company(
        symbol="HDFCLIFE.NS",
        name="HDFC Life Insurance",
        short_name="HDFC Life",
        sector="Banking & Financial Services",
        aliases=["HDFC Life Insurance", "HDFC Life", "HDFCLIFE", "HDFCLIFE.NS"],
        query_string='"HDFC Life Insurance" OR "HDFC Life"',
    ),

    # --- Energy, Oil & Utilities ---
    Company(
        symbol="RELIANCE.NS",
        name="Reliance Industries",
        short_name="Reliance",
        sector="Energy & Oil",
        aliases=["Reliance Industries", "Reliance", "RIL", "RELIANCE.NS"],
        query_string='"Reliance Industries" OR "Reliance"',
        is_default=True,
    ),
    Company(
        symbol="ONGC.NS",
        name="Oil and Natural Gas Corporation",
        short_name="ONGC",
        sector="Energy & Oil",
        aliases=["Oil and Natural Gas Corporation", "ONGC", "ONGC.NS"],
        query_string='"Oil and Natural Gas Corporation" OR "ONGC"',
    ),
    Company(
        symbol="NTPC.NS",
        name="NTPC Limited",
        short_name="NTPC",
        sector="Utilities & Power",
        aliases=["NTPC Limited", "NTPC", "NTPC.NS"],
        query_string='"NTPC Limited" OR "NTPC"',
    ),
    Company(
        symbol="POWERGRID.NS",
        name="Power Grid Corporation of India",
        short_name="Power Grid",
        sector="Utilities & Power",
        aliases=["Power Grid Corporation", "Power Grid", "POWERGRID", "POWERGRID.NS"],
        query_string='"Power Grid Corporation" OR "Power Grid"',
    ),
    Company(
        symbol="BPCL.NS",
        name="Bharat Petroleum Corporation",
        short_name="BPCL",
        sector="Energy & Oil",
        aliases=["Bharat Petroleum", "BPCL", "BPCL.NS"],
        query_string='"Bharat Petroleum" OR "BPCL"',
    ),
    Company(
        symbol="IOC.NS",
        name="Indian Oil Corporation",
        short_name="Indian Oil",
        sector="Energy & Oil",
        aliases=["Indian Oil Corporation", "Indian Oil", "IOC", "IOC.NS"],
        query_string='"Indian Oil Corporation" OR "Indian Oil"',
    ),
    Company(
        symbol="COALINDIA.NS",
        name="Coal India",
        short_name="Coal India",
        sector="Metals & Mining",
        aliases=["Coal India", "COALINDIA", "COALINDIA.NS"],
        query_string='"Coal India"',
    ),

    # --- Fast-Moving Consumer Goods (FMCG) ---
    Company(
        symbol="ITC.NS",
        name="ITC Limited",
        short_name="ITC",
        sector="Consumer Goods / FMCG",
        aliases=["ITC Limited", "ITC", "ITC.NS"],
        query_string='"ITC Limited" OR "ITC"',
    ),
    Company(
        symbol="HINDUNILVR.NS",
        name="Hindustan Unilever",
        short_name="Hindustan Unilever",
        sector="Consumer Goods / FMCG",
        aliases=["Hindustan Unilever", "HUL", "HINDUNILVR", "HINDUNILVR.NS"],
        query_string='"Hindustan Unilever" OR "HUL"',
    ),
    Company(
        symbol="NESTLEIND.NS",
        name="Nestle India",
        short_name="Nestle India",
        sector="Consumer Goods / FMCG",
        aliases=["Nestle India", "NESTLEIND", "NESTLEIND.NS"],
        query_string='"Nestle India"',
    ),
    Company(
        symbol="BRITANNIA.NS",
        name="Britannia Industries",
        short_name="Britannia",
        sector="Consumer Goods / FMCG",
        aliases=["Britannia Industries", "Britannia", "BRITANNIA", "BRITANNIA.NS"],
        query_string='"Britannia Industries" OR "Britannia"',
    ),
    Company(
        symbol="TATACONSUM.NS",
        name="Tata Consumer Products",
        short_name="Tata Consumer",
        sector="Consumer Goods / FMCG",
        aliases=["Tata Consumer Products", "Tata Consumer", "TATACONSUM", "TATACONSUM.NS"],
        query_string='"Tata Consumer Products" OR "Tata Consumer"',
    ),

    # --- Automobile ---
    Company(
        symbol="TATAMOTORS.NS",
        name="Tata Motors",
        short_name="Tata Motors",
        sector="Automobile",
        aliases=["Tata Motors", "TATAMOTORS", "TATAMOTORS.NS"],
        query_string='"Tata Motors"',
    ),
    Company(
        symbol="MARUTI.NS",
        name="Maruti Suzuki India",
        short_name="Maruti Suzuki",
        sector="Automobile",
        aliases=["Maruti Suzuki", "Maruti", "MARUTI", "MARUTI.NS"],
        query_string='"Maruti Suzuki" OR "Maruti"',
    ),
    Company(
        symbol="M&M.NS",
        name="Mahindra & Mahindra",
        short_name="Mahindra",
        sector="Automobile",
        aliases=["Mahindra & Mahindra", "Mahindra and Mahindra", "M&M", "M&M.NS"],
        query_string='"Mahindra & Mahindra" OR "M&M"',
    ),
    Company(
        symbol="BAJAJ-AUTO.NS",
        name="Bajaj Auto",
        short_name="Bajaj Auto",
        sector="Automobile",
        aliases=["Bajaj Auto", "BAJAJ-AUTO", "BAJAJ-AUTO.NS"],
        query_string='"Bajaj Auto"',
    ),
    Company(
        symbol="EICHERMOT.NS",
        name="Eicher Motors",
        short_name="Eicher Motors",
        sector="Automobile",
        aliases=["Eicher Motors", "Royal Enfield", "EICHERMOT", "EICHERMOT.NS"],
        query_string='"Eicher Motors" OR "Royal Enfield"',
    ),
    Company(
        symbol="HEROMOTOCO.NS",
        name="Hero MotoCorp",
        short_name="Hero MotoCorp",
        sector="Automobile",
        aliases=["Hero MotoCorp", "Hero Honda", "HEROMOTOCO", "HEROMOTOCO.NS"],
        query_string='"Hero MotoCorp"',
    ),

    # --- Metals & Mining ---
    Company(
        symbol="TATASTEEL.NS",
        name="Tata Steel",
        short_name="Tata Steel",
        sector="Metals & Mining",
        aliases=["Tata Steel", "TATASTEEL", "TATASTEEL.NS"],
        query_string='"Tata Steel"',
    ),
    Company(
        symbol="JSWSTEEL.NS",
        name="JSW Steel",
        short_name="JSW Steel",
        sector="Metals & Mining",
        aliases=["JSW Steel", "JSWSTEEL", "JSWSTEEL.NS"],
        query_string='"JSW Steel"',
    ),
    Company(
        symbol="HINDALCO.NS",
        name="Hindalco Industries",
        short_name="Hindalco",
        sector="Metals & Mining",
        aliases=["Hindalco Industries", "Hindalco", "HINDALCO", "HINDALCO.NS"],
        query_string='"Hindalco Industries" OR "Hindalco"',
    ),
    Company(
        symbol="VEDL.NS",
        name="Vedanta Limited",
        short_name="Vedanta",
        sector="Metals & Mining",
        aliases=["Vedanta Limited", "Vedanta", "VEDL", "VEDL.NS"],
        query_string='"Vedanta Limited" OR "Vedanta"',
    ),

    # --- Pharmaceuticals & Healthcare ---
    Company(
        symbol="SUNPHARMA.NS",
        name="Sun Pharmaceutical Industries",
        short_name="Sun Pharma",
        sector="Pharmaceuticals & Healthcare",
        aliases=["Sun Pharmaceutical", "Sun Pharma", "SUNPHARMA", "SUNPHARMA.NS"],
        query_string='"Sun Pharmaceutical" OR "Sun Pharma"',
    ),
    Company(
        symbol="DRREDDY.NS",
        name="Dr. Reddy's Laboratories",
        short_name="Dr. Reddy's",
        sector="Pharmaceuticals & Healthcare",
        aliases=["Dr. Reddy's Laboratories", "Dr Reddys", "DRREDDY", "DRREDDY.NS"],
        query_string='"Dr. Reddy\'s" OR "Dr Reddys"',
    ),
    Company(
        symbol="CIPLA.NS",
        name="Cipla Limited",
        short_name="Cipla",
        sector="Pharmaceuticals & Healthcare",
        aliases=["Cipla", "Cipla Limited", "CIPLA", "CIPLA.NS"],
        query_string='"Cipla"',
    ),
    Company(
        symbol="DIVISLAB.NS",
        name="Divi's Laboratories",
        short_name="Divi's Lab",
        sector="Pharmaceuticals & Healthcare",
        aliases=["Divi's Laboratories", "Divis Laboratories", "DIVISLAB", "DIVISLAB.NS"],
        query_string='"Divi\'s Laboratories" OR "Divis"',
    ),
    Company(
        symbol="APOLLOHOSP.NS",
        name="Apollo Hospitals Enterprise",
        short_name="Apollo Hospitals",
        sector="Pharmaceuticals & Healthcare",
        aliases=["Apollo Hospitals", "APOLLOHOSP", "APOLLOHOSP.NS"],
        query_string='"Apollo Hospitals"',
    ),

    # --- Infrastructure, Telecom & Conglomerate ---
    Company(
        symbol="LT.NS",
        name="Larsen & Toubro",
        short_name="L&T",
        sector="Infrastructure & Construction",
        aliases=["Larsen & Toubro", "L&T", "LT", "LT.NS"],
        query_string='"Larsen & Toubro" OR "L&T"',
    ),
    Company(
        symbol="BHARTIARTL.NS",
        name="Bharti Airtel",
        short_name="Airtel",
        sector="Telecommunications",
        aliases=["Bharti Airtel", "Airtel", "BHARTIARTL", "BHARTIARTL.NS"],
        query_string='"Bharti Airtel" OR "Airtel"',
    ),
    Company(
        symbol="ADANIENT.NS",
        name="Adani Enterprises",
        short_name="Adani Enterprises",
        sector="Conglomerates & Infrastructure",
        aliases=["Adani Enterprises", "ADANIENT", "ADANIENT.NS"],
        query_string='"Adani Enterprises"',
    ),
    Company(
        symbol="ADANIPORTS.NS",
        name="Adani Ports and Special Economic Zone",
        short_name="Adani Ports",
        sector="Conglomerates & Infrastructure",
        aliases=["Adani Ports", "ADANIPORTS", "ADANIPORTS.NS"],
        query_string='"Adani Ports"',
    ),

    # --- Consumer Discretionary & Others ---
    Company(
        symbol="TITAN.NS",
        name="Titan Company",
        short_name="Titan",
        sector="Consumer Discretionary",
        aliases=["Titan Company", "Titan", "TITAN", "TITAN.NS"],
        query_string='"Titan Company" OR "Titan"',
    ),
    Company(
        symbol="ASIANPAINT.NS",
        name="Asian Paints",
        short_name="Asian Paints",
        sector="Consumer Goods / Paints",
        aliases=["Asian Paints", "ASIANPAINT", "ASIANPAINT.NS"],
        query_string='"Asian Paints"',
    ),
    Company(
        symbol="ULTRACEMCO.NS",
        name="UltraTech Cement",
        short_name="UltraTech Cement",
        sector="Cement & Building Materials",
        aliases=["UltraTech Cement", "ULTRACEMCO", "ULTRACEMCO.NS"],
        query_string='"UltraTech Cement"',
    ),
    Company(
        symbol="GRASIM.NS",
        name="Grasim Industries",
        short_name="Grasim",
        sector="Conglomerates & Materials",
        aliases=["Grasim Industries", "Grasim", "GRASIM", "GRASIM.NS"],
        query_string='"Grasim Industries" OR "Grasim"',
    ),
    Company(
        symbol="ZOMATO.NS",
        name="Zomato Limited",
        short_name="Zomato",
        sector="Technology / Consumer Services",
        aliases=["Zomato", "ZOMATO", "ZOMATO.NS"],
        query_string='"Zomato"',
    ),
    Company(
        symbol="INDIGO.NS",
        name="InterGlobe Aviation",
        short_name="IndiGo",
        sector="Aviation & Transport",
        aliases=["InterGlobe Aviation", "IndiGo", "INDIGO", "INDIGO.NS"],
        query_string='"InterGlobe Aviation" OR "IndiGo"',
    ),
]

# Fast lookup index by symbol
SYMBOL_INDEX: Dict[str, Company] = {c.symbol: c for c in SECURITY_MASTER}


def get_company_master() -> List[Company]:
    """Returns all companies in the security master universe."""
    return SECURITY_MASTER


def get_default_companies() -> List[Company]:
    """Returns the default watchlist companies."""
    return [c for c in SECURITY_MASTER if c.is_default]


def get_default_tickers() -> List[str]:
    """Returns list of ticker symbols for the default watchlist."""
    return [c.symbol for c in SECURITY_MASTER if c.is_default]


def get_all_sectors() -> List[str]:
    """Returns sorted list of distinct sectors in the security master."""
    return sorted(list({c.sector for c in SECURITY_MASTER}))


def get_company(symbol_or_alias: str) -> Optional[Company]:
    """
    Finds a company by exact symbol or alias match (case-insensitive).
    """
    if not symbol_or_alias:
        return None

    cleaned = symbol_or_alias.strip()
    upper = cleaned.upper()

    # Exact symbol match
    if upper in SYMBOL_INDEX:
        return SYMBOL_INDEX[upper]

    # Try appending .NS if missing
    if f"{upper}.NS" in SYMBOL_INDEX:
        return SYMBOL_INDEX[f"{upper}.NS"]

    # Match aliases
    for c in SECURITY_MASTER:
        for alias in c.aliases:
            if alias.upper() == upper:
                return c
        if c.name.upper() == upper or c.short_name.upper() == upper:
            return c

    return None


def resolve_ticker(text: str) -> Optional[str]:
    """
    Resolves text string to canonical NSE ticker symbol (e.g., 'SBI' -> 'SBIN.NS').
    """
    comp = get_company(text)
    return comp.symbol if comp else None


def search_companies(query: str, sector: Optional[str] = None) -> List[Company]:
    """
    Searches security master by name, symbol, short name, or aliases.
    Optionally filters by sector.
    """
    results: List[Company] = []
    q_norm = query.strip().upper() if query else ""

    for c in SECURITY_MASTER:
        if sector and sector != "All" and c.sector != sector:
            continue

        if not q_norm:
            results.append(c)
            continue

        # Check symbol, name, short name, and aliases
        matches = (
            q_norm in c.symbol.upper()
            or q_norm in c.name.upper()
            or q_norm in c.short_name.upper()
            or any(q_norm in alias.upper() for alias in c.aliases)
        )
        if matches:
            results.append(c)

    return results


def get_ticker_query_map() -> Dict[str, str]:
    """Returns mapping from ticker symbol to search query string."""
    return {c.symbol: c.query_string for c in SECURITY_MASTER}


def get_company_display_map() -> Dict[str, str]:
    """Returns mapping from ticker symbol to friendly display name."""
    return {c.symbol: c.short_name for c in SECURITY_MASTER}
