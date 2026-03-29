"""Load symbol universe and sector metadata from watchlist config."""

from typing import List, Dict
from config.watchlist_nse import WATCHLIST, SYMBOL_TO_SECTOR, SYMBOL_TO_NAME, ALL_SYMBOLS, ALL_SECTORS


def get_all_symbols() -> List[str]:
    return ALL_SYMBOLS


def get_sector(symbol: str) -> str:
    return SYMBOL_TO_SECTOR.get(symbol, "Unknown")


def get_name(symbol: str) -> str:
    return SYMBOL_TO_NAME.get(symbol, symbol)


def get_symbols_by_sector(sector: str) -> List[str]:
    return [sym for sym, sec, _ in WATCHLIST if sec == sector]


def get_all_sectors() -> List[str]:
    return ALL_SECTORS


def get_sector_map() -> Dict[str, str]:
    return SYMBOL_TO_SECTOR.copy()
