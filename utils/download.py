
import yfinance as yf
import pandas as pd
from .data_parser import parse_yahoo_data


def yahoo(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Download historical data from Yahoo Finance.

    Args:
        symbol (str): The stock symbol to download data for.
        start (str): The start date for the data in 'YYYY-MM-DD' format.
        end (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing the historical data.
    """
    data = pd.DataFrame()
    data = yf.download(symbol, start=start, end=end)
    return parse_yahoo_data(data)