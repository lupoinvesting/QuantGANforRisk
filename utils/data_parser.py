from utils.models import daily_schema

from pandera.pandas import check_output
from pandas import DataFrame, bdate_range
import numpy as np

@check_output(daily_schema)
def parse_yahoo_data(data: DataFrame) -> DataFrame:
    """
    Parse Yahoo Finance data to ensure it meets the schema requirements.

    Args:
        data (pd.DataFrame): The DataFrame to be validated.

    Returns:
        pd.DataFrame: The validated DataFrame.
    """
    data.columns = data.columns.droplevel(1)
    data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}, inplace=True)
    data.index.name = "date"
    return data


def return_to_df(returns_array, start_price = 100,start_date = "2000-01-01") -> DataFrame:
    """
    Create synthetic data for testing purposes.
    Returns:
        pd.DataFrame: A DataFrame with synthetic data.
    """
    prices = start_price * (1 + returns_array).cumprod()
    trading_days = len(returns_array)
    
    dates = bdate_range(start=start_date, periods=trading_days)
    df = DataFrame({"date":dates, "close": prices})
    df.set_index("date", inplace=True)
    return df

def returns_to_csv(returns_array: np.ndarray, symbol_name:str="DCSYN"):
    """save synthetic data to CSV files.
    Args:
        returns_array (np.ndarray): Array of returns.
        symbol_name (str): Symbol name for the CSV file.
    """
    for idx, data in enumerate(returns_array):
        filename_str = f"../data/{symbol_name}_{idx:04d}.csv"
        print(f"Saving {symbol_name}_{idx:04d}.csv")
        return_to_df(data).to_csv(f"data/{symbol_name}_{idx:04d}.csv")
        



    