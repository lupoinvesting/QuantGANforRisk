from pandera.pandas import Column, DataFrameSchema, Check, Index
from pandas import DatetimeTZDtype, CategoricalDtype
from numpy import datetime64, int64


daily_schema = DataFrameSchema(
    {

        "open": Column(float, Check(lambda s: s > 0), nullable=True),
        "high": Column(float, Check(lambda s: s > 0), nullable=True),
        "low": Column(float, Check(lambda s: s >= 0), nullable=True),
        "close": Column(float, Check(lambda s: s > 0), nullable=True),
        "volume": Column(int, Check(lambda s: s >= 0), nullable=True)
        
    },
    index=Index(
        datetime64,
        name="date"
    ),
    strict="filter",
)
