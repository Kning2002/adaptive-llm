"""Five continuous calendar features in the original minute-level order."""

import numpy as np
import pandas as pd


def time_features(dates, freq="t"):
    if freq != "t":
        raise ValueError("The MVP supports minute-level calendar features only")
    index = pd.DatetimeIndex(dates)
    return np.vstack(
        [
            index.minute / 59.0 - 0.5,
            index.hour / 23.0 - 0.5,
            index.dayofweek / 6.0 - 0.5,
            (index.day - 1) / 30.0 - 0.5,
            (index.dayofyear - 1) / 365.0 - 0.5,
        ]
    )
