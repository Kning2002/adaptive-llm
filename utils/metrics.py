"""One range-normalized metric implementation; no clipping or filtering."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(truth, prediction):
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    if truth.shape != prediction.shape or truth.size < 2:
        raise ValueError("Metrics require matching arrays with at least two values")
    truth, prediction = truth.reshape(-1), prediction.reshape(-1)
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite inputs")
    denominator = np.max(truth) - np.min(truth)
    if np.isclose(denominator, 0.0):
        raise ValueError("Range-normalized metrics are undefined for constant targets")
    mae = mean_absolute_error(truth, prediction)
    mse = mean_squared_error(truth, prediction)
    nmse = mse / denominator**2
    result = dict(
        MAE=float(mae),
        RMSE=float(np.sqrt(mse)),
        nMAE=float(mae / denominator),
        nRMSE=float(np.sqrt(nmse)),
        R2=float(r2_score(truth, prediction)),
        MSE=float(mse),
        NMSE=float(nmse),
    )
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Evaluation produced non-finite metrics")
    return result
