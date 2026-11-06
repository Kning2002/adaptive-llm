"""Forecast windows preserving the existing fill, split and scaling protocol."""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.time_features import time_features


def read_csv(path, **kwargs):
    """Only retry decoding errors, not arbitrary CSV or filesystem failures."""
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="SHIFT-JIS", **kwargs)


class PVForecastDataset(Dataset):
    def __init__(self, config, split="train"):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        self.seq_len = config.seq_len
        self.label_len = config.label_len
        self.pred_len = config.pred_len
        frame = read_csv(Path(config.data_path))
        if "date" not in frame.columns and "timestamp" in frame.columns:
            frame = frame.rename(columns={"timestamp": "date"})

        # Preserve the original global ffill -> bfill order before splitting.
        frame = frame.ffill().bfill()
        target = config.target
        columns = [name for name in config.feature_cols if name != target] + [target]
        frame = frame[["date"] + columns]
        ratios = (config.train_ratio, config.val_ratio, config.test_ratio)
        ratio_sum = sum(ratios)
        train_end = int(len(frame) * (ratios[0] / ratio_sum))
        val_end = train_end + int(len(frame) * (ratios[1] / ratio_sum))
        boundaries = {
            "train": (0, train_end),
            "val": (train_end - self.seq_len, val_end),
            "test": (val_end - self.seq_len, len(frame)),
        }
        start, end = boundaries[split]
        if start < 0 or end - start < self.seq_len + self.pred_len:
            raise ValueError(f"Not enough rows for {split} windows")
        self.scaler = StandardScaler().fit(frame[columns].iloc[:train_end].values)
        self.target_scaler = StandardScaler().fit(frame[[target]].iloc[:train_end].values)
        data = self.scaler.transform(frame[columns].values)
        self.data_x = data[start:end]
        self.data_y = data[start:end, -1:]
        timestamps = pd.to_datetime(frame["date"].iloc[start:end].values)
        self.data_stamp = time_features(timestamps, freq="t").transpose(1, 0)
        self.window_count = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __len__(self):
        return self.window_count

    def __getitem__(self, index):
        if not 0 <= index < self.window_count:
            raise IndexError(index)
        history_end = index + self.seq_len
        label_start = history_end - self.label_len
        label_end = history_end + self.pred_len
        return (
            self.data_x[index:history_end],
            self.data_y[label_start:label_end],
            self.data_stamp[index:history_end],
            self.data_stamp[label_start:label_end],
        )

    def inverse_transform(self, values):
        return self.target_scaler.inverse_transform(values)
