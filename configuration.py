"""One explicit configuration for the current Full model."""

import json
import math
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from pathlib import Path
from types import SimpleNamespace

import torch

from data_provider.dataset import read_csv


@dataclass
class Config:
    target: str = "Active_Power"
    feature_cols: list = field(
        default_factory=lambda: [
            "lmd_totalirrad",
            "lmd_diffuseirrad",
            "lmd_temperature",
            "lmd_pressure",
            "lmd_winddirection",
            "lmd_windspeed",
            "Active_Power",
        ]
    )
    seq_len: int = 96
    label_len: int = 96
    pred_len: int = 96
    train_ratio: float = 0.5
    val_ratio: float = 0.25
    test_ratio: float = 0.25
    seed: int = 4213
    batch_size: int = 16
    num_workers: int = 0
    train_epochs: int = 20
    patience: int = 3
    learning_rate: float = 1e-4
    huber_delta: float = 1.0
    prompt_diversity_weight: float = 0.001
    d_model: int = 32
    d_ff: int = 64
    n_heads: int = 8
    d_layers: int = 4
    dropout: float = 0.1
    top_k: int = 5
    num_kernels: int = 6
    llm_dim: int = 768
    llm_layers: int = 6
    period_lstm_dim: int = 256
    period_lstm_layers: int = 2
    period_lstm_dropout: float = 0.1
    period_lstm_alpha_init: float = 0.1
    adaptive_prompt_len: int = 4
    adaptive_prompt_hidden: int = 64
    adaptive_prompt_dropout: float = 0.1
    adaptive_prompt_alpha_init: float = 0.05

    @classmethod
    def load(cls, path=None):
        values = json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else {}
        config = cls(**values)
        config.validate()
        return config

    def validate(self):
        for key, value in asdict(self).items():
            if key not in ("target", "feature_cols") and (
                isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            ):
                raise ValueError(f"{key} must be a finite number")
        if (
            min(self.train_ratio, self.val_ratio, self.test_ratio) <= 0
            or abs(self.train_ratio + self.val_ratio + self.test_ratio - 1) > 1e-8
        ):
            raise ValueError("Positive train/val/test ratios must sum to one")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("target must be a nonempty string")
        if not isinstance(self.feature_cols, list) or not all(
            isinstance(name, str) and name and name not in ("date", "timestamp")
            for name in self.feature_cols
        ):
            raise ValueError("feature_cols must contain numeric field names, not timestamps")
        if self.target not in self.feature_cols or len(set(self.feature_cols)) != len(
            self.feature_cols
        ):
            raise ValueError("feature_cols must contain the target exactly once")
        if len(self.feature_cols) < 2:
            raise ValueError("The Full model needs historical auxiliary covariates")
        for key in (
            "seq_len",
            "pred_len",
            "batch_size",
            "train_epochs",
            "patience",
            "llm_layers",
            "llm_dim",
            "d_model",
            "d_ff",
            "n_heads",
            "d_layers",
            "top_k",
            "num_kernels",
            "period_lstm_dim",
            "period_lstm_layers",
            "adaptive_prompt_len",
            "adaptive_prompt_hidden",
        ):
            if not isinstance(getattr(self, key), Integral) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("seed", "num_workers", "label_len"):
            if not isinstance(getattr(self, key), Integral) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if not 0 <= self.label_len <= self.seq_len:
            raise ValueError("label_len must be between zero and seq_len")
        if self.learning_rate <= 0 or self.huber_delta <= 0 or self.prompt_diversity_weight < 0:
            raise ValueError("Invalid learning rate, Huber delta or prompt regularization")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_model % 2:
            raise ValueError("d_model must be even for sinusoidal positional embeddings")
        if self.adaptive_prompt_hidden % self.n_heads:
            raise ValueError("adaptive_prompt_hidden must be divisible by n_heads")
        if self.adaptive_prompt_len > 32:
            raise ValueError("adaptive_prompt_len must not exceed 32")
        for key in ("dropout", "period_lstm_dropout", "adaptive_prompt_dropout"):
            if not 0 <= getattr(self, key) < 1:
                raise ValueError(f"{key} must be in [0, 1)")
        for key in ("period_lstm_alpha_init", "adaptive_prompt_alpha_init"):
            if not 0 < getattr(self, key) < 1:
                raise ValueError(f"{key} must be in (0, 1)")

    def runtime(self, data_path, llm_path, output, device):
        self.validate()
        data_path = Path(data_path).resolve()
        llm_path = Path(llm_path).resolve()
        if not data_path.is_file() or not (llm_path / "config.json").is_file():
            raise FileNotFoundError("A prepared CSV and a local GPT-2 directory are required")
        columns = read_csv(data_path, nrows=1).columns
        if not ("date" in columns or "timestamp" in columns):
            raise ValueError("CSV must contain a date or timestamp column")
        if missing := sorted(set(self.feature_cols) - set(columns)):
            raise ValueError(f"Missing CSV fields: {missing}")
        values = asdict(self)
        values.update(
            device=torch.device(device),
            data_path=str(data_path),
            llm_model_path=str(llm_path),
            enc_in=len(self.feature_cols),
            c_out=1,
        )
        return SimpleNamespace(**values)
