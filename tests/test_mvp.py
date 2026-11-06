"""CPU-only tests; all data and tiny GPT-2 weights live in temporary folders."""

import ast
import re
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import GPT2Config, GPT2Model

from configuration import Config
from data_provider.data_factory import data_provider
from models import Model
from run import forecast, seed_all, train
from run import test as evaluate
from utils.metrics import compute_metrics
from utils.numerical_safety import check_gradients, check_parameters, require_finite


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory(prefix="adaptivellm_mvp_") as folder:
        root = Path(folder)
        backbone = root / "tiny_gpt2"
        seed_all(4213)
        GPT2Model(
            GPT2Config(
                vocab_size=64,
                n_embd=32,
                n_layer=1,
                n_head=4,
                n_positions=128,
                n_ctx=128,
                n_inner=64,
                use_cache=False,
            )
        ).save_pretrained(backbone)
        config = Config(
            d_model=8,
            d_ff=16,
            n_heads=2,
            d_layers=1,
            dropout=0.0,
            top_k=2,
            num_kernels=2,
            llm_dim=32,
            llm_layers=1,
            period_lstm_dim=12,
            period_lstm_layers=2,
            period_lstm_dropout=0.0,
            adaptive_prompt_hidden=8,
            adaptive_prompt_dropout=0.0,
            batch_size=16,
            train_epochs=1,
        )
        t = np.arange(768)
        frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=len(t), freq="15min")})
        for index, column in enumerate(config.feature_cols):
            frame[column] = np.sin(t / (7 + index)) + index
        frame["Active_Power"] = np.maximum(10 * np.sin(2 * np.pi * t / 96), 0)
        data = root / "synthetic.csv"
        frame.to_csv(data, index=False)
        output = root / "outputs"
        output.mkdir()
        args = config.runtime(data, backbone, output, "cpu")
        yield root, config, args, output


class MVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_full_shapes_gradients_and_frozen_gpt2(self):
        with fixture() as (_, _, args, _):
            model = Model(args).float()
            _, loader = data_provider(args, "train")
            prediction, target = forecast(model, next(iter(loader)), args)
            self.assertEqual(tuple(prediction.shape), (16, 96, 1))
            loss = torch.nn.functional.huber_loss(
                prediction, target
            ) + 0.001 * model.prompt_diversity_loss(prediction)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            active = [p for p in model.parameters() if p.requires_grad]
            self.assertTrue(
                all(p.grad is not None and torch.isfinite(p.grad).all() for p in active)
            )
            backbone = model.LLM_encoder.llm_model
            self.assertTrue(
                all(not p.requires_grad and p.grad is None for p in backbone.parameters())
            )
            signal = torch.randn(2, 96, 1)
            periodic, residual = model.target_decomp(signal)
            self.assertEqual(periodic.shape, residual.shape)
            torch.testing.assert_close(periodic + residual, signal)

    def test_one_epoch_and_score_protocol(self):
        with fixture() as (_, _, args, output):
            seed_all(args.seed)
            model = Model(args).float()
            train(model, args, output)
            result = evaluate(model, args, output)
            self.assertTrue((output / "best.pt").is_file())
            self.assertEqual(result["shape"], [97, 96, 1])
            self.assertEqual(result["primary_windows"], 2)
            self.assertTrue(all(np.isfinite(v) for v in result["primary"].values()))
            labels = np.load(output / "truth.npy")
            self.assertEqual(float(np.ptp(labels[::96])), result["normalization_denominator"])

    def test_loader_same_windows_and_partition_history(self):
        with fixture() as (_, _, args, _):
            train_set, _ = data_provider(args, "train")
            val_set, _ = data_provider(args, "val")
            test_set, _ = data_provider(args, "test")
            self.assertEqual([len(train_set), len(val_set), len(test_set)], [193, 97, 97])
            self.assertEqual(train_set.data_x.shape[1], 7)
            # Validation begins with 96 already observed rows from training.
            np.testing.assert_allclose(val_set.data_x[:96], train_set.data_x[-96:])
            self.assertEqual(test_set[0][1].shape, (192, 1))

    def test_only_one_model(self):
        files = {p.name for p in (Path(__file__).resolve().parents[1] / "models").glob("*.py")}
        self.assertEqual(files, {"__init__.py", "adaptive_llm.py", "components.py"})
        self.assertEqual(Config.load().seq_len, 96)
        with self.assertRaises(TypeError):
            Config(model="TimesNet")

    def test_no_hidden_absolute_paths(self):
        root = Path(__file__).resolve().parents[1]
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertFalse(re.match(r"^[A-Za-z]:[/\\]", node.value), path)
                    prefixes = tuple(f"/{name}/" for name in ("home", "root", "Users"))
                    self.assertFalse(node.value.startswith(prefixes), path)

    def test_invalid_configurations_fail_explicitly(self):
        cases = [
            dict(n_heads=0),
            dict(n_heads=1.5),
            dict(dropout=1.0),
            dict(d_model=7, n_heads=1),
            dict(learning_rate=float("nan")),
            dict(train_ratio=float("inf")),
            dict(period_lstm_alpha_init=0),
            dict(num_workers=-1),
            dict(batch_size=True),
            dict(adaptive_prompt_hidden=7),
            dict(feature_cols=["date", "Active_Power"]),
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config(**values).validate()

    def test_metrics_and_constant_target_policy(self):
        truth = np.array([0.0, 2.0, 4.0])
        prediction = np.array([-1.0, 2.0, 5.0])
        result = compute_metrics(truth, prediction)
        self.assertAlmostEqual(result["MAE"], 2 / 3)
        self.assertAlmostEqual(result["RMSE"], np.sqrt(2 / 3))
        self.assertAlmostEqual(result["nMAE"], 1 / 6)
        self.assertAlmostEqual(result["NMSE"], (2 / 3) / 16)
        self.assertAlmostEqual(result["R2"], 0.75)
        for actual, predicted in [
            (np.ones(3), np.zeros(3)),
            (truth, prediction[:2]),
            (truth, np.array([0.0, np.nan, 1.0])),
        ]:
            with self.assertRaises(ValueError):
                compute_metrics(actual, predicted)

    def test_numerical_guards(self):
        with self.assertRaises(FloatingPointError):
            require_finite("test", torch.tensor(float("nan")))
        module = torch.nn.Linear(1, 1)
        module.weight.grad = torch.full_like(module.weight, float("inf"))
        with self.assertRaises(FloatingPointError):
            check_gradients(module, "test")
        with torch.no_grad():
            module.weight.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            check_parameters(module, "test")


if __name__ == "__main__":
    unittest.main()
