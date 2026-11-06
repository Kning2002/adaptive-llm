"""Optional same-weight regression against the unmodified first MVP.

Set ADAPTIVELLM_LEGACY_ROOT to that package directory to enable this test.
No reference data, checkpoints or machine-specific paths are shipped.
"""

import json
import os
import subprocess
import sys
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_provider.data_factory import data_provider
from models import Model
from run import forecast
from tests.test_mvp import fixture
from utils.metrics import compute_metrics

REFERENCE_RUNNER = r"""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from configuration import Config
from data_provider.data_factory import data_provider
from models import Model
from run import forecast, seed_all, metric_set

torch.set_num_threads(1)
root = Path(sys.argv[1])
config = Config.load(root / 'reference_config.json')
args = config.runtime(root / 'synthetic.csv', root / 'tiny_gpt2', root / 'outputs', 'cpu')
seed_all(args.seed)
model = Model(args).float().eval()
dataset, _ = data_provider(args, 'train')
batch = [torch.as_tensor(np.stack([dataset[i][field] for i in range(16)])) for field in range(5)]
prediction, target = forecast(model, batch, args)
regularizer = model.prompt_diversity_loss(prediction)
loss = torch.nn.functional.huber_loss(prediction, target, delta=args.huber_delta)
loss = loss + args.prompt_diversity_weight * regularizer
loss.backward()
payload = dict(state=model.state_dict(), output=prediction.detach(),
    regularizer=regularizer.detach(),
    gradients={name: p.grad.detach() for name, p in model.named_parameters() if p.grad is not None},
    datasets={})
for index, (ratios, label_len) in enumerate([
        ((.5, .25, .25), 96), ((.6, .2, .2), 48), ((.5, .25, .25), 0)]):
    config.train_ratio, config.val_ratio, config.test_ratio = ratios
    config.label_len = label_len
    case_args = config.runtime(root / 'synthetic.csv', root / 'tiny_gpt2', root / 'outputs', 'cpu')
    for split in ('train', 'val', 'test'):
        ds, _ = data_provider(case_args, split)
        payload['datasets'][f'{index}_{split}'] = dict(
            x=torch.as_tensor(ds.data_x.copy()), y=torch.as_tensor(ds.data_y.copy()),
            markers=torch.as_tensor(ds.data_stamp.copy()), count=len(ds),
            first=[torch.as_tensor(value.copy()) for value in ds[0][:4]],
            last=[torch.as_tensor(value.copy()) for value in ds[len(ds)-1][:4]],
            inverse=torch.as_tensor(ds.inverse_transform(ds.data_y).copy()))
truth = np.array([0., 1., 0., 5., 9., 3.], dtype=np.float32)
prediction = np.array([-.1, 1.2, .4, 4.3, 8., 3.7], dtype=np.float32)
payload['metrics'] = metric_set(truth, prediction)
torch.save(payload, root / 'reference.pt')
"""


@unittest.skipUnless(
    os.environ.get("ADAPTIVELLM_LEGACY_ROOT"), "Optional legacy package not selected"
)
class LegacyParityTests(unittest.TestCase):
    def test_checkpoint_forward_backward_windows_and_metrics(self):
        legacy_root = Path(os.environ["ADAPTIVELLM_LEGACY_ROOT"]).resolve()
        self.assertTrue((legacy_root / "run.py").is_file())
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with fixture() as (root, config, args, _):
                # Include leading gaps and gaps around partition boundaries.
                frame = pd.read_csv(root / "synthetic.csv")
                frame.loc[[0, 1, 383, 384, 575, 576], "Active_Power"] = np.nan
                frame.loc[[0, 200, 400], config.feature_cols[0]] = np.nan
                frame.to_csv(root / "synthetic.csv", index=False)
                (root / "reference_config.json").write_text(
                    json.dumps(asdict(config)), encoding="utf-8"
                )
                subprocess.run(
                    [sys.executable, "-B", "-c", REFERENCE_RUNNER, str(root)],
                    cwd=legacy_root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                reference = torch.load(root / "reference.pt", weights_only=True)
                model = Model(args).float().eval()
                self.assertEqual(set(model.state_dict()), set(reference["state"]))
                model.load_state_dict(reference["state"], strict=True)
                dataset, _ = data_provider(args, "train")
                batch = [
                    torch.as_tensor(np.stack([dataset[i][field] for i in range(16)]))
                    for field in range(4)
                ]
                prediction, target = forecast(model, batch, args)
                # Separate processes/layouts can select different floating-point kernels.
                torch.testing.assert_close(prediction, reference["output"], rtol=1e-5, atol=1e-6)
                regularizer = model.prompt_diversity_loss(prediction)
                torch.testing.assert_close(
                    regularizer, reference["regularizer"], rtol=1e-5, atol=1e-7
                )
                loss = torch.nn.functional.huber_loss(prediction, target, delta=args.huber_delta)
                (loss + args.prompt_diversity_weight * regularizer).backward()
                gradients = {
                    name: p.grad for name, p in model.named_parameters() if p.grad is not None
                }
                self.assertEqual(set(gradients), set(reference["gradients"]))
                for name, gradient in gradients.items():
                    torch.testing.assert_close(
                        gradient, reference["gradients"][name], rtol=1e-5, atol=1e-7
                    )
                for index, (ratios, label_len) in enumerate(
                    [
                        ((0.5, 0.25, 0.25), 96),
                        ((0.6, 0.2, 0.2), 48),
                        ((0.5, 0.25, 0.25), 0),
                    ]
                ):
                    case_config = replace(
                        config,
                        train_ratio=ratios[0],
                        val_ratio=ratios[1],
                        test_ratio=ratios[2],
                        label_len=label_len,
                    )
                    case_args = case_config.runtime(
                        root / "synthetic.csv",
                        root / "tiny_gpt2",
                        root / "outputs",
                        "cpu",
                    )
                    for split in ("train", "val", "test"):
                        with self.subTest(case=index, split=split):
                            dataset, _ = data_provider(case_args, split)
                            expected = reference["datasets"][f"{index}_{split}"]
                            self.assertEqual(len(dataset), expected["count"])
                            for array, key in [
                                (dataset.data_x, "x"),
                                (dataset.data_y, "y"),
                                (dataset.data_stamp, "markers"),
                                (dataset.inverse_transform(dataset.data_y), "inverse"),
                            ]:
                                np.testing.assert_allclose(
                                    array, expected[key].numpy(), rtol=1e-12, atol=1e-12
                                )
                            for position, key in [
                                (0, "first"),
                                (len(dataset) - 1, "last"),
                            ]:
                                for actual, stored in zip(dataset[position], expected[key]):
                                    np.testing.assert_allclose(
                                        actual, stored.numpy(), rtol=1e-12, atol=1e-12
                                    )
                actual = compute_metrics(
                    np.array([0.0, 1.0, 0.0, 5.0, 9.0, 3.0], dtype=np.float32),
                    np.array([-0.1, 1.2, 0.4, 4.3, 8.0, 3.7], dtype=np.float32),
                )
                self.assertEqual(actual, reference["metrics"])
        finally:
            torch.set_num_threads(threads)
