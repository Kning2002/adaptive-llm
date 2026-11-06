"""Portable FP32 training/check/testing entry for AdaptiveLLM Full."""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from configuration import Config
from data_provider.data_factory import data_provider
from models import Model
from utils.metrics import compute_metrics
from utils.numerical_safety import check_gradients, check_parameters, require_finite


def save_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def forecast(model, batch, args):
    x, y, xm, ym = [tensor.float().to(args.device) for tensor in batch]
    output = model(x, xm, None, ym)[:, -args.pred_len :, -1:]
    target = y[:, -args.pred_len :, -1:]
    require_finite("forecast", output, target)
    return output, target


def validate(model, loader, args, criterion):
    model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            prediction, target = forecast(model, batch, args)
            loss = criterion(prediction, target)
            require_finite("validation loss", loss)
            values.append(float(loss))
    return float(np.mean(values))


def train(model, args, output):
    _, train_loader = data_provider(args, "train")
    _, val_loader = data_provider(args, "val")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )
    criterion = torch.nn.HuberLoss(delta=args.huber_delta)
    check_parameters(model, "before training")
    best, bad_epochs = float("inf"), 0
    history = []
    started = time.monotonic()
    for epoch in range(1, args.train_epochs + 1):
        model.train()
        losses, forecast_losses = [], []
        learning_rate = optimizer.param_groups[0]["lr"]
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            prediction, target = forecast(model, batch, args)
            prediction_loss = criterion(prediction, target)
            penalty = model.prompt_diversity_loss(prediction)
            loss = prediction_loss + args.prompt_diversity_weight * penalty
            require_finite("training loss", loss)
            loss.backward()
            check_gradients(model, f"epoch {epoch} step {step}")
            optimizer.step()
            losses.append(float(loss.detach()))
            forecast_losses.append(float(prediction_loss.detach()))
            if step % 100 == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} loss={losses[-1]:.7f}",
                    flush=True,
                )
        check_parameters(model, f"epoch {epoch} boundary")
        validation = validate(model, val_loader, args, criterion)
        history.append(
            dict(
                epoch=epoch,
                train_loss=float(np.mean(losses)),
                train_forecast_loss=float(np.mean(forecast_losses)),
                val_loss=validation,
                learning_rate=learning_rate,
                elapsed_seconds=time.monotonic() - started,
            )
        )
        save_json(output / "history.json", history)
        print(
            f"epoch={epoch} train={history[-1]['train_loss']:.7f} val={validation:.7f}",
            flush=True,
        )
        # Match the previous EarlyStopping rule: equal validation losses are accepted.
        if validation <= best:
            best, bad_epochs = validation, 0
            torch.save(model.state_dict(), output / "best.pt.tmp")
            (output / "best.pt.tmp").replace(output / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping", flush=True)
                break
        # Existing PEMS epoch schedule.
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * 0.95**epoch
    model.load_state_dict(
        torch.load(output / "best.pt", map_location=args.device, weights_only=True)
    )


def test(model, args, output):
    dataset, loader = data_provider(args, "test")
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in loader:
            prediction, target = forecast(model, batch, args)
            for tensor, store in [(prediction, predictions), (target, targets)]:
                array = tensor.cpu().numpy()
                array = dataset.inverse_transform(array.reshape(-1, 1)).reshape(array.shape)
                store.append(array)
    prediction, truth = np.concatenate(predictions), np.concatenate(targets)
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("Non-finite inverse-transformed predictions")
    np.save(output / "prediction.npy", prediction)
    np.save(output / "truth.npy", truth)
    indices = np.arange(0, len(truth), args.pred_len)
    np.save(output / "primary_window_indices.npy", indices)
    result = dict(
        primary=compute_metrics(truth[indices], prediction[indices]),
        all_windows=compute_metrics(truth, prediction),
        shape=list(truth.shape),
        test_windows=len(truth),
        primary_windows=len(indices),
        normalization_denominator=float(np.ptp(truth[indices])),
        scoring="Ordered test windows [::pred_len]; no target-based clipping or filtering",
    )
    save_json(output / "metrics.json", result)
    print(json.dumps(result["primary"], indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Prepared CSV with date and numeric fields",
    )
    parser.add_argument(
        "--llm-path", type=Path, required=True, help="Local GPT-2 config and weights"
    )
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--mode", choices=["train", "test", "check"], default="train")
    parser.add_argument("--checkpoint", type=Path, help="Required in test mode")
    options = parser.parse_args()
    if options.mode == "test" and not options.checkpoint:
        parser.error("--checkpoint is required in test mode")
    if options.output.exists():
        raise FileExistsError("Use a new output directory")
    config = Config.load(options.config)
    device = (
        ("cuda:0" if torch.cuda.is_available() else "cpu")
        if options.device == "auto"
        else options.device
    )
    args = config.runtime(options.data, options.llm_path, options.output, device)
    options.output.mkdir(parents=True)
    save_json(options.output / "config.json", vars(args))
    save_json(
        options.output / "runtime.json",
        dict(
            torch=torch.__version__,
            device=device,
            precision="float32",
            data_sha256=hashlib.sha256(options.data.read_bytes()).hexdigest(),
            gpt2_config_sha256=hashlib.sha256(
                (options.llm_path / "config.json").read_bytes()
            ).hexdigest(),
        ),
    )
    seed_all(args.seed)
    model = Model(args).float().to(args.device)
    if options.mode == "check":
        _, loader = data_provider(args, "train")
        prediction, target = forecast(model, next(iter(loader)), args)
        loss = torch.nn.functional.huber_loss(prediction, target, delta=args.huber_delta)
        loss = loss + args.prompt_diversity_weight * model.prompt_diversity_loss(prediction)
        loss.backward()
        check_gradients(model, "check")
        print(f"CHECK PASS: output={list(prediction.shape)}, loss={float(loss.detach()):.7f}")
    else:
        if options.mode == "train":
            train(model, args, options.output)
        else:
            model.load_state_dict(
                torch.load(options.checkpoint, map_location=args.device, weights_only=True)
            )
        test(model, args, options.output)
    save_json(options.output / "COMPLETE.json", dict(mode=options.mode, status="COMPLETED"))


if __name__ == "__main__":
    main()
