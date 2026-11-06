"""Fail fast on non-finite tensors, gradients and checkpoints."""

import torch


def require_finite(label, *values):
    for index, value in enumerate(values):
        if value is None:
            continue
        tensor = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(
                f"{label}: non-finite tensor #{index}, shape={tuple(tensor.shape)}"
            )


def check_parameters(model, label):
    # Called at initialization/epoch boundaries, not on every training step.
    for name, parameter in model.named_parameters():
        require_finite(f"{label}: parameter {name}", parameter)
    for name, buffer in model.named_buffers():
        require_finite(f"{label}: buffer {name}", buffer)


def check_gradients(model, label):
    # Explicit failure is required for unscaled gradients, not overflowed *scaled*
    # AMP gradients (GradScaler may legitimately skip such an update).
    gradients = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
    if not gradients:
        return
    flags = torch.stack([torch.isfinite(g).all() for _, g in gradients])
    if not flags.all():
        names = [name for (name, _), good in zip(gradients, flags.cpu().tolist()) if not good]
        raise FloatingPointError(f"{label}: non-finite gradients: {names[:8]}")
