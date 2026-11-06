"""The original custom loader with the Full model's existing batch policy."""

import os

from torch.utils.data import DataLoader

from data_provider.dataset import PVForecastDataset


def data_provider(args, flag):
    if flag not in ("train", "val", "test"):
        raise ValueError("flag must be train, val or test")
    dataset = PVForecastDataset(args, split=flag)
    if len(dataset) <= 0:
        raise ValueError(f"Not enough rows for {flag} windows")
    loader = DataLoader(
        dataset,
        batch_size=1 if flag == "test" else args.batch_size,
        shuffle=flag != "test",
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=args.device.type == "cuda",
        persistent_workers=args.num_workers > 0 and os.name != "nt",
    )
    if not len(loader):
        raise ValueError(f"{flag} cannot provide one full batch; reduce batch_size")
    return dataset, loader
