"""Periodic decomposition and historical covariate encoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.conv_blocks import InceptionBlock
from layers.embedding import DataEmbedding


def fft_periods(x, k=2):
    # Sequence length 96 is not a power of two: CUDA half FFT is unsupported.
    with torch.autocast(device_type=x.device.type, enabled=False):
        xf = torch.fft.rfft(x.float(), dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0

    max_k = max(1, min(k, frequency_list.shape[0]))
    _, top_list = torch.topk(frequency_list, max_k)

    safe_top = torch.clamp(top_list, min=1)
    period = x.shape[1] // safe_top
    period = torch.clamp(period, min=1)
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        top_k,
        num_kernels,
    ):
        super().__init__()
        self.k = top_k
        self.conv = nn.Sequential(
            InceptionBlock(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlock(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        # x: [B, T, N]
        bsz, t_len, channels = x.size()
        period_list, period_weight = fft_periods(x, self.k)

        res = []
        for i in range(period_list.shape[0]):
            period = max(int(period_list[i].item()), 1)

            if t_len % period != 0:
                length = ((t_len // period) + 1) * period
                pad_len = length - t_len
                padding = torch.zeros([bsz, pad_len, channels], device=x.device, dtype=x.dtype)
                out = torch.cat([x, padding], dim=1)
            else:
                length = t_len
                out = x

            out = (
                out.reshape(bsz, length // period, period, channels)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(bsz, -1, channels)
            res.append(out[:, :t_len, :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, t_len, channels, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x


class TargetPeriodDecomp(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = int(top_k)

    def forward(self, x):
        # x: [B, T, C]
        xf = torch.fft.rfft(x, dim=1)
        freq = torch.abs(xf).clone()
        if freq.shape[1] > 0:
            freq[:, 0, :] = 0

        k = max(1, min(self.top_k, freq.shape[1]))
        top_values = torch.topk(freq, k=k, dim=1).values
        threshold = top_values[:, -1:, :]
        keep_mask = freq >= threshold
        if keep_mask.shape[1] > 0:
            keep_mask = keep_mask.clone()
            keep_mask[:, 0, :] = False

        periodic = torch.fft.irfft(xf.masked_fill(~keep_mask, 0), n=x.shape[1], dim=1)
        residual = x - periodic
        return periodic, residual


class CovariateBranch(nn.Module):
    def __init__(self, configs, c_in):
        super().__init__()
        self.embedding = DataEmbedding(c_in, configs.d_model, "timeF", "t", configs.dropout)
        self.times_block = TimesBlock(
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            top_k=configs.top_k,
            num_kernels=configs.num_kernels,
        )
        self.times_norm = nn.LayerNorm(configs.d_model)

    def forward(self, x_enc_other, x_mark_enc):
        means = x_enc_other.mean(1, keepdim=True).detach()
        x_enc_other = x_enc_other - means
        stdev = torch.sqrt(torch.var(x_enc_other, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_other = x_enc_other / stdev

        enc_out = self.embedding(x_enc_other, x_mark_enc)
        enc_out = self.times_norm(self.times_block(enc_out))
        return enc_out
