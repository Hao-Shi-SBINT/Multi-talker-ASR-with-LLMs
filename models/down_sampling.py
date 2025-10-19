import torch
import torch.nn as nn
import torch.nn.functional as F

class WavLMPostDownsample(nn.Module):
    """
    Two 1D-convolution layers applied AFTER WavLM to downsample time steps
    (default total stride = 4) and optionally change the channel dimension.

    Inputs:
      x:        (B, T, D_in)  — WavLM last_hidden_state
      lengths:  Optional[(B,)] — valid lengths at WavLM frame resolution

    Outputs:
      y:            (B, T', D_out) — downsampled sequence
      new_lengths:  Optional[(B,)]  — lengths updated by the conv strides
    """
    def __init__(
        self,
        d_in: int,
        d_mid: int | None = None,
        d_out: int | None = None,
        k1: int = 3, s1: int = 2, dlt1: int = 1,   # first conv: kernel/stride/dilation
        k2: int = 3, s2: int = 2, dlt2: int = 1,   # second conv: kernel/stride/dilation
        dropout: float = 0.1,
        use_bn: bool = True,
        act=F.gelu,
    ):
        super().__init__()
        d_mid = d_mid or d_in
        d_out = d_out or d_in
        self.act = act
        self.do = nn.Dropout(dropout)

        def same_pad(k, d):  # “same-ish” padding to preserve centers
            return (d * (k - 1)) // 2

        # Conv1d expects (B, C, T), so we'll transpose in forward()
        self.conv1 = nn.Conv1d(d_in,  d_mid, kernel_size=k1, stride=s1, dilation=dlt1, padding=same_pad(k1, dlt1))
        self.conv2 = nn.Conv1d(d_mid, d_out, kernel_size=k2, stride=s2, dilation=dlt2, padding=same_pad(k2, dlt2))
        self.norm1 = nn.BatchNorm1d(d_mid) if use_bn else nn.Identity()
        self.norm2 = nn.BatchNorm1d(d_out) if use_bn else nn.Identity()

        # Cache params for length computation
        self.k1, self.s1, self.dlt1, self.p1 = k1, s1, dlt1, same_pad(k1, dlt1)
        self.k2, self.s2, self.dlt2, self.p2 = k2, s2, dlt2, same_pad(k2, dlt2)

    @torch.no_grad()
    def _conv_out_len(self, L: torch.Tensor, k, s, d, p):
        # Standard Conv1d length formula: floor((L + 2p - d*(k-1) - 1)/s + 1)
        return torch.div(L + 2*p - d*(k - 1) - 1, s, rounding_mode='floor') + 1

    @torch.no_grad()
    def update_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        # Apply the two Conv1d length transforms
        L1 = self._conv_out_len(lengths, self.k1, self.s1, self.dlt1, self.p1)
        L2 = self._conv_out_len(L1,     self.k2, self.s2, self.dlt2, self.p2)
        return L2.clamp_min(0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        # (B, T, D) -> (B, D, T) for Conv1d
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.do(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.do(x)

        # (B, D_out, T') -> (B, T', D_out)
        y = x.transpose(1, 2)

        new_lengths = self.update_lengths(lengths) if lengths is not None else None
        return y, new_lengths

