"""Experiment configuration and global constants."""
from dataclasses import dataclass


@dataclass
class Cfg:
    epochs: int = 40
    hidden: int = 64
    seed: int = 0
    rate_reg: float = 0.0     # L1 firing-rate penalty (sparsity)
    n_levels: int = 6         # encoder levels per channel
    batch: int = 64
    lr_snn: float = 2e-3
    lr_lstm: float = 5e-3


WARM = 10   # time steps ignored in the loss (state warm-up)
SKIP = 50   # samples ignored at the start of each test stream
