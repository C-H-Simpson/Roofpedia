import random

from torch.utils.data import Dataset


class BackgroundResamplingLoader(Dataset):
    """Do resampling with replacement, compensating for class imbalance.

    If signal_p == -1, then return in actual proportions.
    """

    def __init__(self, signal_tiles, background_tiles, signal_p=0.5):
        self.signal_tiles = signal_tiles
        self.background_tiles = background_tiles

        self.n_signal = len(self.signal_tiles)
        self.n_background = len(self.background_tiles)
        self.length = int(self.n_background / (1 - signal_p))

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i < self.n_background:
            return self.background_tiles[i]
        else:
            idx = (i - self.n_background) % self.n_signal
            return self.signal_tiles[idx]
