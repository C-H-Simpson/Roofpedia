import random

from torch.utils.data import Dataset


class BackgroundResamplingLoader(Dataset):
    """Do resampling with replacement, compensating for class imbalance.

    If signal_p == -1, then return in actual proportions.
    If signal_p == 1, then return signal only.
    """

    def __init__(self, signal_tiles, background_tiles, signal_p=0.5):
        self.signal_p = signal_p
        self.signal_tiles = signal_tiles
        self.background_tiles = background_tiles

        self.n_signal = len(self.signal_tiles)
        self.n_background = len(self.background_tiles)
        if signal_p == 1:
            self.length = self.n_signal
            self.n_background = 0
        elif signal_p == -1:
            self.length = self.n_background + self.n_signal
        else:
            assert (signal_p > 0) and (signal_p < 1)
            self.length = int(self.n_background / (1 - signal_p))

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i < self.n_background:
            return self.background_tiles[i]
        else:
            idx = (i - self.n_background) % self.n_signal
            return self.signal_tiles[idx]
