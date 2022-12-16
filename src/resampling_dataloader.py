from torch.utils.data import Dataset
import random


class BackgroundResamplingLoader(Dataset):
    """Do resampling with replacement, compensating for class imbalance.

    If signal_p == -1, then return in actual proportions.
    If signal_p == 1, then return signal only.
    If signal_p is between 0 and 1, return a resampled amount of background that
     makes the signal proportion correct.

    Show each signal tile the correct number of times, but
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
            self.length = int(self.n_signal / self.signal_p)
            self.n_background = self.length - self.n_signal

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i < self.n_background:
            return random.choice(self.background_tiles)
        else:
            idx = (i - self.n_background) % self.n_signal
            return self.signal_tiles[idx]


class SignalResamplingLoader(Dataset):
    """Do resampling with replacement, compensating for class imbalance."""

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
            self.background_tiles[i]
        else:
            idx = (i - self.n_background) % self.n_signal
            return self.signal_tiles[idx]
