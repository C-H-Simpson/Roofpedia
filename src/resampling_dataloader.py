import random


class BackgroundResamplingLoader(Dataset):
    """Do resampling with replacement, compensating for class imbalance.

    If signal_p == -1, then return in actual proportions.
    """

    def __init__(self, signal_tiles, background_tiles, signal_p=0.5):
        self.signal_tiles = signal_tiles
        self.background_tiles = background_tiles

        self.n_signal = len(self.signal_tiles)
        self.n_background = len(self.background_tiles)
        self.length = self.n_signal + self.n_background
        assert signal_p == -1 or (signal_p >= 0 and signal_p <= 1)

    def __iter__(self):
        if signal_p != -1:
            if random.random() < signal_p:
                yield self.signal_tiles[int(random.random() * self.n_signal)]
            else:
                yield self.background_tiles[int(random.random() * self.n_background)]
        else:
            idx = random.random() * self.length
            if idx < self.n_signal:
                yield self.signal_tiles[idx]
            else:
                yield self.background_tiles[idx - self.n_signal]
