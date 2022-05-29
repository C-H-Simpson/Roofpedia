import random

class BackgroundResamplingDataset(Dataset):
    def __init__(self, signal_tiles, background_tiles, signal_p=0.5):
        self.signal_tiles = signal_tiles
        self.background_tiles = background_tiles

        # Construct a list in which the signal tiles appear multiple times
        # but the background tiles appear fewer times.
        n_signal = len(self.signal_tiles)
        n_background = len(self.background_tiles)

        self.idx_list = list(range(len(signal_tiles)))

    def __iter__(self):
        if random.random() > signal_p:
            yield self.background_tiles[int(random.random() * n_background)]
        else:
            yield self.postive_tiles[int(random.random() * n_signal)]
            
