# %%
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T


plt.rcParams["savefig.bbox"] = "tight"
orig_img = Image.open(
    r"C:\Users\ucbqc38\Documents\RoofPedia\Roofpedia_vc\dataset\1s\images\19\262051\174218.png"
)
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(
        nrows=num_rows, ncols=num_cols, squeeze=False, figsize=(10, 10)
    )
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title="Original image")
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


# %%
perspective_transformer = T.RandomPerspective(distortion_scale=0.3, p=1.0)
perspective_imgs = [perspective_transformer(orig_img) for _ in range(4)]
plot(perspective_imgs)
# %%
augmenter = T.AugMix()
imgs = [augmenter(orig_img) for _ in range(4)]
plot(imgs)

# %%
brightness_transformer = T.ColorJitter(brightness=0.5)
brightness_imgs = [brightness_transformer(orig_img) for _ in range(4)]
plot(brightness_imgs)

# %%
brightness_transformer = T.ColorJitter(brightness=0.2, contrast=0.3, hue=0.1)
brightness_imgs = [brightness_transformer(orig_img) for _ in range(4)]
plot(brightness_imgs)
# %%

# %%
brightness_transformer = T.RandomAdjustSharpness(1, p=1.0)
brightness_imgs = [brightness_transformer(orig_img) for _ in range(4)]
plot(brightness_imgs)
# %%
