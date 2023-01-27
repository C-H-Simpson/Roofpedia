# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import cv2 as cv

from src.augmentations import get_transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

aug = get_transforms()

# %%


plt.rcParams["savefig.bbox"] = "tight"
orig_img = np.asarray(cv.imread("dataset/k1/training_s/images/531536/181323.png"))
orig_mask = np.asarray((cv.imread(str("dataset/k1/training_s/labels/531536/181323.png"), cv.IMREAD_GRAYSCALE) > 200).astype("int64"))
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
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
out = aug["flip_rotate_A"](image=orig_img, mask=orig_mask)
img = out["image"]
mask = out["mask"]
img = np.transpose(out["image"], (1, 2, 0))
plot([orig_img, img, orig_mask, mask], with_orig=False)
plt.show()

# %%
out = aug["medium_augs_B"](image=orig_img, mask=orig_mask)
img = out["image"]
mask = out["mask"]
img = np.transpose(out["image"], (1, 2, 0))
plot([orig_img, img, orig_mask, mask], with_orig=False)
plt.show()

# %%
out = A.RandomGamma((50, 300), p=1)(image=orig_img, mask=orig_mask)
img = out["image"]
mask = out["mask"]
# img = np.transpose(out["image"], (1, 2, 0))
plot([orig_img, img, orig_mask, mask], with_orig=False)
plt.show()
# %%
out = A.ElasticTransform(alpha=1, sigma=1, p=1)(image=orig_img, mask=orig_mask)
img = out["image"]
mask = out["mask"]
# img = np.transpose(out["image"], (1, 2, 0))
plot([orig_img, img, orig_mask, mask], with_orig=False)
plt.show()
# %%
out = A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, p=1)(image=orig_img, mask=orig_mask)
img = out["image"]
mask = out["mask"]
# img = np.transpose(out["image"], (1, 2, 0))
plot([orig_img, img, orig_mask, mask], with_orig=False)
plt.show()
# %%
out = aug["non_spatial_D"](image=orig_img, mask=orig_mask)
img = out["image"]
mask = out["mask"]
img = np.transpose(out["image"], (1, 2, 0))
orig_norm = aug["no_augs_A"](image=orig_img)["image"]
orig_norm = np.transpose(orig_norm, (1, 2, 0))
plot([orig_norm, img, orig_mask, mask], with_orig=False)
plt.show()

# %%
out = aug["blackout"](image=orig_img, mask=orig_mask)
img = out["image"]
mask = out["mask"]
img = np.transpose(out["image"], (1, 2, 0))
orig_norm = aug["no_augs_A"](image=orig_img)["image"]
orig_norm = np.transpose(orig_norm, (1, 2, 0))
plot([orig_norm, img, orig_mask, mask], with_orig=False)
plt.show()
# %%
orig_img.std()
