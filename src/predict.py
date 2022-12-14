import os
import shutil
from pathlib import Path

import numpy as np
import toml
import torch
import torch.backends.cudnn
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from src.colors import make_palette
from src.plain_dataloader import get_named_dataset_loader
from src.unet import UNet


def predict(tiles_dir, mask_dir, tile_size, device, chkpt, batch_size=1):
    # load device
    net = UNet(2).to(device)
    net = nn.DataParallel(net)
    net.load_state_dict(chkpt["state_dict"])
    net.eval()

    loader = get_named_dataset_loader(tile_size, batch_size, tiles_dir)
    assert len(loader)

    # don't track tensors with autograd during prediction
    with torch.no_grad():
        for images, _, (X, Y) in tqdm(
            loader, desc="Prediction", unit="batch", ascii=True
        ):
            images = images.to(device)
            outputs = net(images)

            # manually compute segmentation mask class probabilities per pixel
            probs = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()

            # Write the tiles to raster files.
            for prob, _x, _y in zip(probs, X, Y):
                tilename = f"{_x.item():d}/{_y.item():d}.png"
                mask = np.argmax(prob, axis=0)
                mask = mask * 200
                mask = mask.astype(np.uint8)

                palette = make_palette("dark", "light")
                out = Image.fromarray(mask, mode="P")
                out.putpalette(palette)

                path = Path(mask_dir) / tilename
                path.parent.mkdir(exist_ok=True)

                out.save(path, optimize=True)

                # Copy the image metadata to keep it as a valid georeferenced raster.
                metadata_path = str(Path(tiles_dir) / "images" / f"{tilename}.aux.xml")
                shutil.copy(metadata_path, path.parent)

    print("Prediction Done, saved masks to ", mask_dir)
