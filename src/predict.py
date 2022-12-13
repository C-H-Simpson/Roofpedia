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
from src.plain_dataloader import get_plain_dataset_loader
from src.unet import UNet


def predict(tiles_dir, mask_dir, tile_size, device, chkpt, batch_size=1):
    # load device
    net = UNet(2).to(device)
    net = nn.DataParallel(net)
    net.load_state_dict(chkpt["state_dict"])
    net.eval()

    loader = get_plain_dataset_loader(tile_size, batch_size, tiles_dir)
    assert len(loader)

    # don't track tensors with autograd during prediction
    with torch.no_grad():
        for images, tiles, img_paths in tqdm(
            loader, desc="Eval", unit="batch", ascii=True
        ):
            images = images.to(device)
            outputs = net(images)

            # manually compute segmentation mask class probabilities per pixel
            probs = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()

            # Write the tiles to raster files.
            for prob, img_path in zip(probs, img_paths):
                mask = np.argmax(prob, axis=0)
                mask = mask * 200
                mask = mask.astype(np.uint8)

                palette = make_palette("dark", "light")
                out = Image.fromarray(mask, mode="P")
                out.putpalette(palette)

                path = Path(mask_dir) / img_path.parent.stem / img_path.name
                path.parent.mkdir(exist_ok=True)

                out.save(path, optimize=True)

                # Copy the image metadata to keep it as a valid georeferenced raster.
                metadata_path = str(img_path) + ".aux.xml"
                shutil.copy(metadata_path, path.parent)

    print("Prediction Done, saved masks to " + mask_dir)


if __name__ == "__main__":
    config = toml.load("config/predict-config.toml")

    city_name = config["city_name"]
    target_type = config["target_type"]
    tiles_dir = os.path.join("results", "02Images", city_name)
    mask_dir = os.path.join("results", "03Masks", target_type, city_name)
    checkpoint_path = Path(config["checkpoint_path"]) / "final_checkpoint.pth"

    tile_size = config["img_size"]

    # load checkpoints
    device = torch.device("cuda")
    chkpt = torch.load(checkpoint_path, map_location=device)

    predict(tiles_dir, mask_dir, tile_size, device, chkpt)
