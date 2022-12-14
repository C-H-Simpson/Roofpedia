from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Normalize,
    RandomAdjustSharpness,
    Resize,
)

from src.transforms import (
    ConvertImageMode,
    ImageToTensor,
    JointCompose,
    JointRandomHorizontalFlip,
    JointRandomRotation,
    JointRandomVerticalFlip,
    JointTransform,
    MaskToTensor,
)

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_transforms(target_size):
    # using imagenet mean and std for Normalization
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    augs = dict(
        no_augs=JointCompose(
            [
                JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
                JointTransform(
                    Resize(target_size, Image.BILINEAR),
                    Resize(target_size, Image.NEAREST),
                ),
                JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
                JointTransform(ImageToTensor(), MaskToTensor()),
                JointTransform(Normalize(mean=mean, std=std), None),
            ]
        ),
        flips_and_rotations=JointCompose(
            [
                JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
                JointTransform(
                    Resize(target_size, Image.BILINEAR),
                    Resize(target_size, Image.NEAREST),
                ),
                JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
                JointRandomHorizontalFlip(0.5),
                JointRandomVerticalFlip(0.5),
                JointRandomRotation(0.5, 90),
                JointRandomRotation(0.5, 90),
                JointRandomRotation(0.5, 90),
                JointTransform(ImageToTensor(), MaskToTensor()),
                JointTransform(Normalize(mean=mean, std=std), None),
            ]
        ),
        colorjitter=JointCompose(
            [
                JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
                JointTransform(
                    Resize(target_size, Image.BILINEAR),
                    Resize(target_size, Image.NEAREST),
                ),
                JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
                JointRandomRotation(0.5, 90),
                JointRandomRotation(0.5, 90),
                JointRandomRotation(0.5, 90),
                JointRandomHorizontalFlip(0.5),
                JointRandomVerticalFlip(0.5),
                JointTransform(
                    ColorJitter(contrast=0.3, brightness=0.2, hue=0.1), None,
                ),
                JointTransform(ImageToTensor(), MaskToTensor()),
                JointTransform(Normalize(mean=mean, std=std), None),
            ]
        ),
        sharpening2=JointCompose(
            [
                JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
                JointTransform(
                    Resize(target_size, Image.BILINEAR),
                    Resize(target_size, Image.NEAREST),
                ),
                JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
                JointRandomRotation(0.5, 90),
                JointRandomRotation(0.5, 90),
                JointRandomRotation(0.5, 90),
                JointRandomHorizontalFlip(0.5),
                JointRandomVerticalFlip(0.5),
                JointTransform(RandomAdjustSharpness(2, 0.5), None,),
                JointTransform(RandomAdjustSharpness(2, 0.5), None,),
                JointTransform(ImageToTensor(), MaskToTensor()),
                JointTransform(Normalize(mean=mean, std=std), None),
            ]
        ),
        albumentations=A.Compose(
            [
                A.Resize(256, 256),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, p=0.5),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.5
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        ),
        blackout=A.Compose(
            [
                A.Resize(256, 256),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, p=0.5),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.5
                ),
                A.Normalize(mean=mean, std=std),
                A.MaskDropout((0, 1)),
                A.augmentations.dropout.coarse_dropout.CoarseDropout(
                    p=1, max_height=16, max_width=16, max_holes=8
                ),
                ToTensorV2(),
            ]
        ),
    )
    return augs
