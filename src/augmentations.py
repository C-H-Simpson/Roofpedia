from PIL import Image
from torchvision.transforms import (CenterCrop, ColorJitter, Normalize,
                                    RandomAdjustSharpness, Resize)

from src.transforms import (ConvertImageMode, ImageToTensor, JointCompose,
                            JointFullyRandomRotation, JointRandomCrop,
                            JointRandomHorizontalFlip, JointRandomRotation,
                            JointRandomVerticalFlip, JointTransform,
                            MaskToTensor)


def get_transforms(target_size):
    # using imagenet mean and std for Normalization
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    augs = dict(
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
        brightness5=JointCompose(
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
                    ColorJitter(
                        brightness=0.5,
                    ),
                    None,
                ),
                JointTransform(ImageToTensor(), MaskToTensor()),
                JointTransform(Normalize(mean=mean, std=std), None),
            ]
        ),
        hue2=JointCompose(
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
                    ColorJitter(
                        hue=0.2,
                    ),
                    None,
                ),
                JointTransform(ImageToTensor(), MaskToTensor()),
                JointTransform(Normalize(mean=mean, std=std), None),
            ]
        ),
        constrast5=JointCompose(
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
                    ColorJitter(
                        contrast=0.5,
                    ),
                    None,
                ),
                JointTransform(ImageToTensor(), MaskToTensor()),
                JointTransform(Normalize(mean=mean, std=std), None),
            ]
        ),
        blurring5=JointCompose(
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
                    RandomAdjustSharpness(0.5),
                    None,
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
                JointTransform(
                    RandomAdjustSharpness(2),
                    None,
                ),
                JointTransform(ImageToTensor(), MaskToTensor()),
                JointTransform(Normalize(mean=mean, std=std), None),
            ]
        ),
        # elastic206=JointCompose(
        #     [
        #         JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
        #         JointTransform(
        #             Resize(target_size, Image.BILINEAR),
        #             Resize(target_size, Image.NEAREST),
        #         ),
        #         JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
        #         JointRandomRotation(0.5, 90),
        #         JointRandomRotation(0.5, 90),
        #         JointRandomRotation(0.5, 90),
        #         JointRandomHorizontalFlip(0.5),
        #         JointRandomVerticalFlip(0.5),
        #         JointTransform(
        #             ElasticTransform(alpha=2, sigma=0.06),
        #             None,
        #         ),
        #         JointTransform(ImageToTensor(), MaskToTensor()),
        #         JointTransform(Normalize(mean=mean, std=std), None),
        #     ]
        # ),
        flips_and_fullrotations=JointCompose(
            [
                JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
                JointTransform(
                    Resize(target_size, Image.BILINEAR),
                    Resize(target_size, Image.NEAREST),
                ),
                JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
                JointRandomHorizontalFlip(0.5),
                JointRandomVerticalFlip(0.5),
                JointFullyRandomRotation(360),
                JointTransform(ImageToTensor(), MaskToTensor()),
                JointTransform(Normalize(mean=mean, std=std), None),
            ]
        ),
        flips_and_rotations_and_crops=JointCompose(
            [
                JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
                JointRandomCrop(
                    (target_size, target_size),
                    (int(target_size * 0.9), int(target_size * 0.9)),
                ),
                JointTransform(
                    Resize(target_size, Image.BILINEAR),
                    Resize(target_size, Image.NEAREST),
                ),
                JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
                JointRandomHorizontalFlip(0.5),
                JointRandomVerticalFlip(0.5),
                JointFullyRandomRotation(360),
                JointTransform(ImageToTensor(), MaskToTensor()),
                JointTransform(Normalize(mean=mean, std=std), None),
            ]
        ),
    )
    return augs
