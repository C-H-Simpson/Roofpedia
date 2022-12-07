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
    # JointFullyRandomRotation, # doesn't work
    JointRandomCrop,
    JointRandomHorizontalFlip,
    JointRandomRotation,
    JointRandomVerticalFlip,
    JointTransform,
    MaskToTensor,
)

import torchvision.transforms as T


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
        # brightness5=JointCompose(
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
        #             ColorJitter(
        #                 brightness=0.5,
        #             ),
        #             None,
        #         ),
        #         JointTransform(ImageToTensor(), MaskToTensor()),
        #         JointTransform(Normalize(mean=mean, std=std), None),
        #     ]
        # ),
        # hue2=JointCompose(
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
        #             ColorJitter(
        #                 hue=0.2,
        #             ),
        #             None,
        #         ),
        #         JointTransform(ImageToTensor(), MaskToTensor()),
        #         JointTransform(Normalize(mean=mean, std=std), None),
        #     ]
        # ),
        # constrast5=JointCompose(
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
        #             ColorJitter(
        #                 contrast=0.5,
        #             ),
        #             None,
        #         ),
        #         JointTransform(ImageToTensor(), MaskToTensor()),
        #         JointTransform(Normalize(mean=mean, std=std), None),
        #     ]
        # ),
        colorjitter_sharpness=JointCompose(
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
                    ColorJitter(contrast=0.3, brightness=0.2, hue=0.1),
                    None,
                ),
                JointTransform(
                    RandomAdjustSharpness(2, 0.5),
                    None,
                ),
                JointTransform(
                    RandomAdjustSharpness(2, 0.5),
                    None,
                ),
                JointTransform(ImageToTensor(), MaskToTensor()),
                JointTransform(Normalize(mean=mean, std=std), None),
            ]
        ),
        # blurring5=JointCompose(
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
        #             RandomAdjustSharpness(0.5),
        #             None,
        #         ),
        #         JointTransform(ImageToTensor(), MaskToTensor()),
        #         JointTransform(Normalize(mean=mean, std=std), None),
        #     ]
        # ),
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
        persp_01=JointCompose(
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
                JointTransform(T.RandomPerspective(distortion_scale=0.1, p=0.9), None),
                JointTransform(ImageToTensor(), MaskToTensor()),
                JointTransform(Normalize(mean=mean, std=std), None),
            ]
        ),
        persp_03=JointCompose(
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
                JointTransform(T.RandomPerspective(distortion_scale=0.3, p=0.9), None),
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
        #             T.ElasticTransform(alpha=2, sigma=0.06), # Doesn't work
        #             None,
        #         ),
        #         JointTransform(ImageToTensor(), MaskToTensor()),
        #         JointTransform(Normalize(mean=mean, std=std), None),
        #     ]
        # ),
        # flips_and_fullrotations=JointCompose(
        #     [
        #         JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
        #         JointTransform(
        #             Resize(target_size, Image.BILINEAR),
        #             Resize(target_size, Image.NEAREST),
        #         ),
        #         JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
        #         JointRandomHorizontalFlip(0.5),
        #         JointRandomVerticalFlip(0.5),
        #         # JointFullyRandomRotation(360), // doesn't work
        #         JointTransform(ImageToTensor(), MaskToTensor()),
        #         JointTransform(Normalize(mean=mean, std=std), None),
        #     ]
        # ),
        # flips_and_rotations_and_crops=JointCompose(
        #     [
        #         JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
        #         JointRandomCrop( # doesn't work
        #             (target_size, target_size),
        #             (int(target_size * 0.9), int(target_size * 0.9)),
        #         ),
        #         JointTransform(
        #             Resize(target_size, Image.BILINEAR),
        #             Resize(target_size, Image.NEAREST),
        #         ),
        #         JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
        #         JointRandomHorizontalFlip(0.5),
        #         JointRandomVerticalFlip(0.5),
        #         # JointFullyRandomRotation(360), # doesn't work
        #         JointRandomRotation(0.5, 90),
        #         JointRandomRotation(0.5, 90),
        #         JointRandomRotation(0.5, 90),
        #         JointTransform(ImageToTensor(), MaskToTensor()),
        #         JointTransform(Normalize(mean=mean, std=std), None),
        #     ]
        # ),
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
    )
    return augs
