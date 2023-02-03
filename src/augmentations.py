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

def norm(img, **kwds):
    return (img.astype("float32") / (img.sum(-1, keepdims=True).astype("float32")+1e-10))

def tofloat(img, **kwds):
    return img.astype("float32")


def get_transforms(target_size=256):
    # using imagenet mean and std for Normalization
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    augs = dict(
        # no_augs=JointCompose(
        #     [
        #         JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
        #         JointTransform(
        #             Resize(target_size, Image.BILINEAR),
        #             Resize(target_size, Image.NEAREST),
        #         ),
        #         JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
        #         JointTransform(ImageToTensor(), MaskToTensor()),
        #         JointTransform(Normalize(mean=mean, std=std), None),
        #     ]
        # ),
        # flips_and_rotations=JointCompose(
        #     [
        #         JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
        #         JointTransform(
        #             Resize(target_size, Image.BILINEAR),
        #             Resize(target_size, Image.NEAREST),
        #         ),
        #         JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
        #         JointRandomHorizontalFlip(0.5),
        #         JointRandomVerticalFlip(0.5),
        #         JointRandomRotation(0.5, 90),
        #         JointRandomRotation(0.5, 90),
        #         JointRandomRotation(0.5, 90),
        #         JointTransform(ImageToTensor(), MaskToTensor()),
        #         JointTransform(Normalize(mean=mean, std=std), None),
        #     ]
        # ),
        # colorjitter=JointCompose(
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
        #             ColorJitter(contrast=0.3, brightness=0.2, hue=0.1), None,
        #         ),
        #         JointTransform(ImageToTensor(), MaskToTensor()),
        #         JointTransform(Normalize(mean=mean, std=std), None),
        #     ]
        # ),
        # sharpening2=JointCompose(
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
        #         JointTransform(RandomAdjustSharpness(2, 0.5), None,),
        #         JointTransform(RandomAdjustSharpness(2, 0.5), None,),
        #         JointTransform(ImageToTensor(), MaskToTensor()),
        #         JointTransform(Normalize(mean=mean, std=std), None),
        #     ]
        # ),
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
        flip_rotate_A=A.Compose(
            [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        ),
        medium_augs_A=A.Compose(
            [
                # based on https://albumentations.ai/docs/examples/example_kaggle_salt/
                A.OneOf(
                    [
                        A.RandomSizedCrop(
                            min_max_height=(50, 101),
                            height=target_size,
                            width=target_size,
                            p=0.5,
                        ),
                        A.PadIfNeeded(
                            min_height=target_size, min_width=target_size, p=0.5
                        ),
                    ],
                    p=1,
                ),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                        ),
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
                    ],
                    p=0.8,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        ),
        medium_augs_B=A.Compose(
            [
                # based on https://albumentations.ai/docs/examples/example_kaggle_salt/
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.CLAHE(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.OneOf(
                    [
                        A.RandomSizedCrop(
                            min_max_height=(200, 255),
                            height=target_size,
                            width=target_size,
                            p=0.5,
                        ),
                        A.PadIfNeeded(
                            min_height=target_size, min_width=target_size, p=0.5
                        ),
                    ],
                    p=0.5,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        ),
        non_spatial_B=A.Compose(
            [
                # based on https://albumentations.ai/docs/examples/example_kaggle_salt/
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.CLAHE(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        ),
        non_spatial_A=A.Compose(
            [
                # based on https://albumentations.ai/docs/examples/example_kaggle_salt/
                A.OneOf(
                    [
                        A.RandomSizedCrop(
                            min_max_height=(50, 101),
                            height=target_size,
                            width=target_size,
                            p=0.5,
                        ),
                        A.PadIfNeeded(
                            min_height=target_size, min_width=target_size, p=0.5
                        ),
                    ],
                    p=1,
                ),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5
                        ),
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
                    ],
                    p=0.8,
                ),
                A.CLAHE(p=0.8),
                A.RandomBrightnessContrast(p=0.8),
                A.RandomGamma(p=0.8),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        ),
        non_spatial_C=A.Compose(
            [
                # add RGBshift to flips etc
                # based on https://albumentations.ai/docs/examples/example_kaggle_salt/
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        ),
        non_spatial_D=A.Compose(
            [
                # add RGBshift and random gamma to flips etc.
                # based on https://albumentations.ai/docs/examples/example_kaggle_salt/
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
                A.RandomGamma((50, 300), p=0.8),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        ),
        non_spatial_E=A.Compose(
            [
                # try slight elastic transform
                # based on https://albumentations.ai/docs/examples/example_kaggle_salt/
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.ElasticTransform(alpha=1, sigma=1, p=1),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
                A.RandomGamma((50, 300), p=0.8),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        ),
        non_spatial_F=A.Compose(
            [
                # try grid distortion
                # based on https://albumentations.ai/docs/examples/example_kaggle_salt/
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.GridDistortion(p=1),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
                A.RandomGamma((50, 300), p=0.8),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        ),
        non_spatial_G=A.Compose(
            [
                # try shift-scale-rotate
                # based on https://albumentations.ai/docs/examples/example_kaggle_salt/
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, p=0.5),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
                A.RandomGamma((50, 300), p=0.8),
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
        no_augs_A=A.Compose(
            [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        ),
        no_augs_B=A.Compose(
            [
                A.Lambda(image=norm),
                ToTensorV2(),
            ]
        ),
        no_augs_C=A.Compose(
            [
                A.Lambda(image=tofloat),
                ToTensorV2(),
            ]
        ),
        non_spatial_E_nonorm=A.Compose(
            [
                # try slight elastic transform
                # based on https://albumentations.ai/docs/examples/example_kaggle_salt/
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.ElasticTransform(alpha=1, sigma=1, p=1),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
                A.RandomGamma((50, 300), p=0.8),
                A.Lambda(image=tofloat),
                ToTensorV2(),
            ]
        ),
        non_spatial_H_nonorm=A.Compose(
            [
                # try slight elastic transform
                # based on https://albumentations.ai/docs/examples/example_kaggle_salt/
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.ElasticTransform(alpha=1, sigma=1, p=1),
                # Don't do RGBShift as we want it to learn colours
                # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
                A.RandomGamma((50, 300), p=0.8),
                A.Lambda(image=tofloat),
                ToTensorV2(),
            ]
        ),
        non_spatial_E_colornorm=A.Compose(
            [
                # try slight elastic transform
                # based on https://albumentations.ai/docs/examples/example_kaggle_salt/
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.ElasticTransform(alpha=1, sigma=1, p=1),
                # Don't do RGBShift as we want it to learn colours
                # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
                A.RandomGamma((50, 300), p=0.8),
                A.Lambda(image=norm),
                ToTensorV2(),
            ]
        ),
        non_spatial_E_colornorm_rgbshift=A.Compose(
            [
                # try slight elastic transform
                # based on https://albumentations.ai/docs/examples/example_kaggle_salt/
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.ElasticTransform(alpha=1, sigma=1, p=1),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
                A.RandomGamma((50, 300), p=0.8),
                A.Lambda(image=norm),
                ToTensorV2(),
            ]
        )
    )
    return augs