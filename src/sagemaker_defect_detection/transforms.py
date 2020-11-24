from typing import Callable
import torchvision.transforms as transforms

import albumentations as albu
import albumentations.pytorch.transforms as albu_transforms


PROBABILITY = 0.5
ROTATION_ANGLE = 90
NUM_CHANNELS = 3
# required for resnet
IMAGE_RESIZE_HEIGHT = 256
IMAGE_RESIZE_WIDTH = 256
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
# standard imagenet1k mean and standard deviation of RGB channels
MEAN_RED = 0.485
MEAN_GREEN = 0.456
MEAN_BLUE = 0.406
STD_RED = 0.229
STD_GREEN = 0.224
STD_BLUE = 0.225


def get_transform(split: str) -> Callable:
    """
    Image data transformations such as normalization for train split for classification task

    Parameters
    ----------
    split : str
        train or else

    Returns
    -------
    Callable
        Image transformation function
    """
    normalize = transforms.Normalize(mean=[MEAN_RED, MEAN_GREEN, MEAN_BLUE], std=[STD_RED, STD_GREEN, STD_BLUE])
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_HEIGHT),
                transforms.RandomRotation(ROTATION_ANGLE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    else:
        return transforms.Compose(
            [
                transforms.Resize(IMAGE_RESIZE_HEIGHT),
                transforms.CenterCrop(IMAGE_HEIGHT),
                transforms.ToTensor(),
                normalize,
            ]
        )


def get_augmentation(split: str) -> Callable:
    """
    Obtains proper image augmentation in train split for detection task.
    We have splitted transformations done for detection task into augmentation and preprocessing
    for clarity

    Parameters
    ----------
    split : str
        train or else

    Returns
    -------
    Callable
        Image augmentation function
    """
    if split == "train":
        return albu.Compose(
            [
                albu.Resize(IMAGE_RESIZE_HEIGHT, IMAGE_RESIZE_WIDTH, always_apply=True),
                albu.RandomCrop(IMAGE_HEIGHT, IMAGE_WIDTH, always_apply=True),
                albu.RandomRotate90(p=PROBABILITY),
                albu.HorizontalFlip(p=PROBABILITY),
                albu.RandomBrightness(p=PROBABILITY),
            ],
            bbox_params=albu.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_visibility=0.2,
            ),
        )
    else:
        return albu.Compose(
            [albu.Resize(IMAGE_HEIGHT, IMAGE_WIDTH)],
            bbox_params=albu.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )


def get_preprocess() -> Callable:
    """
    Image normalization using albumentation for detection task that aligns well with image augmentation

    Returns
    -------
    Callable
        Image normalization function
    """
    return albu.Compose(
        [
            albu.Normalize(mean=[MEAN_RED, MEAN_GREEN, MEAN_BLUE], std=[STD_RED, STD_GREEN, STD_BLUE]),
            albu_transforms.ToTensorV2(),
        ]
    )
