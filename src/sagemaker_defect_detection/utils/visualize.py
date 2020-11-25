from typing import Iterable, List, Union, Tuple

import numpy as np

import torch

from matplotlib import pyplot as plt

import cv2

import torch


TEXT_COLOR = (255, 255, 255)  # White
CLASSES = {
    "crazing": "Cr",
    "inclusion": "In",
    "pitted_surface": "PS",
    "patches": "Pa",
    "rolled-in_scale": "RS",
    "scratches": "Sc",
}
CATEGORY_ID_TO_NAME = {i: name for i, name in enumerate(CLASSES.keys(), start=1)}


def unnormalize_to_hwc(
    image: torch.Tensor, mean: List[float] = [0.485, 0.456, 0.406], std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    Unnormalizes and a normlized image tensor [0, 1] CHW -> HWC [0, 255]

    Parameters
    ----------
    image : torch.Tensor
        Normalized image
    mean : List[float], optional
        RGB averages used in normalization, by default [0.485, 0.456, 0.406] from imagenet1k
    std : List[float], optional
        RGB standard deviations used in normalization, by default [0.229, 0.224, 0.225] from imagenet1k

    Returns
    -------
    np.ndarray
        Unnormalized image as numpy array
    """
    image = image.numpy().transpose(1, 2, 0)  # HWC
    image = (image * std + mean).clip(0, 1)
    image = (image * 255).astype(np.uint8)
    return image


def visualize_bbox(img: np.ndarray, bbox: np.ndarray, class_name: str, color, thickness: int = 2) -> np.ndarray:
    """
    Uses cv2 to draw colored bounding boxes and class names in an image

    Parameters
    ----------
    img : np.ndarray
        [description]
    bbox : np.ndarray
        [description]
    class_name : str
        Class name
    color : tuple
        BGR tuple
    thickness : int, optional
        Bouding box thickness, by default 2
    """
    x_min, y_min, x_max, y_max = tuple(map(int, bbox))

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(
    image: np.ndarray,
    bboxes: Iterable[Union[torch.Tensor, np.ndarray]] = [],
    category_ids: Iterable[Union[torch.Tensor, np.ndarray]] = [],
    colors: Iterable[Tuple[int, int, int]] = [],
    titles: Iterable[str] = [],
    category_id_to_name=CATEGORY_ID_TO_NAME,
    dpi=150,
) -> None:
    """
    Applies the bounding boxes and category ids to an image

    Parameters
    ----------
    image : np.ndarray
        Image as numpy array
    bboxes : Iterable[Union[torch.Tensor, np.ndarray]], optional
        Bouding boxes, by default []
    category_ids : Iterable[Union[torch.Tensor, np.ndarray]], optional
        Category ids, by default []
    colors : Iterable[Tuple[int, int, int]], optional
        Colors for each bounding box, by default [()]
    titles : Iterable[str], optional
        Titles for each image, by default []
    category_id_to_name : Dict[str, str], optional
        Dictionary of category ids to names, by default CATEGORY_ID_TO_NAME
    dpi : int, optional
        DPI for clarity, by default 150
    """
    bboxes, category_ids, colors, titles = list(map(list, [bboxes, category_ids, colors, titles]))  # type: ignore
    n = len(bboxes)
    assert (
        n == len(category_ids) == len(colors) == len(titles) - 1
    ), f"number of bboxes, category ids, colors and titles (minus one) do not match"

    plt.figure(dpi=dpi)
    ncols = n + 1
    plt.subplot(1, ncols, 1)
    img = image.copy()
    plt.axis("off")
    plt.title(titles[0])
    plt.imshow(image)
    if not len(bboxes):
        return

    titles = titles[1:]
    for i in range(2, ncols + 1):
        img = image.copy()
        plt.subplot(1, ncols, i)
        plt.axis("off")
        j = i - 2
        plt.title(titles[j])
        for bbox, category_id in zip(bboxes[j], category_ids[j]):  # type: ignore
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.numpy()

            if isinstance(category_id, torch.Tensor):
                category_id = category_id.numpy()

            if isinstance(category_id, np.ndarray):
                category_id = category_id.item()

            class_name = category_id_to_name[category_id]
            img = visualize_bbox(img, bbox, class_name, color=colors[j])

        plt.imshow(img)
    return
