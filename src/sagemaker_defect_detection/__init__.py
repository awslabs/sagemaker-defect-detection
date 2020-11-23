try:
    import pytorch_lightning
except ModuleNotFoundError:
    print("installing the dependencies for sagemaker_defect_detection package ...")
    import subprocess

    subprocess.run(
        "python -m pip install -q albumentations==0.4.6 pytorch_lightning==0.8.5 pycocotools==2.0.1", shell=True
    )

from sagemaker_defect_detection.models.ddn import Classification, Detection, RoI, RPN
from sagemaker_defect_detection.dataset.neu import NEUCLS, NEUDET
from sagemaker_defect_detection.transforms import get_transform, get_augmentation, get_preprocess

__all__ = [
    "Classification",
    "Detection",
    "RoI",
    "RPN",
    "NEUCLS",
    "NEUDET",
    "get_transform",
    "get_augmentation",
    "get_preprocess",
]
