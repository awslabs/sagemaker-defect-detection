from typing import List, Tuple, Optional, Callable
import os
from pathlib import Path
from collections import namedtuple

from xml.etree.ElementTree import ElementTree

import numpy as np

import cv2

import torch
from torch.utils.data.dataset import Dataset

from torchvision.datasets import ImageFolder


class NEUCLS(ImageFolder):
    """
    NEU-CLS dataset processing and loading
    """

    def __init__(
        self,
        root: str,
        split: str,
        augmentation: Optional[Callable] = None,
        preprocessing: Optional[Callable] = None,
        seed: int = 123,
        **kwargs,
    ) -> None:
        """
        NEU-CLS dataset

        Parameters
        ----------
        root : str
            Dataset root path
        split : str
            Data split from train, val and test
        augmentation : Optional[Callable], optional
            Image augmentation function, by default None
        preprocess : Optional[Callable], optional
            Image preprocessing function, by default None
        seed : int, optional
            Random number generator seed, by default 123

        Raises
        ------
        ValueError
            If unsupported split is used
        """
        super().__init__(root, **kwargs)
        self.samples: List[Tuple[str, int]]
        self.split = split
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        n_items = len(self.samples)
        np.random.seed(seed)
        perm = np.random.permutation(list(range(n_items)))
        # TODO: add split ratios as parameters
        train_end = int(0.6 * n_items)
        val_end = int(0.2 * n_items) + train_end
        if split == "train":
            self.samples = [self.samples[i] for i in perm[:train_end]]
        elif split == "val":
            self.samples = [self.samples[i] for i in perm[train_end:val_end]]
        elif split == "test":
            self.samples = [self.samples[i] for i in perm[val_end:]]
        else:
            raise ValueError(f"Unknown split mode. Choose from `train`, `val` or `test`. Given {split}")


DetectionSample = namedtuple("DetectionSample", ["image_path", "class_idx", "annotations"])


class NEUDET(Dataset):
    """
    NEU-DET dataset processing and loading
    """

    def __init__(
        self,
        root: str,
        split: str,
        augmentation: Optional[Callable] = None,
        preprocess: Optional[Callable] = None,
        seed: int = 123,
    ):
        """
        NEU-DET dataset

        Parameters
        ----------
        root : str
            Dataset root path
        split : str
            Data split from train, val and test
        augmentation : Optional[Callable], optional
            Image augmentation function, by default None
        preprocess : Optional[Callable], optional
            Image preprocessing function, by default None
        seed : int, optional
            Random number generator seed, by default 123

        Raises
        ------
        ValueError
            If unsupported split is used
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.classes, self.class_to_idx = self._find_classes()
        self.samples: List[DetectionSample] = self._make_dataset()
        self.augmentation = augmentation
        self.preprocess = preprocess
        n_items = len(self.samples)
        np.random.seed(seed)
        perm = np.random.permutation(list(range(n_items)))
        train_end = int(0.6 * n_items)
        val_end = int(0.2 * n_items) + train_end
        if split == "train":
            self.samples = [self.samples[i] for i in perm[:train_end]]
        elif split == "val":
            self.samples = [self.samples[i] for i in perm[train_end:val_end]]
        elif split == "test":
            self.samples = [self.samples[i] for i in perm[val_end:]]
        else:
            raise ValueError(f"Unknown split mode. Choose from `train`, `val` or `test`. Given {split}")

    def _make_dataset(self) -> List[DetectionSample]:
        instances = []
        base_dir = self.root.expanduser()
        for target_cls in sorted(self.class_to_idx.keys()):
            cls_idx = self.class_to_idx[target_cls]
            target_dir = base_dir / target_cls
            if not target_dir.is_dir():
                continue

            images = sorted(list((target_dir / "images").glob("*.jpg")))
            annotations = sorted(list((target_dir / "annotations").glob("*.xml")))
            assert len(images) == len(annotations), f"something is wrong. Mismatched number of images and annotations"
            for path, ann in zip(images, annotations):
                instances.append(DetectionSample(str(path), int(cls_idx), str(ann)))

        return instances

    def _find_classes(self):
        classes = sorted([d.name for d in os.scandir(str(self.root)) if d.is_dir()])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes, 1)}  # no bg label in NEU
        return classes, class_to_idx

    @staticmethod
    def _get_bboxes(ann: str) -> List[List[int]]:
        tree = ElementTree().parse(ann)
        bboxes = []
        for bndbox in tree.iterfind("object/bndbox"):
            # should subtract 1 like coco?
            bbox = [int(bndbox.findtext(t)) - 1 for t in ("xmin", "ymin", "xmax", "ymax")]  # type: ignore
            assert bbox[2] > bbox[0] and bbox[3] > bbox[1], f"box size error, given {bbox}"
            bboxes.append(bbox)

        return bboxes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        # Note: images are grayscaled BUT resnet needs 3 channels
        image = cv2.imread(self.samples[idx].image_path)  # BGR channel last
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = self._get_bboxes(self.samples[idx].annotations)
        num_objs = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor([self.samples[idx].class_idx] * num_objs, dtype=torch.int64)
        image_id = torch.tensor([idx], dtype=torch.int64)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd

        if self.augmentation is not None:
            sample = self.augmentation(**{"image": image, "bboxes": boxes, "labels": labels})
            image = sample["image"]
            target["boxes"] = torch.as_tensor(sample["bboxes"], dtype=torch.float32)
            # guards against crops that don't pass the min_visibility augmentation threshold
            if not target["boxes"].numel():
                return None

            target["labels"] = torch.as_tensor(sample["labels"], dtype=torch.int64)

        if self.preprocess is not None:
            image = self.preprocess(image=image)["image"]

        boxes = target["boxes"]
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        return image, target, image_id

    def collate_fn(self, batch):
        batch = filter(lambda x: x is not None, batch)
        return tuple(zip(*batch))
