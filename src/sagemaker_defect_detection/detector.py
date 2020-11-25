# mypy: ignore-errors
from typing import Optional
from pathlib import Path
from os import path as osp
from collections import OrderedDict
from argparse import ArgumentParser, Namespace
from multiprocessing import cpu_count
import os
import math
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.models.detection.image_list import ImageList

import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from sagemaker_defect_detection import Detection, NEUDET, Classification, RPN, RoI, get_augmentation, get_preprocess

from sagemaker_defect_detection.utils.coco_eval import CocoEvaluator
from sagemaker_defect_detection.utils.coco_utils import convert_to_coco_api
from sagemaker_defect_detection.utils import freeze, load_checkpoint


class DDNDetection(pl.LightningModule):
    def __init__(
        self,
        train_rpn: bool,
        train_roi: bool,
        finetune_rpn: bool,
        finetune_roi: bool,
        data_path: str,
        backbone: str,
        num_classes: int,
        learning_rate: float,
        batch_size: int,
        momentum: float,
        weight_decay: float,
        seed: int,
        pretrained_mfn_ckpt: Optional[str] = None,
        pretrained_rpn_ckpt: Optional[str] = None,
        pretrained_roi_ckpt: Optional[str] = None,
        finetuned_rpn_ckpt: Optional[str] = None,
        finetuned_roi_ckpt: Optional[str] = None,
        resume_sagemaker_from_checkpoint: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.train_rpn = train_rpn
        self.train_roi = train_roi
        self.finetune_rpn = finetune_rpn
        self.finetune_roi = finetune_roi
        self.data_path = data_path
        self.backbone = backbone
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.seed = seed

        self.train_dataset = NEUDET(
            self.data_path,
            split="train",
            augmentation=get_augmentation("train"),
            preprocess=get_preprocess(),
            seed=self.seed,
        )
        self.val_dataset = NEUDET(
            self.data_path,
            split="val",
            augmentation=get_augmentation("val"),
            preprocess=get_preprocess(),
            seed=self.seed,
        )

        self.pretrained_mfn_ckpt = pretrained_mfn_ckpt
        self.pretrained_rpn_ckpt = pretrained_rpn_ckpt
        self.pretrained_roi_ckpt = pretrained_roi_ckpt
        self.finetuned_rpn_ckpt = finetuned_rpn_ckpt
        self.finetuned_roi_ckpt = finetuned_roi_ckpt
        self.resume_sagemaker_from_checkpoint = resume_sagemaker_from_checkpoint

        self.coco_evaluator = self._get_evaluator(self.val_dataset)

    def setup(self, stage) -> None:
        if self.train_rpn:  # step 2
            self.mfn = load_checkpoint(
                Classification(self.backbone, self.num_classes - 1).mfn, self.pretrained_mfn_ckpt, "model.mfn"
            )
            self.rpn = RPN()

        elif self.train_roi:  # step 3
            self.mfn = load_checkpoint(
                Classification(self.backbone, self.num_classes - 1).mfn, self.pretrained_rpn_ckpt, prefix="mfn"
            )
            freeze(self.mfn)

            self.rpn = load_checkpoint(RPN(), self.pretrained_rpn_ckpt, prefix="rpn")
            freeze(self.rpn)

            self.roi = RoI(self.num_classes)

        elif self.finetune_rpn:  # step 4 or extra finetune rpn
            if self.finetuned_rpn_ckpt and self.finetuned_roi_ckpt:  # extra finetune rpn
                self.mfn = load_checkpoint(
                    Classification(self.backbone, self.num_classes - 1).mfn, self.finetuned_rpn_ckpt, prefix="mfn"
                )
                freeze(self.mfn)

                self.rpn = load_checkpoint(RPN(), self.finetuned_rpn_ckpt, prefix="rpn")

                self.roi = load_checkpoint(RoI(self.num_classes), self.finetuned_roi_ckpt, prefix="roi")
                freeze(self.roi)

                self.model = Detection(self.mfn, self.rpn, self.roi)

            else:
                self.mfn = load_checkpoint(
                    Classification(self.backbone, self.num_classes - 1).mfn, self.pretrained_rpn_ckpt, prefix="mfn"
                )
                freeze(self.mfn)

                self.rpn = load_checkpoint(RPN(), self.pretrained_rpn_ckpt, prefix="rpn")

                self.roi = load_checkpoint(RoI(self.num_classes), self.pretrained_roi_ckpt, prefix="roi")
                freeze(self.roi)

                self.model = Detection(self.mfn, self.rpn, self.roi)

        elif self.finetune_roi:  # step 5 or extra finetune roi
            if self.finetuned_rpn_ckpt and self.finetuned_roi_ckpt:  # extra finetune roi
                self.mfn = load_checkpoint(
                    Classification(self.backbone, self.num_classes - 1).mfn, self.finetuned_rpn_ckpt, prefix="mfn"
                )
                freeze(self.mfn)

                self.rpn = load_checkpoint(RPN(), self.finetuned_rpn_ckpt, prefix="rpn")
                freeze(self.rpn)

                self.roi = load_checkpoint(RoI(self.num_classes), self.finetuned_roi_ckpt, prefix="roi")

                self.model = Detection(self.mfn, self.rpn, self.roi)

            else:
                self.mfn = load_checkpoint(
                    Classification(self.backbone, self.num_classes - 1).mfn, self.finetuned_rpn_ckpt, prefix="mfn"
                )
                freeze(self.mfn)

                self.rpn = load_checkpoint(RPN(), self.finetuned_rpn_ckpt, prefix="rpn")
                freeze(self.rpn)

                self.roi = load_checkpoint(RoI(self.num_classes), self.pretrained_roi_ckpt, prefix="roi")

                self.model = Detection(self.mfn, self.rpn, self.roi)

        else:  # step 6: final/joint model
            load_checkpoint_fn = load_checkpoint
            if self.finetuned_roi_ckpt is not None:
                ckpt_path = self.finetuned_rpn_ckpt
            elif self.resume_sagemaker_from_checkpoint is not None:
                ckpt_path = self.resume_sagemaker_from_checkpoint
            else:
                ckpt_path = None
                # ignore load_checkpoint
                load_checkpoint_fn = lambda *args: args[0]

            self.mfn = load_checkpoint_fn(Classification(self.backbone, self.num_classes - 1).mfn, ckpt_path, "mfn")
            self.rpn = load_checkpoint_fn(RPN(), ckpt_path, "rpn")
            self.roi = load_checkpoint_fn(RoI(self.num_classes), ckpt_path, "roi")
            self.model = Detection(self.mfn, self.rpn, self.roi)

        return

    @auto_move_data
    def forward(self, images, *args, **kwargs):
        if self.train_rpn:  # step 2
            images = torch.stack(images)
            features = self.mfn(images)
            features = OrderedDict({str(i): t.unsqueeze(0) for i, t in enumerate(features)})
            images = ImageList(images, [(224, 224)])
            return self.rpn(images, features, targets=kwargs.get("targets"))

        elif self.train_roi:  # step 3
            self.mfn.eval()
            self.rpn.eval()
            images = torch.stack(images)
            features = self.mfn(images)
            features = OrderedDict({str(i): t.unsqueeze(0) for i, t in enumerate(features)})
            images = ImageList(images, [(224, 224)])
            proposals, _ = self.rpn(images, features, targets=None)
            return self.roi(features, proposals, [(224, 224)], targets=kwargs.get("targets"))

        elif self.finetune_rpn:
            self.model.backbone.eval()
            self.model.roi_heads.eval()
            return self.model(images, targets=kwargs.get("targets"))

        elif self.finetune_roi:
            self.model.backbone.eval()
            self.model.rpn.eval()
            return self.model(images, targets=kwargs.get("targets"))
        else:
            return self.model(images, targets=kwargs.get("targets"))

    def _get_evaluator(self, dataset):
        coco = convert_to_coco_api(dataset)
        return CocoEvaluator(coco, ["bbox"])

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            shuffle=True,
            num_workers=cpu_count(),
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.val_dataset.collate_fn,
            shuffle=False,
            num_workers=cpu_count() // 2,
        )

        self.coco_evaluator = self._get_evaluator(val_loader.dataset)
        return val_loader

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        if self.train_rpn:
            targets = [{"boxes": t["boxes"]} for t in targets]
            _, loss_dict = self(images, targets=targets)
            loss = sum(loss for loss in loss_dict.values())
            return OrderedDict({"loss": loss, "progress_bar": loss_dict, "log": loss_dict})

        elif self.train_roi:
            _, loss_dict = self(images, targets=targets)
            loss = sum(loss for loss in loss_dict.values())
            return OrderedDict({"loss": loss, "progress_bar": loss_dict, "log": loss_dict})
        else:
            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]
            loss_dict = self(images, targets=targets)
            # loss keys: ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']
            loss = sum(loss for loss in loss_dict.values())
            if not math.isfinite(loss.item()):
                sys.exit(1)

            return OrderedDict({"loss": loss, "progress_bar": loss_dict, "log": loss_dict})

    @auto_move_data
    def validation_step(self, batch, batch_idx):
        images, targets, _ = batch
        if self.train_rpn:  # rpn doesn't compute loss for val
            return {}
        elif self.train_roi:
            # TODO: scores are predictions scores, not a metric! iou? + acc?
            return {}
        else:
            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]
            outputs = self(images, targets=targets)
            ret = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            self.coco_evaluator.update(ret)
            return {}

    @auto_move_data
    def validation_epoch_end(self, outputs):
        if self.train_rpn:
            return {}
        elif self.train_roi:
            # TODO: above
            return {}
        else:
            self.coco_evaluator.synchronize_between_processes()
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()
            metric = self.coco_evaluator.coco_eval["bbox"].stats[0]
            metric = torch.as_tensor(metric)
            tensorboard_logs = {"main_score": metric}
            self.coco_evaluator = self._get_evaluator(self.val_dataset)  # need to update for the new evaluation
            return {"val_loss": metric, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        aa = parser.add_argument
        aa("--train-rpn", action="store_true")
        aa("--train-roi", action="store_true")
        aa("--finetune-rpn", action="store_true")
        aa("--finetune-roi", action="store_true")
        aa("--data-path", metavar="DIR", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
        aa("--backbone", default="resnet34", help="backbone model either resnet34 (default) or resnet50")
        aa("--num-classes", default=7, type=int, metavar="N", help="number of classes including the background")
        aa(
            "-b",
            "--batch-size",
            default=16,
            type=int,
            metavar="N",
            help="mini-batch size (default: 16), this is the total "
            "batch size of all GPUs on the current node when "
            "using Data Parallel or Distributed Data Parallel",
        )
        aa(
            "--lr",
            "--learning-rate",
            default=1e-3,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="learning_rate",
        )
        aa("--momentum", default=0.9, type=float, metavar="M", help="momentum")
        aa(
            "--wd",
            "--weight-decay",
            default=1e-4,
            type=float,
            metavar="W",
            help="weight decay (default: 1e-4)",
            dest="weight_decay",
        )
        aa("--seed", type=int, default=123, help="seed for initializing training")
        aa("--pretrained-mfn-ckpt", type=str)
        aa("--pretrained-rpn-ckpt", type=str)
        aa("--pretrained-roi-ckpt", type=str)
        aa("--finetuned-rpn-ckpt", type=str)
        aa("--finetuned-roi-ckpt", type=str)
        aa("--resume-from-checkpoint", type=str)
        aa("--resume-sagemaker-from-checkpoint", type=str, default=os.getenv("SM_CHANNEL_PRETRAINED_CHECKPOINT", None))
        return parser


def get_args():
    parent_parser = ArgumentParser(add_help=False)
    aa = parent_parser.add_argument
    aa("--epochs", type=int, default=1, help="number of training epochs")
    aa("--save-path", metavar="DIR", default=os.environ["SM_MODEL_DIR"], type=str, help="path to save output")
    aa("--gpus", type=int, default=os.getenv("SM_NUM_GPUS", 1), help="how many gpus")
    aa(
        "--distributed-backend",
        type=str,
        default="",
        choices=("dp", "ddp", "ddp2"),
        help="supports three options dp, ddp, ddp2",
    )
    # aa("--use-16bit", dest="use_16bit", action="store_true", help="if true uses 16 bit precision")

    parser = DDNDetection.add_model_specific_args(parent_parser)
    return parser.parse_args()


def model_fn(model_dir):
    # TODO: `model_fn` doesn't get more args
    # see: https://github.com/aws/sagemaker-inference-toolkit/issues/65
    backbone = "resnet34"
    num_classes = 7  # including the background

    mfn = load_checkpoint(Classification(backbone, num_classes - 1).mfn, model_dir, "mfn")
    rpn = load_checkpoint(RPN(), model_dir, "rpn")
    roi = load_checkpoint(RoI(num_classes), model_dir, "roi")
    model = Detection(mfn, rpn, roi)
    model = model.eval()
    freeze(model)
    return model


def main(args: Namespace) -> None:
    ddn = DDNDetection(**vars(args))

    if args.seed is not None:
        pl.seed_everything(args.seed)  # doesn't do multi-gpu
        if torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(args.seed)

    # TODO: add deterministic training
    # torch.backends.cudnn.deterministic = True

    if ddn.train_rpn:
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(args.save_path, "{epoch}-{loss:.3f}"),
            save_top_k=1,
            verbose=True,
            monitor="loss",
            mode="min",
        )
        early_stop_callback = None

    elif ddn.train_roi:
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(args.save_path, "{epoch}-{loss:.3f}"),
            save_top_k=1,
            verbose=True,
            monitor="loss",
            mode="min",
        )
        early_stop_callback = None

    else:
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(args.save_path, "{epoch}-{loss:.3f}-{main_score:.3f}"),
            save_top_k=1,
            verbose=True,
            monitor="main_score",
            mode="max",
        )
        early_stop_callback = EarlyStopping("main_score", patience=50, mode="max")

    trainer = pl.Trainer(
        default_root_dir=args.save_path,
        num_sanity_val_steps=1,
        limit_val_batches=1.0,
        gpus=args.gpus,
        max_epochs=args.epochs,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        distributed_backend=args.distributed_backend or None,
        # precision=16 if args.use_16bit else 32, # TODO: apex
        weights_summary="top",
        resume_from_checkpoint=None if args.resume_from_checkpoint == "" else args.resume_from_checkpoint,
    )

    trainer.fit(ddn)
    return


if __name__ == "__main__":
    main(get_args())
