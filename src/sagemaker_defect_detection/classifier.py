# mypy: ignore-errors
from typing import Dict
import os
from collections import OrderedDict
from argparse import ArgumentParser, Namespace
from multiprocessing import cpu_count

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning.metrics.functional as plm

from sagemaker_defect_detection import Classification, NEUCLS, get_transform
from sagemaker_defect_detection.utils import load_checkpoint, freeze


def metrics(name: str, out: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    pred = torch.argmax(out, 1).detach()
    target = target.detach()
    metrics = {}
    metrics[name + "_acc"] = plm.accuracy(pred, target)
    metrics[name + "_prec"] = plm.precision(pred, target)
    metrics[name + "_recall"] = plm.recall(pred, target)
    metrics[name + "_f1_score"] = plm.recall(pred, target)
    return metrics


class DDNClassification(pl.LightningModule):
    def __init__(
        self,
        data_path: str,
        backbone: str,
        freeze_backbone: bool,
        num_classes: int,
        learning_rate: float,
        batch_size: int,
        momentum: float,
        weight_decay: float,
        seed: int,
        **kwargs
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.seed = seed

        self.train_dataset = NEUCLS(self.data_path, split="train", transform=get_transform("train"), seed=self.seed)
        self.val_dataset = NEUCLS(self.data_path, split="val", transform=get_transform("val"), seed=self.seed)
        self.test_dataset = NEUCLS(self.data_path, split="test", transform=get_transform("test"), seed=self.seed)

        self.model = Classification(self.backbone, self.num_classes)
        if self.freeze_backbone:
            for param in self.model.mfn.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):  # ignore
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        metrics_dict = metrics("train", output, target)
        tqdm_dict = {"train_loss": loss_val, **metrics_dict}
        output = OrderedDict({"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict})
        return output

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        metrics_dict = metrics("val", output, target)
        output = OrderedDict({"val_loss": loss_val, **metrics_dict})
        return output

    def validation_epoch_end(self, outputs):
        log_dict = {}
        for metric_name in outputs[0]:
            log_dict[metric_name] = torch.stack([x[metric_name] for x in outputs]).mean()

        return {"log": log_dict, "progress_bar": log_dict, **log_dict}

    def test_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        metrics_dict = metrics("test", output, target)
        output = OrderedDict({"test_loss": loss_val, **metrics_dict})
        return output

    def test_epoch_end(self, outputs):
        log_dict = {}
        for metric_name in outputs[0]:
            log_dict[metric_name] = torch.stack([x[metric_name] for x in outputs]).mean()

        return {"log": log_dict, "progress_bar": log_dict, **log_dict}

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay
        )
        return optimizer

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count() // 2,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
        )
        return test_loader

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        aa = parser.add_argument
        aa(
            "--data-path",
            metavar="DIR",
            type=str,
            default=os.getenv("SM_CHANNEL_TRAINING", ""),
        )
        aa(
            "--backbone",
            default="resnet34",
        )
        aa(
            "--freeze-backbone",
            action="store_true",
        )
        aa(
            "--num-classes",
            default=6,
            type=int,
            metavar="N",
        )
        aa(
            "-b",
            "--batch-size",
            default=64,
            type=int,
            metavar="N",
        )
        aa(
            "--lr",
            "--learning-rate",
            default=1e-3,
            type=float,
            metavar="LR",
            dest="learning_rate",
        )
        aa("--momentum", default=0.9, type=float, metavar="M", help="momentum")
        aa(
            "--wd",
            "--weight-decay",
            default=1e-4,
            type=float,
            metavar="W",
            dest="weight_decay",
        )
        aa(
            "--seed",
            type=int,
            default=42,
        )
        return parser


def get_args() -> Namespace:
    parent_parser = ArgumentParser(add_help=False)
    aa = parent_parser.add_argument
    aa("--epochs", type=int, default=100, help="number of training epochs")
    aa("--save-path", metavar="DIR", default=os.getenv("SM_MODEL_DIR", ""), type=str, help="path to save output")
    aa("--gpus", type=int, default=os.getenv("SM_NUM_GPUS", 1), help="how many gpus")
    aa(
        "--distributed-backend",
        type=str,
        default="",
        choices=("dp", "ddp", "ddp2"),
        help="supports three options dp, ddp, ddp2",
    )
    aa("--use-16bit", dest="use_16bit", action="store_true", help="if true uses 16 bit precision")

    parser = DDNClassification.add_model_specific_args(parent_parser)
    return parser.parse_args()


def model_fn(model_dir):
    # TODO: `model_fn` doesn't get more args
    # see: https://github.com/aws/sagemaker-inference-toolkit/issues/65
    backbone = "resnet34"
    num_classes = 6

    model = load_checkpoint(Classification(backbone, num_classes), model_dir, prefix="model")
    model = model.eval()
    freeze(model)
    return model


def main(args: Namespace) -> None:
    model = DDNClassification(**vars(args))

    if args.seed is not None:
        pl.seed_everything(args.seed)
        if torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(args.seed)

    # TODO: add deterministic training
    # torch.backends.cudnn.deterministic = True

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_path, "{epoch}-{val_loss:.3f}-{val_acc:.3f}"),
        save_top_k=1,
        verbose=True,
        monitor="val_acc",
        mode="max",
    )
    early_stop_callback = EarlyStopping("val_loss", patience=10)
    trainer = pl.Trainer(
        default_root_dir=args.save_path,
        gpus=args.gpus,
        max_epochs=args.epochs,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=10,
        num_sanity_val_steps=0,
        distributed_backend=args.distributed_backend or None,
        # precision=16 if args.use_16bit else 32, # TODO: amp apex support
    )

    trainer.fit(model)
    trainer.test()
    return


if __name__ == "__main__":
    main(get_args())
