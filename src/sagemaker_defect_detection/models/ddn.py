import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.ops import MultiScaleRoIAlign


def get_backbone(name: str) -> nn.Module:
    """
    Get official pretrained ResNet34 and ResNet50 as backbones

    Parameters
    ----------
    name : str
        Either `resnet34` or `resnet50`

    Returns
    -------
    nn.Module
        resnet34 or resnet50 pytorch modules

    Raises
    ------
    ValueError
        If unsupported name is used
    """
    if name == "resnet34":
        return torchvision.models.resnet34(pretrained=True)
    elif name == "resnet50":
        return torchvision.models.resnet50(pretrained=True)
    else:
        raise ValueError("Unsupported backbone")


def init_weights(m) -> None:
    """
    Weight initialization

    Parameters
    ----------
    m : [type]
        Module used in recursive call
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)

    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.0)

    return


class MFN(nn.Module):
    def __init__(self, backbone: str):
        """
        Implementation of MFN model as described in

        Yu He, Kechen Song, Qinggang Meng, Yunhui Yan,
        “An End-to-end Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features,”
        IEEE Transactions on Instrumentation and Measuremente, 2020,69(4),1493-1504.

        Parameters
        ----------
        backbone : str
            Either `resnet34` or `resnet50`
        """
        super().__init__()
        self.backbone = get_backbone(backbone)
        # input 224x224 -> conv1 output size 112x112
        self.start_layer = nn.Sequential(
            self.backbone.conv1,  # type: ignore
            self.backbone.bn1,  # type: ignore
            self.backbone.relu,  # type: ignore
            self.backbone.maxpool,  # type: ignore
        )
        self.r2 = self.backbone.layer1  # 64/256x56x56 <- (resnet34/resnet50)
        self.r3 = self.backbone.layer2  # 128/512x28x28
        self.r4 = self.backbone.layer3  # 256/1024x14x14
        self.r5 = self.backbone.layer4  # 512/2048x7x7
        in_channel = 64 if backbone == "resnet34" else 256
        self.b2 = nn.Sequential(
            nn.Conv2d(
                in_channel, in_channel, kernel_size=3, padding=1, stride=2
            ),  # 56 -> 28 without Relu or batchnorm not in the paper ???
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=2),  # 28 -> 14
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel * 2, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channel * 2),
            nn.ReLU(inplace=True),
        ).apply(
            init_weights
        )  # after r2: 128/512x14x14  <-
        self.b3 = nn.MaxPool2d(2)  # after r3: 128/512x14x14  <-
        in_channel *= 2  # 128/512
        self.b4 = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        ).apply(
            init_weights
        )  # after r4: 128/512x14x14
        in_channel *= 4  # 512 / 2048
        self.b5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channel, in_channel, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # <- after r5 which is 512x7x7 -> 512x14x14
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel // 4, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU(inplace=True),
        ).apply(init_weights)

        self.out_channels = 512 if backbone == "resnet34" else 2048  # required for FasterRCNN

    def forward(self, x):
        x = self.start_layer(x)
        x = self.r2(x)
        b2_out = self.b2(x)
        x = self.r3(x)
        b3_out = self.b3(x)
        x = self.r4(x)
        b4_out = self.b4(x)
        x = self.r5(x)
        b5_out = self.b5(x)
        # BatchNorm works better than L2 normalize
        # out = torch.cat([F.normalize(o, p=2, dim=1) for o in (b2_out, b3_out, b4_out, b5_out)], dim=1)
        out = torch.cat((b2_out, b3_out, b4_out, b5_out), dim=1)
        return out


class Classification(nn.Module):
    """
    Classification network

    Parameters
    ----------
    backbone : str
        Either `resnet34` or `resnet50`

    num_classes : int
        Number of classes
    """

    def __init__(self, backbone: str, num_classes: int) -> None:
        super().__init__()
        self.mfn = MFN(backbone)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.mfn.out_channels * 14 ** 2, num_classes)

    def forward(self, x):
        return self.fc(self.flatten(self.mfn(x)))


class RPN(nn.Module):
    """
    RPN Module as described in

    Yu He, Kechen Song, Qinggang Meng, Yunhui Yan,
    “An End-to-end Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features,”
    IEEE Transactions on Instrumentation and Measuremente, 2020,69(4),1493-1504.
    """

    def __init__(
        self,
        out_channels: int = 512,
        rpn_pre_nms_top_n_train: int = 1000,  # torchvision default 2000,
        rpn_pre_nms_top_n_test: int = 500,  # torchvision default 1000,
        rpn_post_nms_top_n_train: int = 1000,  # torchvision default 2000,
        rpn_post_nms_top_n_test: int = 500,  # torchvision default 1000,
        rpn_nms_thresh: float = 0.7,
        rpn_fg_iou_thresh: float = 0.7,
        rpn_bg_iou_thresh: float = 0.3,
        rpn_batch_size_per_image: int = 256,
        rpn_positive_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        rpn_anchor_generator = AnchorGenerator(sizes=((64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
        )

    def forward(self, *args, **kwargs):
        return self.rpn(*args, **kwargs)


class CustomTwoMLPHead(nn.Module):
    def __init__(self, in_channels: int, representation_size: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, representation_size),
            nn.ReLU(inplace=True),
            nn.Linear(representation_size, representation_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        return x


class RoI(nn.Module):
    """
    ROI Module as described in

    Yu He, Kechen Song, Qinggang Meng, Yunhui Yan,
    “An End-to-end Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features,”
    IEEE Transactions on Instrumentation and Measuremente, 2020,69(4),1493-1504.
    """

    def __init__(
        self,
        num_classes: int,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
    ) -> None:
        super().__init__()
        roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
        box_head = CustomTwoMLPHead(512 * 7 ** 2, 1024)
        box_predictor = FastRCNNPredictor(1024, num_classes=num_classes)
        self.roi_head = RoIHeads(
            roi_pooler,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

    def forward(self, *args, **kwargs):
        return self.roi_head(*args, **kwargs)


class Detection(GeneralizedRCNN):
    """
    Detection network as described in

    Yu He, Kechen Song, Qinggang Meng, Yunhui Yan,
    “An End-to-end Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features,”
    IEEE Transactions on Instrumentation and Measuremente, 2020,69(4),1493-1504.
    """

    def __init__(self, mfn, rpn, roi):
        dummy_transform = GeneralizedRCNNTransform(800, 1333, [00.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        super().__init__(mfn, rpn, roi, dummy_transform)
