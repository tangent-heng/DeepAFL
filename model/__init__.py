# -*- coding: utf-8 -*-

import torch
from typing import Dict, Tuple, Union, Optional, Callable


from torchvision.models import WeightsEnum
import torchvision
from torch.nn import Flatten
from model import resnet_imagenet

from torchvision.models.resnet import (
    ResNet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)

from .CifarResNet import (
    CifarResNet,
    resnet20,
    resnet32,
    resnet44,
    resnet56,
    resnet110,
    resnet1202,
)


from torchvision.models.vision_transformer import (
    VisionTransformer,
    vit_b_16,
    vit_b_32,
    vit_l_16,
    vit_l_32,
    vit_h_14,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
    ViT_H_14_Weights,
)


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "CifarResNet",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
    "VisionTransformer",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
    "get_backbone_model",
]

# fmt: off
models: Dict[str, Tuple[
        int,    # Input image size
        Callable[[], Union[CifarResNet, ResNet, VisionTransformer, Flatten]], # Model constructor
        Optional[WeightsEnum]
    ]
] = {
    "Flatten": (28, Flatten, None),
    "resnet20":   (32, resnet20  , None),
    "resnet32":   (32, resnet32  , None),
    "resnet44":   (32, resnet44  , None),
    "resnet56":   (32, resnet56  , None),
    "resnet110":  (32, resnet110 , None),
    "resnet1202": (32, resnet1202, None),
    "resnet18":  (224, resnet18,  ResNet18_Weights.DEFAULT ),
    "resnet34":  (224, resnet34,  ResNet34_Weights.DEFAULT ),
    "resnet50":  (224, resnet50,  ResNet50_Weights.DEFAULT ),
    "resnet101": (224, resnet101, ResNet101_Weights.DEFAULT),
    "resnet152": (224, resnet152, ResNet152_Weights.DEFAULT),
    "vit_b_16": (224, vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
    "vit_b_32": (224, vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1         ),
    "vit_l_16": (512, vit_l_16, ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1),
    "vit_l_32": (224, vit_l_32, ViT_L_32_Weights.IMAGENET1K_V1         ),
    "vit_h_14": (518, vit_h_14, ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1),

}



def get_backbone_model(args):
    """
    Get backbone model based on arguments

    Args:
        args: Command line arguments

    Returns:
        Backbone model
    """
    model = None

    if args.pretrained:
        print(f"=> using pre-trained model '{args.backbone}'")

        if args.backbone == 'resnet18':
            model_pretrain = torchvision.models.__dict__[args.backbone](pretrained=True)
            model = resnet_imagenet.__dict__[args.backbone](args.num_classes)
            model.load_state_dict(model_pretrain.state_dict(), strict=False)
            args.feat_size = model_pretrain.fc.weight.size(1)
        elif 'resnet' in args.backbone:
            if args.backbone in resnet_imagenet.__dict__:
                model_pretrain = torchvision.models.__dict__[args.backbone](pretrained=True)
                model = resnet_imagenet.__dict__[args.backbone]()
                model.load_state_dict(model_pretrain.state_dict(), strict=False)
                args.feat_size = model_pretrain.fc.weight.size(1)
        else:
            raise ValueError(f"=> such model not exist now: '{args.backbone}'")
    else:
        raise ValueError(f"=> pre-trained model required: '{args.backbone}'")

    return model
