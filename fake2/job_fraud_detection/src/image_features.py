import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Tuple

import torch
from PIL import Image
from torchvision import models, transforms


DEFAULT_BACKBONE = os.environ.get("JOBSHIELD_IMAGE_BACKBONE", "efficientnet_b0")


@dataclass(frozen=True)
class ImageEmbeddingResult:
    embedding: torch.Tensor  # (dim,)
    dim: int


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def _load_backbone(backbone_name: str = DEFAULT_BACKBONE) -> Tuple[Any, transforms.Compose, torch.device, int]:
    device = _get_device()

    if backbone_name.startswith("resnet"):
        # e.g. resnet18, resnet50
        ctor = getattr(models, backbone_name)
        model = ctor(weights="DEFAULT")
        model.eval()
        model.to(device)

        # Strip classifier head: use global pooled features from before fc
        # For ResNet, children: ... avgpool, fc
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # infer dim by running a dummy
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(device)
            out = feature_extractor(dummy)
            dim = int(out.view(1, -1).shape[1])
        return feature_extractor, preprocess, device, dim

    # EfficientNet family default
    # e.g. efficientnet_b0
    ctor = getattr(models, backbone_name)
    model = ctor(weights="DEFAULT")
    model.eval()
    model.to(device)

    # EfficientNet features: use model.features then global avgpool
    feature_extractor = torch.nn.Sequential(model.features, model.avgpool)

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        out = feature_extractor(dummy)
        dim = int(out.view(1, -1).shape[1])

    return feature_extractor, preprocess, device, dim


def compute_image_embedding(image_path: str, backbone_name: str = DEFAULT_BACKBONE) -> ImageEmbeddingResult:
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")

    feature_extractor, preprocess, device, dim = _load_backbone(backbone_name)

    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = feature_extractor(x)
        emb = feats.view(1, -1).squeeze(0).detach().cpu()

    # ensure 1D
    emb = emb.flatten()
    if emb.numel() != dim:
        dim = int(emb.numel())

    return ImageEmbeddingResult(embedding=emb, dim=dim)


def compute_image_embedding_from_upload(uploaded_file: Any, backbone_name: str = DEFAULT_BACKBONE) -> ImageEmbeddingResult:
    """uploaded_file: werkzeug FileStorage

    Saves nothing to disk by default; reads bytes into PIL.
    """
    feature_extractor, preprocess, device, dim = _load_backbone(backbone_name)

    img = Image.open(uploaded_file.stream).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = feature_extractor(x)
        emb = feats.view(1, -1).squeeze(0).detach().cpu()

    emb = emb.flatten()
    if emb.numel() != dim:
        dim = int(emb.numel())

    return ImageEmbeddingResult(embedding=emb, dim=dim)

