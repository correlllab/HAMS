"""Lazy runtime bridge to the official VLMaps LSeg implementation.

The benchmark owns the streaming map so it can query after every frame.  Dense
pixel features and CLIP text features still come from the official VLMaps LSeg
code and checkpoint.  Heavy dependencies are imported only when this backend
is selected; the normal SigLIP/FAISS benchmark image stays unchanged.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np


OFFICIAL_LSEG_CHECKPOINT_URL = (
    "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
)


def _load_official_checkpoint(torch, checkpoint: Path, device: str) -> dict:
    """Safely load tensors from the legacy Lightning checkpoint.

    The released file pickles one ``ModelCheckpoint`` callback in addition to
    its tensor state dict.  VLMaps never uses that callback at inference time.
    A minimal allowlisted class keeps the runtime independent of the obsolete
    training-only PyTorch Lightning stack while retaining ``weights_only``
    deserialization.
    """
    parent_name = "pytorch_lightning"
    callbacks_name = f"{parent_name}.callbacks"
    module_name = f"{callbacks_name}.model_checkpoint"
    parent = sys.modules.get(parent_name) or types.ModuleType(parent_name)
    callbacks = sys.modules.get(callbacks_name) or types.ModuleType(callbacks_name)
    module = sys.modules.get(module_name) or types.ModuleType(module_name)
    model_checkpoint = getattr(module, "ModelCheckpoint", None)
    if model_checkpoint is None:
        model_checkpoint = type("ModelCheckpoint", (), {})
        model_checkpoint.__module__ = module_name
        module.ModelCheckpoint = model_checkpoint
    parent.callbacks = callbacks
    callbacks.model_checkpoint = module
    sys.modules[parent_name] = parent
    sys.modules[callbacks_name] = callbacks
    sys.modules[module_name] = module
    with torch.serialization.safe_globals([model_checkpoint]):
        return torch.load(
            checkpoint,
            map_location=device,
            weights_only=True,
        )


class OfficialLSegBackend:
    """Feature encoder backed by ``vlmaps.lseg`` from the official repository."""

    name = "official_vlmaps_lseg"

    def __init__(
        self,
        vlmaps_root: str | Path = "/opt/vlmaps",
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
    ):
        root = Path(vlmaps_root).resolve()
        if not (root / "vlmaps" / "lseg").is_dir():
            raise RuntimeError(
                f"official VLMaps checkout not found at {root}; clone "
                "https://github.com/vlmaps/vlmaps"
            )
        sys.path.insert(0, str(root))

        import clip
        import gdown
        import torch
        import torchvision.transforms as transforms
        from vlmaps.lseg.modules.models.lseg_net import LSegEncNet
        from vlmaps.utils.clip_utils import multiple_templates
        from vlmaps.utils.lseg_utils import get_lseg_feat

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("VLMaps device must be auto, cpu, or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("VLMaps requested CUDA but CUDA is unavailable")

        if checkpoint_path is None:
            checkpoint_path = os.environ.get(
                "VLMAPS_CHECKPOINT",
                "/cache/vlmaps/demo_e200.ckpt",
            )
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.is_file():
            print(f"[VLMaps] downloading official LSeg checkpoint to {checkpoint}")
            downloaded = gdown.download(
                OFFICIAL_LSEG_CHECKPOINT_URL,
                output=str(checkpoint),
                quiet=False,
            )
            if not downloaded or not checkpoint.is_file():
                raise RuntimeError("failed to download the official LSeg checkpoint")

        crop_size = 480
        model = LSegEncNet(
            "",
            arch_option=0,
            block_depth=0,
            activation="lrelu",
            crop_size=crop_size,
        )
        checkpoint_data = _load_official_checkpoint(
            torch,
            checkpoint,
            device,
        )
        raw_state = checkpoint_data["state_dict"]
        state = {
            (key[4:] if key.startswith("net.") else key): value
            for key, value in raw_state.items()
        }
        model.load_state_dict(state)
        model.eval().to(device)

        self._torch = torch
        self._clip = clip
        self._get_lseg_feat = get_lseg_feat
        self._multiple_templates = list(multiple_templates)
        self._model = model
        self._device = device
        self._crop_size = crop_size
        self._base_size = 520
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.feature_dim = int(model.out_c)
        print(
            f"[VLMaps] official LSeg ready device={device} "
            f"feature_dim={self.feature_dim}"
        )

    def encode_image(self, rgb: np.ndarray) -> np.ndarray:
        """Return normalized ``H x W x C`` dense LSeg features."""
        features = self._get_lseg_feat(
            self._model,
            np.asarray(rgb, dtype=np.uint8).copy(),
            ["example"],
            self._transform,
            self._device,
            crop_size=self._crop_size,
            base_size=self._base_size,
            norm_mean=[0.5, 0.5, 0.5],
            norm_std=[0.5, 0.5, 0.5],
        )[0]
        features = np.moveaxis(features, 0, -1).astype(np.float32, copy=False)
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        return features / np.maximum(norms, 1e-12)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text with VLMaps' official multi-template CLIP prompts."""
        return self.encode_texts([text])[0]

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 256,
    ) -> np.ndarray:
        """Batch VLMaps' multi-template embeddings for several categories."""
        prompts = [
            template.format(text)
            for text in texts
            for template in self._multiple_templates
        ]
        encoded_batches = []
        for start in range(0, len(prompts), batch_size):
            tokens = self._clip.tokenize(
                prompts[start:start + batch_size]
            ).to(self._device)
            with self._torch.no_grad():
                features = self._model.clip_pretrained.encode_text(tokens).float()
            features /= features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            encoded_batches.append(features.cpu())
        features = self._torch.cat(encoded_batches, dim=0)
        features = features.reshape(
            len(texts),
            len(self._multiple_templates),
            -1,
        ).mean(dim=1)
        return features.numpy().astype(np.float32, copy=False)
