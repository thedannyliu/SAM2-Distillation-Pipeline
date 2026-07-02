#!/usr/bin/env python3
"""Probe TinyViT timm feature metadata for EdgeTAM config generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sam2_distill.edgetam.config import TinyViTEdgeTAMConfig, feature_indices, probe_timm_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=TinyViTEdgeTAMConfig.model_name)
    parser.add_argument("--features", nargs="+", default=list(TinyViTEdgeTAMConfig.features))
    parser.add_argument("--forward-size", type=int, default=0, help="Optional square dummy forward size; 0 skips forward.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-probe", action="store_true", help="Instantiate timm model even when known metadata exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = tuple(args.features)
    report = probe_timm_backbone(args.model_name, features, force_probe=args.force_probe)

    if args.forward_size > 0:
        import timm

        model = timm.create_model(
            args.model_name,
            pretrained=False,
            in_chans=3,
            features_only=True,
            out_indices=feature_indices(features),
        ).to(args.device)
        model.eval()
        x = torch.randn(1, 3, args.forward_size, args.forward_size, device=args.device)
        with torch.inference_mode():
            ys = model(x)
        report["forward_size"] = args.forward_size
        report["forward_shapes"] = [list(y.shape) for y in ys]

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
