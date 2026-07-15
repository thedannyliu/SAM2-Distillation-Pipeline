#!/usr/bin/env python3
"""Download and validate pinned ImageNet-pretrained RepViT checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


MODELS = (
    {
        "name": "repvit_m0_9.dist_450e_in1k",
        "repo_id": "timm/repvit_m0_9.dist_450e_in1k",
        "revision": "003653b800490792c19a0c292b663b8799804ef6",
        "bytes": 22_232_904,
        "sha256": "c2dd32622df856ea7a68d6b241c1623f0e6fbec8acde381a3519cb5a5463e9cf",
        "channels": [48, 96, 192, 384],
        "spatial_sizes": [56, 28, 14, 7],
    },
    {
        "name": "repvit_m2_3.dist_450e_in1k",
        "repo_id": "timm/repvit_m2_3.dist_450e_in1k",
        "revision": "07560b72ade535b66673f9c022214acf2141fd5b",
        "bytes": 95_542_216,
        "sha256": "14ae2ea276682bb33dfa8313abdca16ac8f8ee7cb76e95f938c431f39afd8d6f",
        "channels": [80, 160, 320, 640],
        "spatial_sizes": [56, 28, 14, 7],
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "/group-volume/danny-dataset/sam2_distill/checkpoints/repvit"
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-timm-smoke",
        action="store_true",
        help="Verify files but do not instantiate the timm feature backbone.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    repo_id: str,
    revision: str,
    filename: str,
    destination: Path,
    force: bool,
) -> None:
    if destination.is_file() and not force:
        print(f"exists: {destination}", flush=True)
        return

    from huggingface_hub import hf_hub_download

    print(
        f"download: hf://{repo_id}/{filename}@{revision}",
        flush=True,
    )
    source = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def validate_safetensors(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    from safetensors import safe_open

    actual_bytes = path.stat().st_size
    actual_sha256 = sha256(path)
    if actual_bytes != spec["bytes"]:
        raise RuntimeError(
            f"Size mismatch for {path}: got {actual_bytes}, expected {spec['bytes']}"
        )
    if actual_sha256 != spec["sha256"]:
        raise RuntimeError(
            f"SHA256 mismatch for {path}: got {actual_sha256}, "
            f"expected {spec['sha256']}"
        )
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor_keys = list(handle.keys())
    if not tensor_keys:
        raise RuntimeError(f"No tensors found in {path}")
    return {
        "checkpoint": str(path),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "num_tensors": len(tensor_keys),
    }


def smoke_timm_model(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    try:
        import timm
        import torch
    except ImportError as error:
        raise RuntimeError(
            "RepViT smoke requires the existing requirements-stage1.txt packages; "
            "do not replace the container PyTorch runtime."
        ) from error

    model_name = str(spec["name"])
    try:
        base_model = timm.create_model(
            model_name,
            pretrained=False,
            checkpoint_path=str(path),
        ).eval()
        from timm.models._features import FeatureListNet

        model = FeatureListNet(base_model, out_indices=(0, 1, 2, 3)).eval()
    except RuntimeError as error:
        raise RuntimeError(
            f"timm {timm.__version__} could not instantiate or load {model_name}; "
            "verify the pinned checkpoint and the container timm package."
        ) from error
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    with torch.inference_mode():
        features = model(torch.zeros(1, 3, 224, 224))
    shapes = [list(feature.shape) for feature in features]
    channels = [shape[1] for shape in shapes]
    spatial_sizes = [shape[-1] for shape in shapes]
    if channels != spec["channels"] or spatial_sizes != spec["spatial_sizes"]:
        raise RuntimeError(
            f"Feature contract mismatch for {model_name}: shapes={shapes}, "
            f"expected channels={spec['channels']} and sizes={spec['spatial_sizes']}"
        )
    return {
        "timm_version": timm.__version__,
        "parameter_count": parameter_count,
        "feature_shapes_224": shapes,
    }


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for spec in MODELS:
        stem = str(spec["name"])
        checkpoint = args.out_root / f"{stem}.safetensors"
        config = args.out_root / f"{stem}.config.json"
        download_file(
            str(spec["repo_id"]),
            str(spec["revision"]),
            "model.safetensors",
            checkpoint,
            args.force,
        )
        download_file(
            str(spec["repo_id"]),
            str(spec["revision"]),
            "config.json",
            config,
            args.force,
        )
        summary = {
            "status": "pass",
            "model_name": stem,
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "config": str(config),
            **validate_safetensors(checkpoint, spec),
        }
        if args.skip_timm_smoke:
            summary["timm_smoke"] = "skipped"
        else:
            summary.update(smoke_timm_model(checkpoint, spec))
            summary["timm_smoke"] = "pass"
        summary_path = checkpoint.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    aggregate = args.out_root / "repvit_pretrained.summary.json"
    aggregate.write_text(
        json.dumps({"status": "pass", "models": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    checksums = args.out_root / "SHA256SUMS.txt"
    checksums.write_text(
        "".join(
            f"{summary['sha256']}  {Path(summary['checkpoint']).name}\n"
            for summary in summaries
        ),
        encoding="utf-8",
    )
    print(f"summary: {aggregate}")
    print(f"checksums: {checksums}")


if __name__ == "__main__":
    main()
