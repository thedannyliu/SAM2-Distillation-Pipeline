#!/usr/bin/env python3
"""Write an EdgeTAM TinyViT YAML config from timm feature metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sam2_distill.edgetam.config import TinyViTEdgeTAMConfig, write_edgetam_tinyvit_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", default=TinyViTEdgeTAMConfig.model_name)
    parser.add_argument("--force-probe", action="store_true", help="Instantiate timm model even when known metadata exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TinyViTEdgeTAMConfig(model_name=args.model_name)
    probe = write_edgetam_tinyvit_yaml(Path(args.out), cfg=cfg, force_probe=args.force_probe)
    print(json.dumps({"wrote": str(Path(args.out).resolve()), "probe": probe}, indent=2))


if __name__ == "__main__":
    main()
