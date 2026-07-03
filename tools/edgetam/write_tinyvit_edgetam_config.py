#!/usr/bin/env python3
"""Write an EdgeTAM TinyViT YAML config from timm feature metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sam2_distill.edgetam.config import (
    TinyViTEdgeTAMConfig,
    probe_timm_backbone,
    write_edgetam_tinyvit_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", default=TinyViTEdgeTAMConfig.model_name)
    parser.add_argument(
        "--template",
        type=Path,
        help="Optional full trainer YAML to clone while replacing TinyViT trunk metadata.",
    )
    parser.add_argument("--force-probe", action="store_true", help="Instantiate timm model even when known metadata exists.")
    return parser.parse_args()


def write_full_trainer_config(
    out: Path,
    template: Path,
    cfg: TinyViTEdgeTAMConfig,
    force_probe: bool,
) -> dict[str, object]:
    import yaml

    probe = probe_timm_backbone(cfg.model_name, cfg.features, force_probe=force_probe)
    trainer_cfg = yaml.safe_load(template.read_text(encoding="utf-8"))
    image_encoder = trainer_cfg["trainer"]["model"]["image_encoder"]
    image_encoder["trunk"]["name"] = cfg.model_name
    image_encoder["trunk"]["features"] = list(cfg.features)
    image_encoder["neck"]["backbone_channel_list"] = probe["backbone_channel_list"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(trainer_cfg, sort_keys=False), encoding="utf-8")
    probe["template"] = str(template)
    probe["config_type"] = "full_trainer"
    return probe


def main() -> None:
    args = parse_args()
    cfg = TinyViTEdgeTAMConfig(model_name=args.model_name)
    out = Path(args.out)
    if args.template is None:
        probe = write_edgetam_tinyvit_yaml(out, cfg=cfg, force_probe=args.force_probe)
        probe["config_type"] = "model_only"
    else:
        probe = write_full_trainer_config(out, args.template, cfg=cfg, force_probe=args.force_probe)
    print(json.dumps({"wrote": str(Path(args.out).resolve()), "probe": probe}, indent=2))


if __name__ == "__main__":
    main()
