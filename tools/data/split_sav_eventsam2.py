#!/usr/bin/env python3
"""Create frozen, video-disjoint SA-V train roles for EventSAM2 experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROLE_COUNTS = {
    "route_train": 40_270,
    "gate_train": 5_034,
    "selection": 2_517,
    "calibration": 2_516,
}


def ranked_video_ids(video_ids: list[str], seed: str) -> list[str]:
    unique = sorted(set(video_ids))
    return sorted(
        unique,
        key=lambda video_id: (
            hashlib.sha256(f"{seed}|{video_id}".encode("utf-8")).digest(),
            video_id,
        ),
    )


def assign_roles(
    video_ids: list[str], counts: dict[str, int], seed: str
) -> dict[str, list[str]]:
    ranked = ranked_video_ids(video_ids, seed)
    if sum(counts.values()) != len(ranked):
        raise ValueError(
            f"Role counts sum to {sum(counts.values()):,}, but received "
            f"{len(ranked):,} unique videos"
        )
    roles: dict[str, list[str]] = {}
    start = 0
    for role, count in counts.items():
        roles[role] = sorted(ranked[start : start + count])
        start += count
    return roles


def parse_counts(value: str | None, video_count: int) -> dict[str, int]:
    if value is None:
        if video_count != sum(ROLE_COUNTS.values()):
            raise ValueError(
                f"Default EventSAM2 split expects {sum(ROLE_COUNTS.values()):,} "
                f"usable videos, found {video_count:,}; pass --role-counts explicitly"
            )
        return ROLE_COUNTS.copy()
    fields = value.split(",")
    counts = {}
    for field in fields:
        role, count = field.split("=", 1)
        counts[role.strip()] = int(count)
    if tuple(counts) != tuple(ROLE_COUNTS):
        raise ValueError(
            "--role-counts must list route_train,gate_train,selection,calibration in order"
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sav-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", default="eventsam2_sav_v1")
    parser.add_argument(
        "--role-counts",
        help="Explicit role counts, e.g. route_train=8,gate_train=1,selection=1,calibration=1",
    )
    parser.add_argument(
        "--selection-screen-videos",
        type=int,
        default=32,
        help="Write the first N ranked selection videos as selection_screen.txt",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.manifest, columns=["video_id", "split"])
    candidates = sorted(
        set(frame.loc[frame["split"] == "train", "video_id"].astype(str))
    )
    usable = []
    for video_id in candidates:
        number = int(video_id.rsplit("_", 1)[-1])
        annotation = (
            args.sav_root
            / "sav_train"
            / f"sav_{number // 1000:03d}"
            / f"{video_id}_manual.json"
        )
        if annotation.is_file():
            usable.append(video_id)

    counts = parse_counts(args.role_counts, len(usable))
    roles = assign_roles(usable, counts, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for role, video_ids in roles.items():
        (args.out_dir / f"{role}.txt").write_text(
            "".join(f"{video_id}\n" for video_id in video_ids), encoding="utf-8"
        )
    screen_count = min(args.selection_screen_videos, len(roles["selection"]))
    screen = ranked_video_ids(roles["selection"], args.seed + "|screen")[:screen_count]
    (args.out_dir / "selection_screen.txt").write_text(
        "".join(f"{video_id}\n" for video_id in screen), encoding="utf-8"
    )

    summary = {
        "seed": args.seed,
        "manifest": str(args.manifest),
        "sav_root": str(args.sav_root),
        "manifest_train_videos": len(candidates),
        "usable_manual_annotation_videos": len(usable),
        "excluded_videos": len(candidates) - len(usable),
        "roles": {role: len(video_ids) for role, video_ids in roles.items()},
        "selection_screen_videos": screen_count,
        "pairwise_overlap": {
            f"{left}__{right}": len(set(roles[left]) & set(roles[right]))
            for index, left in enumerate(roles)
            for right in list(roles)[index + 1 :]
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
