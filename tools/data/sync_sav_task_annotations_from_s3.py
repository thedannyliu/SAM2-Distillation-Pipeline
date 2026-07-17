#!/usr/bin/env python3
"""Sync manifest-selected SA-V manual annotations from the company Data Lake."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import pandas as pd


EXPECTED_MANIFEST_VIDEOS = 50_453
EXPECTED_SELECTED_VIDEOS = 50_337
EXPECTED_MISSING_VIDEOS = 116


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="sdp-ril")
    parser.add_argument(
        "--prefix",
        default="danny-dataset/SA-V/sav_train/",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--json-samples", type=int, default=200)
    return parser.parse_args()


def load_manifest_videos(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    frame = pd.read_parquet(path, columns=["video_id", "split"])
    videos = set(frame.loc[frame["split"] == "train", "video_id"].astype(str))
    if len(videos) != EXPECTED_MANIFEST_VIDEOS:
        raise RuntimeError(
            f"Manifest has {len(videos)} train videos; "
            f"expected {EXPECTED_MANIFEST_VIDEOS}"
        )
    return videos


def list_manual_annotations(client, bucket: str, prefix: str) -> dict[str, tuple[str, int]]:
    annotations: dict[str, tuple[str, int]] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            name = PurePosixPath(key).name
            if not name.endswith("_manual.json"):
                continue
            video_id = name.removesuffix("_manual.json")
            if video_id in annotations:
                raise RuntimeError(f"Duplicate manual annotation for {video_id}")
            annotations[video_id] = (key, int(item["Size"]))
    return annotations


def local_path(key: str, prefix: str, train_root: Path) -> Path:
    if not key.startswith(prefix):
        raise ValueError(f"Object is outside prefix: {key}")
    relative = PurePosixPath(key.removeprefix(prefix))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe object key: {key}")
    return train_root.joinpath(*relative.parts)


def sync_one(client, bucket: str, key: str, size: int, target: Path) -> str:
    if target.is_file() and target.stat().st_size == size:
        return "skipped"

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            body = client.get_object(Bucket=bucket, Key=key)["Body"]
            try:
                for chunk in iter(lambda: body.read(8 * 1024 * 1024), b""):
                    output.write(chunk)
            finally:
                body.close()
        if temporary.stat().st_size != size:
            raise IOError(
                f"Size mismatch for s3://{bucket}/{key}: "
                f"got {temporary.stat().st_size}, expected {size}"
            )
        temporary.chmod(0o660)
        temporary.replace(target)
        return "downloaded"
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def check_json_sample(paths: list[Path], sample_count: int) -> int:
    if sample_count <= 0:
        return 0
    selected = random.Random(310107256).sample(
        paths,
        min(sample_count, len(paths)),
    )

    def load(path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)

    with ThreadPoolExecutor(max_workers=min(8, len(selected))) as executor:
        list(executor.map(load, selected))
    return len(selected)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    os.umask(0o007)
    prefix = args.prefix.strip("/") + "/"

    import boto3
    from botocore.config import Config

    endpoint = os.environ.get("S3_ENDPOINT_URL") or os.environ.get(
        "AWS_ENDPOINT_URL"
    )
    client = boto3.Session().client(
        "s3",
        endpoint_url=endpoint,
        config=Config(
            max_pool_connections=max(args.workers, 10),
            retries={"max_attempts": 10, "mode": "adaptive"},
        ),
    )
    manifest_videos = load_manifest_videos(args.manifest)
    print(f"Endpoint: {endpoint or 'AWS default S3 endpoint'}", flush=True)
    print(f"Source: s3://{args.bucket}/{prefix}", flush=True)

    remote = list_manual_annotations(client, args.bucket, prefix)
    selected = sorted(manifest_videos & set(remote))
    missing = sorted(manifest_videos - set(remote))
    if len(selected) != EXPECTED_SELECTED_VIDEOS:
        raise RuntimeError(
            f"Found {len(selected)} required manual annotations; "
            f"expected {EXPECTED_SELECTED_VIDEOS}"
        )
    if len(missing) != EXPECTED_MISSING_VIDEOS:
        raise RuntimeError(
            f"Missing {len(missing)} manifest annotations; "
            f"expected {EXPECTED_MISSING_VIDEOS}; examples: {missing[:10]}"
        )

    train_root = args.out_root / "sav_train"
    train_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (video_id, *remote[video_id], local_path(remote[video_id][0], prefix, train_root))
        for video_id in selected
    ]
    counts = {"downloaded": 0, "skipped": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(sync_one, client, args.bucket, key, size, target): video_id
            for video_id, key, size, target in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            counts[future.result()] += 1
            if completed % 1000 == 0 or completed == len(futures):
                print(
                    f"sync: {completed}/{len(futures)} "
                    f"downloaded={counts['downloaded']} "
                    f"skipped={counts['skipped']}",
                    flush=True,
                )

    invalid = [
        str(target)
        for _, _, size, target in jobs
        if not target.is_file() or target.stat().st_size != size
    ]
    if invalid:
        raise RuntimeError(f"Local size verification failed: {invalid[:10]}")
    json_checked = check_json_sample(
        [target for _, _, _, target in jobs],
        args.json_samples,
    )

    summary = {
        "status": "pass",
        "source": f"s3://{args.bucket}/{prefix}",
        "manifest": str(args.manifest),
        "sav_train_root": str(train_root),
        "manifest_train_videos": len(manifest_videos),
        "selected_manual_annotations": len(selected),
        "missing_manual_annotations": len(missing),
        "missing_examples": missing[:10],
        "downloaded": counts["downloaded"],
        "already_present": counts["skipped"],
        "bytes": sum(size for _, _, size, _ in jobs),
        "json_files_checked": json_checked,
    }
    summary_path = args.out_root / "sav_train_manual_annotations.provenance.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Provenance: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
