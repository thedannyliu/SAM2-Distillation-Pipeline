#!/usr/bin/env python3
"""Sync mounted SA-V runtime data from the company S3-compatible Data Lake."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath


COMPONENTS = ("JPEGImages", "sav_val", "sav_test")


class DownloadSizeError(IOError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="sdp-ril")
    parser.add_argument("--source-root", default="danny-dataset/SA-V")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/group-volume/danny-dataset/SA-V"),
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--file-retries", type=int, default=8)
    parser.add_argument(
        "--components",
        nargs="+",
        choices=COMPONENTS,
        default=list(COMPONENTS),
    )
    parser.add_argument("--reserve-gib", type=float, default=5.0)
    return parser.parse_args()


def iter_objects(client, bucket: str, prefix: str):
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            if key.endswith("/"):
                continue
            yield key, int(item["Size"])


def target_path(key: str, source_root: str, out_root: Path) -> Path:
    prefix = source_root.strip("/") + "/"
    if not key.startswith(prefix):
        raise ValueError(f"Object is outside source root: {key}")
    relative = PurePosixPath(key.removeprefix(prefix))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe object key: {key}")
    return out_root.joinpath(*relative.parts)


def is_current(path: Path, size: int) -> bool:
    return path.is_file() and path.stat().st_size == size


def download_once(client, bucket: str, key: str, size: int, target: Path) -> None:
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
        actual_size = temporary.stat().st_size
        if actual_size != size:
            raise DownloadSizeError(
                f"Size mismatch for s3://{bucket}/{key}: "
                f"got {actual_size}, expected {size}"
            )
        temporary.chmod(0o660)
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def download_one(
    client,
    bucket: str,
    key: str,
    size: int,
    target: Path,
    file_retries: int,
) -> str:
    if is_current(target, size):
        return "skipped"

    from botocore.exceptions import (
        ConnectionClosedError,
        ConnectTimeoutError,
        EndpointConnectionError,
        ReadTimeoutError,
        ResponseStreamingError,
    )

    retryable = (
        ConnectionClosedError,
        ConnectTimeoutError,
        EndpointConnectionError,
        ReadTimeoutError,
        ResponseStreamingError,
        DownloadSizeError,
    )
    for attempt in range(file_retries + 1):
        try:
            download_once(client, bucket, key, size, target)
            return "downloaded"
        except retryable as error:
            if attempt >= file_retries:
                raise
            delay = min(30.0, 2**attempt) + random.random()
            print(
                f"retry {attempt + 1}/{file_retries} "
                f"s3://{bucket}/{key} after {type(error).__name__}",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def inventory(client, args: argparse.Namespace) -> dict[str, dict[str, int]]:
    result = {}
    for component in args.components:
        prefix = f"{args.source_root.strip('/')}/{component}/"
        counts = {
            "objects": 0,
            "bytes": 0,
            "missing_objects": 0,
            "missing_bytes": 0,
        }
        for key, size in iter_objects(client, args.bucket, prefix):
            counts["objects"] += 1
            counts["bytes"] += size
            if not is_current(target_path(key, args.source_root, args.out_root), size):
                counts["missing_objects"] += 1
                counts["missing_bytes"] += size
        if counts["objects"] == 0:
            raise RuntimeError(f"No objects found at s3://{args.bucket}/{prefix}")
        result[component] = counts
        print(f"inventory {component}: {counts}", flush=True)
    return result


def sync_component(
    client,
    args: argparse.Namespace,
    component: str,
) -> dict[str, int]:
    prefix = f"{args.source_root.strip('/')}/{component}/"
    counts = {"checked": 0, "downloaded": 0, "skipped": 0}
    page = []
    executor = ThreadPoolExecutor(max_workers=args.workers)
    try:
        for key, size in iter_objects(client, args.bucket, prefix):
            page.append((key, size, target_path(key, args.source_root, args.out_root)))
            if len(page) < 1000:
                continue
            _sync_page(
                executor,
                client,
                args.bucket,
                page,
                counts,
                component,
                args.file_retries,
            )
            page = []
        if page:
            _sync_page(
                executor,
                client,
                args.bucket,
                page,
                counts,
                component,
                args.file_retries,
            )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    print(
        f"sync {component}: checked={counts['checked']} "
        f"downloaded={counts['downloaded']} skipped={counts['skipped']}",
        flush=True,
    )
    return counts


def _sync_page(
    executor,
    client,
    bucket,
    page,
    counts,
    component,
    file_retries,
) -> None:
    futures = {
        executor.submit(
            download_one,
            client,
            bucket,
            key,
            size,
            target,
            file_retries,
        ): key
        for key, size, target in page
    }
    for future in as_completed(futures):
        counts[future.result()] += 1
        counts["checked"] += 1
    if counts["checked"] % 5000 == 0:
        print(
            f"sync {component}: checked={counts['checked']} "
            f"downloaded={counts['downloaded']} skipped={counts['skipped']}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.file_retries < 0:
        raise SystemExit("--file-retries cannot be negative")
    os.umask(0o007)
    args.out_root.mkdir(parents=True, exist_ok=True)

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
    print(f"Endpoint: {endpoint or 'AWS default S3 endpoint'}", flush=True)
    print(f"Destination: {args.out_root}", flush=True)
    source_inventory = inventory(client, args)
    missing_bytes = sum(item["missing_bytes"] for item in source_inventory.values())
    free_bytes = shutil.disk_usage(args.out_root).free
    reserve_bytes = int(args.reserve_gib * 1024**3)
    print(f"Missing bytes: {missing_bytes}", flush=True)
    print(f"Filesystem free bytes: {free_bytes}", flush=True)
    if missing_bytes + reserve_bytes > free_bytes:
        raise RuntimeError(
            f"Insufficient filesystem capacity: need {missing_bytes + reserve_bytes}, "
            f"available {free_bytes}"
        )

    sync_results = {
        component: sync_component(client, args, component)
        for component in args.components
    }
    summary = {
        "status": "pass",
        "source": f"s3://{args.bucket}/{args.source_root.strip('/')}/",
        "destination": str(args.out_root),
        "components": args.components,
        "inventory": source_inventory,
        "sync": sync_results,
    }
    summary_path = args.out_root / "runtime_data_sync.provenance.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Provenance: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
