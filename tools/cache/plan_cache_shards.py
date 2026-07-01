#!/usr/bin/env python3
"""Print shard assignments for SAM2 teacher cache jobs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pyarrow.parquet as pq


def read_rows(path: Path) -> int:
    if path.suffix == ".parquet":
        return pq.ParquetFile(path).metadata.num_rows
    if path.suffix == ".csv":
        return sum(1 for _ in path.open("r", encoding="utf-8")) - 1
    raise ValueError("manifest must be .parquet or .csv")


def format_range(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}-{end}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--num-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.manifest).expanduser().resolve())
    total_shards = math.ceil(rows / args.shard_size)
    shards_per_job = math.ceil(total_shards / args.num_jobs)

    print(f"rows={rows}")
    print(f"shard_size={args.shard_size}")
    print(f"total_shards={total_shards}")
    print(f"num_jobs={args.num_jobs}")

    for job_id in range(args.num_jobs):
        start = job_id * shards_per_job
        end = min(total_shards, start + shards_per_job) - 1
        if start > end:
            break
        print(f"job={job_id} shard_ids={format_range(start, end)}")


if __name__ == "__main__":
    main()
