#!/usr/bin/env python3
"""Issue and validate evidence gates for SAM2.1-L two-clock experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sam2_distill.two_clock.verification import (  # noqa: E402
    load_clip_traces,
    validate_optimizer_audit,
)


DEFAULT_CONTRACT = (
    REPO_ROOT / "configs/verification/sam21l_two_clock_reuse_v1.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def git_identity(repo: Path) -> dict[str, Any]:
    status = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    return {
        "path": str(repo.resolve()),
        "commit": git_output(repo, "rev-parse", "HEAD"),
        "branch": git_output(repo, "branch", "--show-current"),
        "clean": not bool(status),
        "dirty_entries": status.splitlines(),
    }


def validate_contract(contract: dict[str, Any], repo_root: Path) -> None:
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported verification contract schema")
    identifiers = []
    for requirement in contract.get("requirements", []):
        identifiers.append(requirement.get("id"))
        for field in ("statement", "code", "test", "runtime_evidence"):
            if not requirement.get(field):
                raise ValueError(
                    f"requirement {requirement.get('id')} has no {field}"
                )
        for field in ("code", "test"):
            reference = requirement[field].split("::", maxsplit=1)
            path = repo_root / reference[0]
            if not path.is_file():
                raise ValueError(
                    f"requirement {requirement['id']} references missing {path}"
                )
            if len(reference) != 2:
                raise ValueError(
                    f"requirement {requirement['id']} has no symbol in {field}"
                )
            symbol = reference[1].split(".")[-1]
            if symbol not in path.read_text(encoding="utf-8"):
                raise ValueError(
                    f"requirement {requirement['id']} references missing "
                    f"symbol {symbol} in {path}"
                )
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("requirement IDs must be non-empty and unique")


def local_identity(
    repo_root: Path, contract_path: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    repository = git_identity(repo_root)
    design = repo_root / contract["design_path"]
    config = repo_root / contract["training_config_path"]
    for path in (contract_path, design, config):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "repository": repository,
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": sha256_file(contract_path),
        },
        "design": {"path": str(design.resolve()), "sha256": sha256_file(design)},
        "training_config": {
            "path": str(config.resolve()),
            "sha256": sha256_file(config),
        },
    }


def full_identity(args: argparse.Namespace, contract: dict[str, Any]) -> dict[str, Any]:
    identity = local_identity(args.repo_root, args.contract, contract)
    paths = {
        "dataset_manifest": args.manifest,
        "sam2_config": args.sam2_config,
        "initializer_checkpoint": args.checkpoint,
    }
    identity["inputs"] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        identity["inputs"][name] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    identity["sam2_repository"] = git_identity(args.sam2_root)
    return identity


def require_clean(identity: dict[str, Any]) -> None:
    if not identity["repository"]["clean"]:
        raise RuntimeError("verification requires a clean experiment repository")
    sam2 = identity.get("sam2_repository")
    if sam2 is not None and not sam2["clean"]:
        raise RuntimeError("verification requires a clean official SAM2 repository")


def same_identity(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    if expected != actual:
        raise RuntimeError("verification identity does not match the current inputs")


def junit_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    keys = ("tests", "failures", "errors", "skipped")
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in keys
    }


def issue_local_stamp(args: argparse.Namespace, contract: dict[str, Any]) -> None:
    validate_contract(contract, args.repo_root)
    identity = local_identity(args.repo_root, args.contract, contract)
    require_clean(identity)
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    junit = args.evidence_dir / "local_pytest.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        *contract["local_tests"],
        f"--junitxml={junit}",
    ]
    print("LOCAL VERIFICATION:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=args.repo_root, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"local verification tests failed with {result.returncode}")
    summary = junit_summary(junit)
    if summary["failures"] or summary["errors"] or summary["tests"] == 0:
        raise RuntimeError(f"invalid local test result: {summary}")
    payload = {
        "schema_version": 1,
        "status": "LOCAL_LOGIC_VERIFIED",
        "issued_at_unix": time.time(),
        "identity": identity,
        "requirements": [row["id"] for row in contract["requirements"]],
        "tests": {"command": command, **summary},
    }
    write_json(args.evidence_dir / "local_stamp.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def validate_local_stamp(
    evidence_dir: Path, current_identity: dict[str, Any]
) -> dict[str, Any]:
    stamp = load_json(evidence_dir / "local_stamp.json")
    if stamp.get("status") != "LOCAL_LOGIC_VERIFIED":
        raise RuntimeError("local verification stamp is not valid")
    same_identity(stamp["identity"], current_identity)
    return stamp


def validate_smoke_artifacts(smoke_dir: Path, world_size: int) -> dict[str, Any]:
    required = [
        smoke_dir / "checkpoints/checkpoint.pt",
        smoke_dir / "checkpoints/last.pt",
        smoke_dir / "resolved_config.yaml",
        smoke_dir / "training_status.json",
        smoke_dir / "gradient_diagnostics.json",
        smoke_dir / "optimizer_audit.json",
    ]
    required.extend(smoke_dir / f"capacity_rank{rank}.json" for rank in range(world_size))
    required.extend(
        smoke_dir / f"mechanism_gradients_rank{rank}.json"
        for rank in range(world_size)
    )
    trace_paths = [
        smoke_dir / f"flight_recorder_rank{rank}.jsonl"
        for rank in range(world_size)
    ]
    required.extend(trace_paths)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"smoke artifacts are missing: {missing}")

    training = load_json(smoke_dir / "training_status.json")
    gradients = load_json(smoke_dir / "gradient_diagnostics.json")
    optimizer = load_json(smoke_dir / "optimizer_audit.json")
    if training.get("status") != "complete":
        raise RuntimeError(f"smoke training did not complete: {training}")
    if gradients.get("status") != "pass" or gradients.get("nonfinite_steps") != 0:
        raise RuntimeError(f"smoke gradients failed: {gradients}")
    validate_optimizer_audit(optimizer)

    capacities = [
        load_json(smoke_dir / f"capacity_rank{rank}.json")
        for rank in range(world_size)
    ]
    if any(
        row.get("world_size") != world_size or row.get("per_gpu_batch") != 1
        for row in capacities
    ):
        raise RuntimeError(f"smoke capacity contract failed: {capacities}")

    mechanism_rows = [
        load_json(smoke_dir / f"mechanism_gradients_rank{rank}.json")
        for rank in range(world_size)
    ]
    if any(row.get("status") != "pass" for row in mechanism_rows):
        raise RuntimeError(f"mechanism gradient diagnostics failed: {mechanism_rows}")
    group_names = set().union(
        *(row.get("groups", {}).keys() for row in mechanism_rows)
    )
    expected_groups = {"age_conditioner", "memory_time_conditioner"}
    if group_names != expected_groups:
        raise RuntimeError(
            f"E smoke mechanism groups are {sorted(group_names)}, "
            f"expected {sorted(expected_groups)}"
        )
    for name in group_names:
        if not any(
            row.get("groups", {}).get(name, {}).get(
                "steps_with_nonzero_gradient", 0
            )
            > 0
            for row in mechanism_rows
        ):
            raise RuntimeError(f"mechanism group never received a gradient: {name}")

    traces = load_clip_traces(trace_paths)
    if any(trace.get("experiment") != "E" for trace in traces):
        raise RuntimeError("remote smoke contains a non-E flight trace")
    if not any(
        frame["action"] == "REUSE"
        for trace in traces
        for frame in trace["frames"]
    ):
        raise RuntimeError("remote smoke did not exercise a reuse frame")
    return {
        "training": training,
        "gradients": gradients,
        "optimizer": optimizer,
        "capacity": capacities,
        "mechanism_gradients": mechanism_rows,
        "flight_traces": len(traces),
    }


def launch_manifest_path(run_dir: Path) -> Path:
    return run_dir / "verification/launch_manifest.json"


def legacy_artifacts(run_dir: Path) -> list[Path]:
    markers = (
        run_dir / "resolved_config.yaml",
        run_dir / "training_status.json",
        run_dir / "checkpoints/checkpoint.pt",
        run_dir / "checkpoints/last.pt",
    )
    return [path for path in markers if path.exists()]


def parse_settings(rows: list[str]) -> dict[str, str]:
    settings = {}
    for row in rows:
        if "=" not in row:
            raise ValueError(f"invalid runtime setting: {row}")
        key, value = row.split("=", maxsplit=1)
        if not key or key in settings:
            raise ValueError(f"duplicate or empty runtime setting: {row}")
        settings[key] = value
    return settings


def record_launch(args: argparse.Namespace, contract: dict[str, Any]) -> None:
    if args.scope == "smoke" and args.target != contract["remote_smoke_target"]:
        raise RuntimeError("smoke target does not match the verification contract")
    if args.scope == "formal" and args.target not in contract["formal_targets"]:
        raise RuntimeError("formal target is not registered in the contract")
    if len(args.gpus.split(",")) != contract["world_size"]:
        raise RuntimeError("launch GPU count does not match the verification contract")
    settings = parse_settings(args.setting)
    if args.scope == "formal" and settings != contract["formal_settings"]:
        raise RuntimeError(
            f"formal runtime settings differ from the contract: {settings}"
        )
    path = launch_manifest_path(args.run_dir)
    existing_legacy_artifacts = legacy_artifacts(args.run_dir)
    if args.scope == "formal" and not path.is_file() and existing_legacy_artifacts:
        raise RuntimeError(
            "pre-existing formal run has no prospective launch manifest; "
            "use audit-existing and a new RUN_ROOT instead: "
            f"{[str(path) for path in existing_legacy_artifacts]}"
        )
    identity = full_identity(args, contract)
    require_clean(identity)
    local = validate_local_stamp(args.evidence_dir, local_identity(args.repo_root, args.contract, contract))
    if args.scope == "formal":
        remote = load_json(args.evidence_dir / "remote_stamp.json")
        if remote.get("status") != "REMOTE_INTEGRATION_VERIFIED":
            raise RuntimeError("remote verification stamp is not valid")
        same_identity(remote["identity"], identity)
    else:
        remote = None
    payload = {
        "schema_version": 1,
        "status": "FORMAL_ELIGIBLE" if args.scope == "formal" else "SMOKE_ELIGIBLE",
        "issued_at_unix": time.time(),
        "scope": args.scope,
        "target": args.target,
        "gpus": args.gpus,
        "seed": args.seed,
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
        "runtime_settings": settings,
        "identity": identity,
        "local_stamp_issued_at_unix": local["issued_at_unix"],
        "remote_stamp_issued_at_unix": (
            remote["issued_at_unix"] if remote is not None else None
        ),
    }
    if path.is_file():
        existing = load_json(path)
        comparable = dict(existing)
        candidate = dict(payload)
        for field in (
            "issued_at_unix",
            "local_stamp_issued_at_unix",
            "remote_stamp_issued_at_unix",
        ):
            comparable.pop(field, None)
            candidate.pop(field, None)
        if comparable != candidate:
            raise RuntimeError("existing launch manifest does not match this launch")
        print(f"Launch manifest already matches: {path}", flush=True)
        return
    write_json(path, payload)
    print(json.dumps(payload, indent=2), flush=True)


def check_launch(args: argparse.Namespace, contract: dict[str, Any]) -> None:
    identity = full_identity(args, contract)
    require_clean(identity)
    launch = load_json(launch_manifest_path(args.run_dir))
    if (
        launch.get("status") != "FORMAL_ELIGIBLE"
        or launch.get("scope") != "formal"
        or launch.get("target") != args.target
    ):
        raise RuntimeError("formal launch manifest has the wrong status or target")
    same_identity(launch["identity"], identity)
    remote = load_json(args.evidence_dir / "remote_stamp.json")
    if remote.get("status") != "REMOTE_INTEGRATION_VERIFIED":
        raise RuntimeError("remote verification stamp is not valid")
    same_identity(remote["identity"], identity)
    print(f"Formal launch identity matches: {args.run_dir}", flush=True)


def issue_remote_stamp(args: argparse.Namespace, contract: dict[str, Any]) -> None:
    identity = full_identity(args, contract)
    require_clean(identity)
    validate_local_stamp(
        args.evidence_dir,
        local_identity(args.repo_root, args.contract, contract),
    )
    launch = load_json(launch_manifest_path(args.smoke_dir))
    if launch.get("scope") != "smoke" or launch.get("target") != contract["remote_smoke_target"]:
        raise RuntimeError("remote smoke launch manifest has the wrong scope or target")
    same_identity(launch["identity"], identity)
    input_audit = load_json(args.input_audit)
    if input_audit.get("status") != "pass":
        raise RuntimeError(f"input audit did not pass: {input_audit}")
    smoke = validate_smoke_artifacts(args.smoke_dir, contract["world_size"])
    payload = {
        "schema_version": 1,
        "status": "REMOTE_INTEGRATION_VERIFIED",
        "issued_at_unix": time.time(),
        "identity": identity,
        "input_audit": str(args.input_audit.resolve()),
        "smoke_dir": str(args.smoke_dir.resolve()),
        "smoke": smoke,
    }
    write_json(args.evidence_dir / "remote_stamp.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def artifact_inventory(run_dir: Path) -> list[dict[str, Any]]:
    patterns = (
        "resolved_config.yaml",
        "training_status.json",
        "training_model_summary.json",
        "gradient_diagnostics.json",
        "optimizer_audit.json",
        "wandb/wandb_run.json",
        "checkpoints/checkpoint*.pt",
        "checkpoints/last.pt",
        "checkpoints/best.pt",
        "best_selection.json",
        "val/selected/R*/sav_eval.json",
        "val/selected/R*/age_metrics.json",
    )
    paths = sorted({path for pattern in patterns for path in run_dir.glob(pattern)})
    return [
        {
            "path": str(path.relative_to(run_dir)),
            "size_bytes": path.stat().st_size,
            "mtime_unix": path.stat().st_mtime,
        }
        for path in paths
        if path.is_file()
    ]


def audit_existing(args: argparse.Namespace, contract: dict[str, Any]) -> None:
    inventory = artifact_inventory(args.run_dir)
    failures = []
    unavailable = ["launch_time_git_identity", "preflight_verification_stamp"]
    status_path = args.run_dir / "training_status.json"
    if status_path.is_file():
        training = load_json(status_path)
        if training.get("status") == "failed":
            failures.append("training_status.json reports failure")
    else:
        training = {"status": "running_or_not_started"}
        unavailable.append("completed_training_status")
    gradient_path = args.run_dir / "gradient_diagnostics.json"
    if gradient_path.is_file():
        gradients = load_json(gradient_path)
        if gradients.get("status") != "pass" or gradients.get("nonfinite_steps"):
            failures.append("gradient diagnostics report non-finite gradients")
    else:
        gradients = None
        unavailable.append("completed_gradient_diagnostics")
    current = full_identity(args, contract)
    payload = {
        "schema_version": 1,
        "status": "RETROACTIVE_AUDIT_FAIL" if failures else (
            "RETROACTIVE_AUDIT_INCOMPLETE"
            if unavailable
            else "RETROACTIVE_AUDIT_PASS"
        ),
        "result_eligibility": "quarantined" if failures else "provisional",
        "audited_at_unix": time.time(),
        "target": args.target,
        "run_dir": str(args.run_dir.resolve()),
        "identity_observed_after_launch": current,
        "training": training,
        "gradients": gradients,
        "artifacts": inventory,
        "failures": failures,
        "unavailable_evidence": unavailable,
    }
    report_dir = args.run_dir / "verification"
    write_json(report_dir / "retroactive_audit.json", payload)
    lines = [
        f"# Retroactive audit: {args.target}",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This run started before the verification gate existed. Current checkout",
        "identity is an after-launch observation, not proof of launch identity.",
        "",
        f"Recorded artifacts: {len(inventory)}",
        f"Result eligibility: `{payload['result_eligibility']}`",
    ]
    if failures:
        lines.extend(["", "Failures:", *[f"- {item}" for item in failures]])
    if unavailable:
        lines.extend(
            ["", "Unavailable evidence:", *[f"- {item}" for item in unavailable]]
        )
    (report_dir / "retroactive_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


def postrun_audit(args: argparse.Namespace, contract: dict[str, Any]) -> None:
    manifest_path = launch_manifest_path(args.run_dir)
    if not manifest_path.is_file():
        raise RuntimeError(
            "run has no prospective launch manifest; use audit-existing instead"
        )
    identity = full_identity(args, contract)
    require_clean(identity)
    launch = load_json(manifest_path)
    same_identity(launch["identity"], identity)
    failures = []
    required = [
        args.run_dir / "resolved_config.yaml",
        args.run_dir / "training_status.json",
        args.run_dir / "training_model_summary.json",
        args.run_dir / "gradient_diagnostics.json",
        args.run_dir / "optimizer_audit.json",
        args.run_dir / "best_selection.json",
        args.run_dir / "checkpoints/best.pt",
    ]
    required.extend(
        args.run_dir / f"checkpoints/checkpoint_{epoch}.pt"
        for epoch in range(1, 6)
    )
    required.extend(
        args.run_dir / f"val/selected/R{interval}/sav_eval.json"
        for interval in range(1, 7)
    )
    if launch.get("wandb_mode") == "online":
        required.append(args.run_dir / "wandb/wandb_run.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        failures.append(f"missing required artifacts: {missing}")
    if not missing:
        training = load_json(args.run_dir / "training_status.json")
        gradients = load_json(args.run_dir / "gradient_diagnostics.json")
        optimizer = load_json(args.run_dir / "optimizer_audit.json")
        if training.get("status") != "complete":
            failures.append("training did not complete")
        if gradients.get("status") != "pass" or gradients.get("nonfinite_steps"):
            failures.append("gradient diagnostics failed")
        try:
            validate_optimizer_audit(optimizer)
        except ValueError as error:
            failures.append(str(error))
        for interval in range(1, 7):
            evaluation = load_json(
                args.run_dir / f"val/selected/R{interval}/sav_eval.json"
            )
            if evaluation.get("status") != "pass":
                failures.append(f"selected R{interval} evaluation did not pass")
    payload = {
        "schema_version": 1,
        "status": "RESULT_ACCEPTED" if not failures else "RESULT_QUARANTINED",
        "audited_at_unix": time.time(),
        "target": args.target,
        "run_dir": str(args.run_dir.resolve()),
        "identity": identity,
        "selected_checkpoint_sha256": (
            sha256_file(args.run_dir / "checkpoints/best.pt")
            if (args.run_dir / "checkpoints/best.pt").is_file()
            else None
        ),
        "failures": failures,
        "artifacts": artifact_inventory(args.run_dir),
    }
    write_json(args.run_dir / "verification/postrun_audit.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


def add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--sam2-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    subparsers = parser.add_subparsers(dest="action", required=True)

    local = subparsers.add_parser("local")
    local.add_argument("--evidence-dir", required=True, type=Path)

    launch = subparsers.add_parser("record-launch")
    launch.add_argument("--evidence-dir", required=True, type=Path)
    launch.add_argument("--run-dir", required=True, type=Path)
    launch.add_argument("--scope", required=True, choices=("smoke", "formal"))
    launch.add_argument("--target", required=True)
    launch.add_argument("--gpus", required=True)
    launch.add_argument("--seed", required=True, type=int)
    launch.add_argument("--wandb-project", required=True)
    launch.add_argument("--wandb-mode", required=True)
    launch.add_argument("--setting", action="append", default=[])
    add_identity_arguments(launch)

    check = subparsers.add_parser("check-launch")
    check.add_argument("--evidence-dir", required=True, type=Path)
    check.add_argument("--run-dir", required=True, type=Path)
    check.add_argument("--target", required=True)
    add_identity_arguments(check)

    remote = subparsers.add_parser("remote")
    remote.add_argument("--evidence-dir", required=True, type=Path)
    remote.add_argument("--smoke-dir", required=True, type=Path)
    remote.add_argument("--input-audit", required=True, type=Path)
    add_identity_arguments(remote)

    existing = subparsers.add_parser("audit-existing")
    existing.add_argument("--run-dir", required=True, type=Path)
    existing.add_argument("--target", required=True)
    add_identity_arguments(existing)

    postrun = subparsers.add_parser("postrun")
    postrun.add_argument("--run-dir", required=True, type=Path)
    postrun.add_argument("--target", required=True)
    add_identity_arguments(postrun)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.contract = args.contract.resolve()
    contract = load_json(args.contract)
    validate_contract(contract, args.repo_root)
    if args.action == "local":
        issue_local_stamp(args, contract)
    elif args.action == "record-launch":
        record_launch(args, contract)
    elif args.action == "check-launch":
        check_launch(args, contract)
    elif args.action == "remote":
        issue_remote_stamp(args, contract)
    elif args.action == "audit-existing":
        audit_existing(args, contract)
    elif args.action == "postrun":
        postrun_audit(args, contract)
    else:
        raise AssertionError(args.action)


if __name__ == "__main__":
    main()
