#!/usr/bin/env python3
"""Capture current archived continuous-batched PLD prototype reproduction environment provenance."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "environment" / "current_environment.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B"

PACKAGE_NAMES = [
    "torch",
    "transformers",
    "vllm",
    "accelerate",
    "huggingface-hub",
    "modal",
    "numpy",
    "tree-sitter",
    "tree-sitter-language-pack",
    "datasets",
    "scikit-learn",
    "pydivsufsort",
    "editdistance",
]

ENV_ALLOWLIST = {
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "CUDA_HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "TOKENIZERS_PARALLELISM",
    "PYTHONPATH",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_HOME",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "MODAL_PROFILE",
}
ENV_PREFIXES = ("CUDA_", "NVIDIA_", "HF_", "HUGGINGFACE_", "TRANSFORMERS_", "VLLM_", "MODAL_", "NCCL_", "PYTORCH_", "TORCH_")
SENSITIVE_PARTS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "PASS", "CREDENTIAL", "AUTH")

MODAL_FILES = [
    ROOT / "vantage_runtime_debian_app.py",
    ROOT / "vantage_runtime_app.py",
    ROOT / "vantage_pytorch_app.py",
    ROOT / "vantage_modal_app.py",
    ROOT / "modal_smoke_gcp_app.py",
    ROOT / "scripts" / "pvp_microbench_modal.py",
]


def _run(cmd: list[str], *, cwd: Path = ROOT, timeout: int = 15) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {"cmd": cmd, "available": False, "reason": "command not found"}
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "available": False,
            "reason": f"timed out after {timeout}s",
            "stdout": (exc.stdout or "")[-4000:],
            "stderr": (exc.stderr or "")[-4000:],
        }
    return {
        "cmd": cmd,
        "available": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip()[-8000:],
        "stderr": proc.stderr.strip()[-4000:],
    }


def _git_value(args: list[str]) -> str | None:
    result = _run(["git", *args], timeout=10)
    if result.get("available"):
        return str(result.get("stdout") or "").strip()
    return None


def capture_git() -> dict[str, Any]:
    status = _git_value(["status", "--short"]) or ""
    return {
        "commit": _git_value(["rev-parse", "HEAD"]),
        "short_commit": _git_value(["rev-parse", "--short", "HEAD"]),
        "branch": _git_value(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
        "tracked_diff_stat": (_git_value(["diff", "--stat"]) or "").splitlines(),
        "untracked_files": [line[3:] for line in status.splitlines() if line.startswith("?? ")],
        "repository_root": str(ROOT),
    }


def capture_datetime() -> dict[str, str]:
    local_now = datetime.now().astimezone()
    utc_now = datetime.now(timezone.utc)
    return {
        "local_iso": local_now.isoformat(timespec="seconds"),
        "utc_iso": utc_now.isoformat(timespec="seconds"),
        "timezone": local_now.tzname() or "",
    }


def capture_python() -> dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": sys.version.replace("\n", " "),
        "version_info": list(sys.version_info[:5]),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
    }


def capture_packages() -> dict[str, dict[str, Any]]:
    packages: dict[str, dict[str, Any]] = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = {"installed": True, "version": importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {"installed": False, "version": None}
    return packages


def capture_torch() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        return {"installed": False, "import_error": repr(exc)}

    out: dict[str, Any] = {
        "installed": True,
        "version": getattr(torch, "__version__", None),
        "cuda_built": getattr(torch.version, "cuda", None),
        "hip_built": getattr(torch.version, "hip", None),
        "debug_build": bool(getattr(torch.version, "debug", False)),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    try:
        out["cudnn_version"] = torch.backends.cudnn.version()
    except Exception as exc:
        out["cudnn_version_error"] = repr(exc)

    devices = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_bytes": int(props.total_memory),
                    "major": int(props.major),
                    "minor": int(props.minor),
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )
    out["cuda_devices"] = devices
    return out


def _parse_nvidia_smi_csv(text: str) -> list[dict[str, str]]:
    rows = []
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        values = [col.strip() for col in row]
        if len(values) >= 7:
            rows.append(
                {
                    "index": values[0],
                    "name": values[1],
                    "uuid": values[2],
                    "memory_total_mib": values[3],
                    "driver_version": values[4],
                    "cuda_version": values[5],
                    "compute_capability": values[6],
                }
            )
    return rows


def capture_gpu_and_driver() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version,cuda_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    gpu_query = _run(query, timeout=10)
    smi_summary = _run(["nvidia-smi"], timeout=10)
    nvcc = _run(["nvcc", "--version"], timeout=10)
    return {
        "nvidia_smi_available": bool(gpu_query.get("available")),
        "nvidia_smi_query": gpu_query,
        "nvidia_smi_summary": smi_summary,
        "nvidia_smi_gpus": _parse_nvidia_smi_csv(str(gpu_query.get("stdout") or ""))
        if gpu_query.get("available")
        else [],
        "nvcc": nvcc,
    }


def capture_env_vars() -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key, value in sorted(os.environ.items()):
        if key in ENV_ALLOWLIST or any(key.startswith(prefix) for prefix in ENV_PREFIXES):
            sensitive = any(part in key.upper() for part in SENSITIVE_PARTS)
            selected[key] = {
                "present": True,
                "value": "<redacted>" if sensitive else value,
                "redacted": sensitive,
            }
    missing = sorted(key for key in ENV_ALLOWLIST if key not in selected)
    return {"selected": selected, "missing_allowlist": missing}


def _extract_balanced_block(lines: list[str], start_idx: int, max_lines: int = 80) -> list[dict[str, Any]]:
    block = []
    depth = 0
    started = False
    for idx in range(start_idx, min(len(lines), start_idx + max_lines)):
        line = lines[idx].rstrip("\n")
        stripped = line.strip()
        if not stripped and started and depth <= 0:
            break
        depth += line.count("(") + line.count("[") + line.count("{")
        depth -= line.count(")") + line.count("]") + line.count("}")
        started = True
        block.append({"line": idx + 1, "text": stripped})
        if started and depth <= 0 and stripped.endswith(")"):
            break
    return block


def capture_modal_references() -> dict[str, Any]:
    refs = []
    for path in MODAL_FILES:
        if not path.exists():
            refs.append({"file": str(path.relative_to(ROOT)), "exists": False})
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        image_blocks = []
        install_blocks = []
        app_names = []
        gpu_requests = []
        env_lines = []
        for idx, line in enumerate(lines):
            if re.search(r"\bimage\s*=\s*\(", line):
                image_blocks.append(_extract_balanced_block(lines, idx))
            if "install_cmds = [" in line:
                install_blocks.append(_extract_balanced_block(lines, idx, max_lines=70))
            app_match = re.search(r"modal\.App\(([^)]*)\)", line)
            if app_match:
                app_names.append({"line": idx + 1, "text": line.strip()})
            gpu_match = re.search(r"gpu\s*=\s*[\"']([^\"']+)[\"']", line)
            if gpu_match:
                gpu_requests.append({"line": idx + 1, "gpu": gpu_match.group(1)})
            if ".env(" in line or '"HF_HOME"' in line or '"PYTHONPATH"' in line:
                env_lines.append({"line": idx + 1, "text": line.strip()})
        refs.append(
            {
                "file": str(path.relative_to(ROOT)),
                "exists": True,
                "image_blocks": image_blocks[:2],
                "runtime_install_blocks": install_blocks[:2],
                "app_names": app_names,
                "gpu_requests": gpu_requests,
                "unique_gpu_requests": sorted({row["gpu"] for row in gpu_requests}),
                "env_reference_lines": env_lines[:20],
            }
        )
    return {
        "note": "References are source-code pointers to Modal image/runtime configuration, not proof of the exact historical image digest.",
        "files": refs,
    }


def _cache_snapshot_for_file(model_id: str, filename: str) -> dict[str, Any]:
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore
    except Exception as exc:
        return {"available": False, "reason": f"huggingface_hub unavailable: {exc!r}"}

    try:
        cached = try_to_load_from_cache(model_id, filename)
    except Exception as exc:
        return {"available": False, "reason": repr(exc)}
    if cached is None:
        return {"available": False, "reason": f"{filename} not found in local HF cache"}
    path = Path(str(cached))
    parts = path.parts
    snapshot = None
    if "snapshots" in parts:
        pos = parts.index("snapshots")
        if pos + 1 < len(parts):
            snapshot = parts[pos + 1]
    return {
        "available": True,
        "filename": filename,
        "path": str(path),
        "snapshot_commit": snapshot,
    }


def capture_model_revisions(model_id: str, tokenizer_id: str, allow_network: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "local_model_config": _cache_snapshot_for_file(model_id, "config.json"),
        "local_tokenizer_config": _cache_snapshot_for_file(tokenizer_id, "tokenizer_config.json"),
        "network_lookup_enabled": allow_network,
    }

    if allow_network:
        try:
            from huggingface_hub import HfApi  # type: ignore

            api = HfApi()
            try:
                model_info = api.model_info(model_id, timeout=10)
            except TypeError:
                model_info = api.model_info(model_id)
            out["remote_model"] = {
                "available": True,
                "sha": getattr(model_info, "sha", None),
                "last_modified": str(getattr(model_info, "last_modified", "")),
            }
            if tokenizer_id == model_id:
                out["remote_tokenizer"] = out["remote_model"]
            else:
                try:
                    tokenizer_info = api.model_info(tokenizer_id, timeout=10)
                except TypeError:
                    tokenizer_info = api.model_info(tokenizer_id)
                out["remote_tokenizer"] = {
                    "available": True,
                    "sha": getattr(tokenizer_info, "sha", None),
                    "last_modified": str(getattr(tokenizer_info, "last_modified", "")),
                }
        except Exception as exc:
            out["remote_lookup_error"] = repr(exc)
    else:
        out["remote_model"] = {
            "available": False,
            "reason": "network lookup disabled; rerun with --allow-network to query current Hub HEAD",
        }
        out["remote_tokenizer"] = out["remote_model"]
    return out


def _unavailable_fields(report: dict[str, Any]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    packages = report["packages"]
    for name in ("torch", "transformers", "vllm"):
        if not packages.get(name, {}).get("installed"):
            missing.append({"field": f"packages.{name}.version", "reason": "package not installed"})
    if not report["gpu_driver"].get("nvidia_smi_available"):
        reason = report["gpu_driver"].get("nvidia_smi_query", {}).get("reason", "nvidia-smi unavailable")
        missing.append({"field": "gpu_driver.nvidia_smi_gpus", "reason": str(reason)})
        missing.append({"field": "cuda_driver_version", "reason": str(reason)})
    torch_info = report["torch"]
    if not torch_info.get("installed"):
        missing.append({"field": "torch.cuda", "reason": torch_info.get("import_error", "torch unavailable")})
    elif not torch_info.get("cuda_available"):
        missing.append({"field": "torch.cuda_devices", "reason": "torch.cuda.is_available() is false"})
    revisions = report["model_and_tokenizer_revisions"]
    if not revisions.get("local_model_config", {}).get("available"):
        missing.append(
            {
                "field": "model_revision.local_cache_snapshot",
                "reason": str(revisions.get("local_model_config", {}).get("reason", "unavailable")),
            }
        )
    if not revisions.get("local_tokenizer_config", {}).get("available"):
        missing.append(
            {
                "field": "tokenizer_revision.local_cache_snapshot",
                "reason": str(revisions.get("local_tokenizer_config", {}).get("reason", "unavailable")),
            }
        )
    if not revisions.get("remote_model", {}).get("available"):
        missing.append(
            {
                "field": "model_revision.remote_head",
                "reason": str(revisions.get("remote_model", {}).get("reason", revisions.get("remote_lookup_error", "unavailable"))),
            }
        )
    missing.append(
        {
            "field": "historical_run_embedded_environment_metadata",
            "reason": "historical timing/correctness run JSON does not embed exact model/tokenizer revision, CUDA driver, or package build metadata",
        }
    )
    return missing


def build_report(model_id: str, tokenizer_id: str, allow_network: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "label": "current reproduction environment",
        "historical_metadata_note": (
            "Historical archived continuous-batched PLD prototype run artifacts lack complete embedded environment "
            "metadata; this file records the current environment used for reproduction."
        ),
        "captured_at": capture_datetime(),
        "git": capture_git(),
        "python": capture_python(),
        "packages": capture_packages(),
        "torch": capture_torch(),
        "gpu_driver": capture_gpu_and_driver(),
        "environment_variables": capture_env_vars(),
        "modal_image_config_references": capture_modal_references(),
        "model_and_tokenizer_revisions": capture_model_revisions(model_id, tokenizer_id, allow_network),
        "capture_command": {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "allow_network": allow_network,
        },
    }
    report["unavailable_fields"] = _unavailable_fields(report)
    return report


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    packages = report["packages"]
    torch_info = report["torch"]
    gpu = report["gpu_driver"]
    revisions = report["model_and_tokenizer_revisions"]
    git = report["git"]

    lines = [
        "# Current Environment Provenance",
        "",
        "**Label:** current reproduction environment.",
        "",
        "Historical archived continuous-batched PLD prototype timing/correctness artifacts do not embed the full "
        "environment metadata. Treat this capture as the current reproduction "
        "environment, not as proof of the exact historical Modal container state.",
        "",
        "## Capture",
        "",
        f"- Local time: `{report['captured_at']['local_iso']}`",
        f"- UTC time: `{report['captured_at']['utc_iso']}`",
        f"- Hostname: `{report['python']['hostname']}`",
        "",
        "## Git",
        "",
        f"- Commit: `{git.get('commit') or 'unavailable'}`",
        f"- Branch: `{git.get('branch') or 'unavailable'}`",
        f"- Dirty worktree: `{_yes_no(git.get('dirty'))}`",
        f"- Status entries: `{len(git.get('status_short', []))}`",
        "",
        "## Runtime",
        "",
        f"- Python executable: `{report['python']['executable']}`",
        f"- Python version: `{report['python']['version']}`",
        f"- Platform: `{report['python']['platform']}`",
        "",
        "## Packages",
        "",
        "| Package | Installed | Version |",
        "|---|---|---|",
    ]
    for name in PACKAGE_NAMES:
        row = packages[name]
        lines.append(f"| `{name}` | {_yes_no(row.get('installed'))} | `{row.get('version') or ''}` |")

    lines += [
        "",
        "## PyTorch And CUDA",
        "",
        f"- PyTorch importable: `{_yes_no(torch_info.get('installed'))}`",
        f"- PyTorch version: `{torch_info.get('version') or ''}`",
        f"- CUDA built version: `{torch_info.get('cuda_built') or ''}`",
        f"- CUDA available to PyTorch: `{_yes_no(torch_info.get('cuda_available'))}`",
        f"- CUDA device count: `{torch_info.get('cuda_device_count', 0)}`",
        f"- cuDNN version: `{torch_info.get('cudnn_version') or ''}`",
        "",
        "## GPU And Driver",
        "",
        f"- `nvidia-smi` available: `{_yes_no(gpu.get('nvidia_smi_available'))}`",
    ]
    if gpu.get("nvidia_smi_gpus"):
        lines += ["", "| Index | GPU | Memory MiB | Driver | CUDA | Compute |", "|---:|---|---:|---|---|---|"]
        for row in gpu["nvidia_smi_gpus"]:
            lines.append(
                "| {index} | {name} | {memory_total_mib} | {driver_version} | {cuda_version} | {compute_capability} |".format(
                    **row
                )
            )
    else:
        lines.append(f"- GPU query reason: `{gpu.get('nvidia_smi_query', {}).get('reason', 'unavailable')}`")

    lines += [
        "",
        "## Model And Tokenizer Revision",
        "",
        f"- Model id: `{revisions.get('model_id')}`",
        f"- Tokenizer id: `{revisions.get('tokenizer_id')}`",
        f"- Local model snapshot: `{revisions.get('local_model_config', {}).get('snapshot_commit') or ''}`",
        f"- Local tokenizer snapshot: `{revisions.get('local_tokenizer_config', {}).get('snapshot_commit') or ''}`",
        f"- Remote model HEAD: `{revisions.get('remote_model', {}).get('sha') or ''}`",
        f"- Network lookup enabled: `{_yes_no(revisions.get('network_lookup_enabled'))}`",
        "",
        "## Modal/Image References",
        "",
        "The JSON artifact records source-code references for Modal app/image blocks, "
        "runtime package install blocks, GPU decorators, and Modal environment settings.",
        "",
    ]
    for file_ref in report["modal_image_config_references"]["files"]:
        if file_ref.get("exists"):
            lines.append(
                f"- `{file_ref['file']}`: apps={len(file_ref.get('app_names', []))}, "
                f"image_blocks={len(file_ref.get('image_blocks', []))}, "
                f"gpu_requests={file_ref.get('unique_gpu_requests', [])}"
            )
        else:
            lines.append(f"- `{file_ref['file']}`: missing")

    lines += [
        "",
        "## Environment Variables",
        "",
        f"- Captured selected variables: `{len(report['environment_variables']['selected'])}`",
        "- Sensitive variable values are redacted by name.",
        "",
        "## Unavailable Fields",
        "",
    ]
    for item in report["unavailable_fields"]:
        lines.append(f"- `{item['field']}`: {item['reason']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Query the Hugging Face Hub for the current model/tokenizer HEAD revision.",
    )
    args = parser.parse_args()

    tokenizer = args.tokenizer or args.model
    report = build_report(args.model, tokenizer, args.allow_network)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, out.with_suffix(".md"))
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
