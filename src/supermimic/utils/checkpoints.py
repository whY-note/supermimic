"""Local run configuration, motion inputs, and versioned model checkpoints.

SwanLab tracks metrics and configuration. Model weights and motions live on disk
so that training, playback, and benchmarks also work without a cloud connection.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path
from typing import Any

DEFAULT_MOTION_FILE = "data/lafan_run1_subject5/motion.npz"


def resolve_motion_files(motion_file: str | list[str]) -> str | list[str]:
  """Expand a local file, packed directory, comma list, or glob to absolute paths."""
  patterns = motion_file.split(",") if isinstance(motion_file, str) else motion_file
  files: list[str] = []
  for pattern in patterns:
    expanded = os.path.expanduser(pattern.strip())
    matches = sorted(glob.glob(expanded)) if glob.has_magic(expanded) else [expanded]
    if not expanded or not matches:
      raise FileNotFoundError(f"No local motions matched: {pattern!r}")
    for match in matches:
      path = Path(match).resolve()
      if not path.is_file() and not (path / "motion_ends.npy").is_file():
        raise FileNotFoundError(
          f"Motion not found: {path}. Provide a local NPZ or packed motion directory "
          "with --motion-file."
        )
      files.append(str(path))
  if not files:
    raise ValueError("At least one local motion file is required")
  if len(files) > 1 and any(Path(path).is_dir() for path in files):
    raise ValueError("A packed motion directory must be used on its own")
  return files[0] if len(files) == 1 else files


def write_run_config(log_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
  """Persist the same JSON-compatible configuration used by SwanLab."""
  # Environment configs include callables and other simulator-specific objects;
  # only runner_cfg and motion_file are reconstructed for inference.
  serialized = json.dumps(config, default=str, indent=2, ensure_ascii=False)
  target = log_dir / "config.json"
  temporary = target.with_suffix(".json.tmp")
  temporary.write_text(serialized + "\n")
  temporary.replace(target)
  return json.loads(serialized)


def load_local_run(
  model_dir: str | Path, config_file: str | Path | None = None
) -> tuple[Path, dict[str, Any]]:
  """Read a saved run or a versioned checkpoint, without contacting a service."""
  root = Path(model_dir).expanduser().resolve()
  if not root.is_dir():
    raise FileNotFoundError(f"Model directory not found: {root}")
  config_path = (
    Path(config_file).expanduser().resolve() if config_file else root / "config.json"
  )
  if not config_path.is_file():
    raise FileNotFoundError(
      f"Run configuration not found: {config_path}. New training runs save "
      "config.json beside the weights. For older weights, provide a matching "
      "JSON configuration with --config-file."
    )
  config = json.loads(config_path.read_text())
  if not isinstance(config, dict):
    raise ValueError(f"Expected a JSON object in {config_path}")
  # The PPO runner uses a different actor architecture and checkpoint layout.
  if config.get("runner_type", "SuperMimicRunnerCfg") != "SuperMimicRunnerCfg":
    raise ValueError("Playback/benchmark currently require a SuperMimic actor, not PPO")
  return root, config


def run_motion_files(
  config: dict[str, Any], override: str | None = None
) -> str | list[str]:
  """Use an explicit local motion or the absolute paths recorded during training."""
  motion_file = override if override is not None else config.get("motion_file")
  if motion_file is None:
    motion_file = (
      config.get("env_cfg", {}).get("commands", {}).get("motion", {}).get("motion_file")
    )
  if not motion_file:
    raise ValueError("No motion recorded in the run; provide --motion-file")
  return resolve_motion_files(motion_file)


def archive_checkpoint(
  files: list[str],
  step: int,
  version: int,
  keep_last: int = 5,
  milestone_every: int = 25,
) -> Path:
  """Snapshot a checkpoint and retain recent versions plus periodic milestones.

  The caller first writes the latest weights in the run directory. Each snapshot
  also contains config.json so that it can be copied and replayed independently.
  """
  if keep_last < 1 or milestone_every < 1:
    raise ValueError("Checkpoint retention counts must be positive")
  root = Path(files[0]).parent
  checkpoints = root / "checkpoints"
  checkpoints.mkdir(exist_ok=True)
  target = checkpoints / f"{step:09d}"
  temporary = checkpoints / f".{step:09d}.tmp"
  temporary.mkdir()
  try:
    for file in [*map(Path, files), root / "config.json"]:
      shutil.copy2(file, temporary / file.name)
    (temporary / "checkpoint.json").write_text(
      json.dumps({"step": step, "version": version}) + "\n"
    )
    temporary.replace(target)
  finally:
    if temporary.exists():
      shutil.rmtree(temporary)

  versions = sorted(path for path in checkpoints.iterdir() if path.name.isdecimal())
  for old in versions[:-keep_last]:
    metadata = json.loads((old / "checkpoint.json").read_text())
    if metadata["version"] % milestone_every:
      shutil.rmtree(old)
  return target
