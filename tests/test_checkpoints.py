"""Regression coverage for local inputs and self-contained checkpoint snapshots."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg
from supermimic.utils.checkpoints import (
  archive_checkpoint,
  load_local_run,
  resolve_motion_files,
  run_motion_files,
  write_run_config,
)


def test_motion_inputs_expand_globs_lists_and_packed_directories(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  for name in ("b.npz", "a.npz"):
    (tmp_path / name).touch()
  expected = [str(tmp_path / "a.npz"), str(tmp_path / "b.npz")]
  assert resolve_motion_files("*.npz") == expected
  assert resolve_motion_files("a.npz, b.npz") == expected
  assert resolve_motion_files(["a.npz", "b.npz"]) == expected
  assert resolve_motion_files("a.npz") == expected[0]

  packed = tmp_path / "packed"
  packed.mkdir()
  (packed / "motion_ends.npy").touch()
  assert resolve_motion_files(str(packed)) == str(packed)
  with pytest.raises(ValueError, match="on its own"):
    resolve_motion_files([str(packed), "a.npz"])


@pytest.mark.parametrize("pattern", ["missing.npz", "missing*.npz", "", "a.npz,"])
def test_missing_motions_fail_before_building_a_simulator(
  tmp_path, monkeypatch, pattern
):
  monkeypatch.chdir(tmp_path)
  (tmp_path / "a.npz").touch()
  with pytest.raises(FileNotFoundError):
    resolve_motion_files(pattern)


def test_local_run_preserves_nondefault_architecture_and_motion_override(tmp_path):
  motion = tmp_path / "motion.npz"
  motion.touch()
  cfg = SuperMimicRunnerCfg(
    dynamics_hidden_dims=(128, 64), latent_dim=16, history_length=3, actor_obs="proprio"
  )
  serialized = write_run_config(
    tmp_path,
    {
      "runner_cfg": asdict(cfg),
      "motion_file": str(motion),
      "model_dir": tmp_path,
      "env_cfg": {
        "callback": test_local_run_preserves_nondefault_architecture_and_motion_override
      },
    },
  )
  root, loaded = load_local_run(tmp_path)
  assert root == tmp_path
  assert loaded == serialized
  assert load_runner_cfg(loaded) == cfg
  assert run_motion_files(loaded) == str(motion)
  assert not (tmp_path / "config.json.tmp").exists()

  override = tmp_path / "override.npz"
  override.touch()
  assert run_motion_files(loaded, str(override)) == str(override)
  assert run_motion_files(
    {"env_cfg": {"commands": {"motion": {"motion_file": str(motion)}}}}
  ) == str(motion)
  with pytest.raises(ValueError, match="--motion-file"):
    run_motion_files({})


def test_missing_config_never_silently_uses_default_network(tmp_path):
  with pytest.raises(FileNotFoundError, match="--config-file"):
    load_local_run(tmp_path)
  config = tmp_path / "external.json"
  config.write_text('{"runner_cfg": {"latent_dim": 17}}')
  _, loaded = load_local_run(tmp_path, config)
  assert loaded["runner_cfg"]["latent_dim"] == 17


def test_ppo_checkpoints_fail_with_an_explanation(tmp_path):
  write_run_config(tmp_path, {"runner_type": "PpoRunnerCfg"})
  with pytest.raises(ValueError, match="PPO"):
    load_local_run(tmp_path)


def test_snapshots_are_independent_and_keep_recent_versions_and_milestones(tmp_path):
  write_run_config(tmp_path, {"runner_cfg": {"latent_dim": 17}})
  weights = tmp_path / "actor.pt"
  for version in range(1, 7):
    weights.write_text(f"weights-{version}")
    archive_checkpoint(
      [str(weights)],
      step=version * 100,
      version=version,
      keep_last=2,
      milestone_every=3,
    )
  snapshots = sorted((tmp_path / "checkpoints").iterdir())
  assert [path.name for path in snapshots] == ["000000300", "000000500", "000000600"]
  assert (snapshots[0] / "actor.pt").read_text() == "weights-3"
  assert (snapshots[-1] / "actor.pt").read_text() == "weights-6"
  _, config = load_local_run(snapshots[0])
  assert config["runner_cfg"]["latent_dim"] == 17
  assert json.loads((snapshots[0] / "checkpoint.json").read_text())["step"] == 300


def test_failed_snapshot_cleans_up_partial_copy(tmp_path, monkeypatch):
  write_run_config(tmp_path, {"runner_cfg": {"latent_dim": 17}})
  weights = tmp_path / "actor.pt"
  weights.write_text("latest")

  def fail_copy(*args):
    raise OSError("disk full")

  monkeypatch.setattr("supermimic.utils.checkpoints.shutil.copy2", fail_copy)
  with pytest.raises(OSError, match="disk full"):
    archive_checkpoint([str(weights)], step=100, version=1)
  assert list((tmp_path / "checkpoints").iterdir()) == []
  assert weights.read_text() == "latest"
