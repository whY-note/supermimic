"""Check that training sends metrics to SwanLab and keeps local inference metadata."""

from dataclasses import asdict, dataclass
from types import SimpleNamespace

import torch

from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import BaseRunner, SuperMimicRunnerCfg
from supermimic.utils.checkpoints import load_local_run
from supermimic.utils.logger import Logger


def test_training_initialization_keeps_matching_local_and_swanlab_config(
  tmp_path, monkeypatch
):
  @dataclass
  class MotionCfg:
    motion_file: str

  @dataclass
  class EnvCfg:
    commands: dict

  cfg = SuperMimicRunnerCfg(
    swanlab_project="test-project",
    swanlab_workspace="test-team",
    swanlab_mode="offline",
    run_name="test-run",
    latent_dim=17,
  )
  env_cfg = EnvCfg(commands={"motion": MotionCfg(str(tmp_path / "motion.npz"))})
  runner = object.__new__(BaseRunner)
  runner.cfg = cfg
  runner.task_id = "SuperMimic-GenCoord-G1"
  runner.env = SimpleNamespace(cfg=env_cfg, unwrapped=SimpleNamespace(cfg=env_cfg))
  runner.logger = SimpleNamespace(run_name=None)
  captured = {}

  def init(**kwargs):
    captured.update(kwargs)
    return SimpleNamespace(name=kwargs["name"])

  monkeypatch.setattr("supermimic.runners.supermimic_runner.swanlab.init", init)
  runner._init_swanlab(tmp_path)

  _, saved = load_local_run(tmp_path)
  assert captured["project"] == "test-project"
  assert captured["workspace"] == "test-team"
  assert captured["mode"] == "offline"
  assert captured["log_dir"] == str(tmp_path / "swanlog")
  assert captured["config"] == saved
  assert saved["runner_cfg"] == captured["config"]["runner_cfg"]
  assert asdict(load_runner_cfg(saved)) == asdict(cfg)
  assert saved["motion_file"] == env_cfg.commands["motion"].motion_file
  assert runner.logger.run_name == "test-run"


def test_logger_sends_metrics_and_steps_to_swanlab(monkeypatch):
  captured = []
  monkeypatch.setattr(
    "supermimic.utils.logger.swanlab.log",
    lambda row, step: captured.append((row, step)),
  )
  logger = Logger("test", num_envs=1, max_episode_length=100, device="cpu")
  logger.rewbuffer.extend([2.0, 4.0])
  logger.lenbuffer.extend([10, 20])
  logger.log(
    it=7,
    total_it=10,
    collect_time=0.5,
    wm_learn_time=0.25,
    policy_learn_time=0.25,
    loss_dict={"WorldModel/loss": torch.tensor(1.5)},
    num_collected_steps=100,
  )
  row, step = captured.pop()
  assert step == 7
  assert row["WorldModel/loss"] == 1.5
  assert row["Train/mean_reward"] == 3.0
  assert row["Train/mean_episode_length"] == 15.0
  assert row["Perf/collection_time"] == 0.5
