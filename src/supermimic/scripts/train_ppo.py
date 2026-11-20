"""Script to test PPORunner."""

from __future__ import annotations

from dataclasses import dataclass

import mjlab.tasks  # noqa: F401
import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.runners.ppo_runner import PpoRunner, PpoRunnerCfg
from supermimic.scripts.train_utils import (
  create_env,
  setup_device,
)
from supermimic.utils.checkpoints import DEFAULT_MOTION_FILE, resolve_motion_files


@dataclass
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  runner: PpoRunnerCfg

  # Experiment & Environment
  num_envs: int = 1024
  device: str | None = None
  motion_file: str = DEFAULT_MOTION_FILE
  """Local NPZ, comma-separated files, glob, or packed motion directory."""

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id, play=False)
    return TrainConfig(
      env=env_cfg,
      runner=PpoRunnerCfg(),
    )


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )

  device = setup_device(args.device)
  print(f"[INFO] Loading environment: {chosen_task}")

  args.env.commands["motion"].motion_file = resolve_motion_files(args.motion_file)
  print(f"[INFO] Using local motion: {args.env.commands['motion'].motion_file}")

  if "MaxCoord" in chosen_task and args.runner.representation == "gencoord":
    args.runner.representation = "maxcoord"
    print("[INFO] Auto-detected maxcoord representation from task_id")

  env = create_env(args.env, args.num_envs, device)

  # Init Runner
  runner = PpoRunner(
    env=env,
    cfg=args.runner,
    task_id=chosen_task,
    device=device,
  )

  runner.run()


if __name__ == "__main__":
  main()
