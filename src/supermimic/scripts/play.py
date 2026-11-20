"""Play a SuperMimic multi-encoder actor against the sim."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import mjlab.tasks  # noqa: F401
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.viewers import (
  SuperMimicNativeViewer,
  SuperMimicViserViewer,
  create_wm_vis_model,
)
from supermimic.utils.checkpoints import load_local_run, run_motion_files


@dataclass
class PlayConfig:
  model_dir: str
  """Local training run directory or a versioned checkpoints/<step> directory."""
  config_file: str | None = None
  """Optional matching config.json for older or separately copied checkpoints."""
  motion_file: str | None = None
  initial_task: Literal["motion", "velocity"] = "motion"
  """Task selected when the viewer opens. Switch live via the GUI dropdown."""
  device: str | None = None
  num_steps_rollout: int = 1000000
  viewer: Literal["auto", "native", "viser"] = "viser"


def play(task_id: str, cfg: PlayConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  print(f"[INFO] Loading environment: {task_id}")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.observations["actor"].enable_corruption = True

  model_dir, run_config = load_local_run(cfg.model_dir, cfg.config_file)
  runner_cfg = load_runner_cfg(run_config)
  check_representation_matches(task_id, runner_cfg.representation)

  command_term = env_cfg.commands["motion"]
  assert isinstance(command_term, MotionCommandCfg)
  motion_files = run_motion_files(run_config, cfg.motion_file)
  command_term.motion_file = motion_files
  print(f"[INFO] Using local motions: {motion_files}")

  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))
  motion_names = env.unwrapped.command_manager.get_term("motion").motion.motion_names

  wm = build_world_model(runner_cfg, env, device)
  representation = wm.representation
  dynamics = wm.dynamics

  wm.load_bundle(torch.load(model_dir / "world_model.pt", map_location=device))
  dynamics.eval()

  actor = build_actor(runner_cfg, wm, device)
  actor.load_state_dict(torch.load(model_dir / "actor.pt", map_location=device))
  actor.eval()

  wm_vis_model = create_wm_vis_model(env.unwrapped.sim.mj_model)

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved = "native" if has_display else "viser"
  else:
    resolved = cfg.viewer
  print(f"[INFO] Starting viewer: {resolved}")

  if resolved == "native":
    SuperMimicNativeViewer(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      sim_context=wm.sim_context,
      task_specs=actor.task_specs,
      initial_task=cfg.initial_task,
    ).run()
  else:
    SuperMimicViserViewer(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      motion_names=motion_names,
      sim_context=wm.sim_context,
      task_specs=actor.task_specs,
      initial_task=cfg.initial_task,
    ).run()

  env.close()


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(PlayConfig, args=remaining_args, config=mjlab.TYRO_FLAGS)
  play(chosen_task, args)


if __name__ == "__main__":
  main()
