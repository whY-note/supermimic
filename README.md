# SuperMimic

[Project page](https://supermimic.github.io) | [Paper](https://arxiv.org/abs/2610.28480)

SuperMimic trains a humanoid whole-body motion tracker with one differentiable world model, built on [mjlab](https://github.com/mujocolab/mjlab).

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone both repos as siblings:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone git@github.com:HybridRobotics/supermimic.git
git clone git@github.com:HybridRobotics/mjlab_research.git
cd supermimic
uv sync
```

## Usage

Available task IDs:

- `SuperMimic-GenCoord-G1` — generalized coordinates representation
- `SuperMimic-MaxCoord-G1` — maximal coordinates representation

Experiment tracking uses **SwanLab 0.9.8**. For cloud tracking, first run
`uv run swanlab login`; use `SWANLAB_MODE=offline` to record locally without login.
The mode can also be set with `--runner.swanlab-mode`. Project and workspace are
configured with `--runner.swanlab-project` and `--runner.swanlab-workspace`.

Prepare a local G1 motion NPZ (the default is
`data/lafan_run1_subject5/motion.npz`). To convert a 30 FPS G1 CSV without uploading:

```python
import numpy as np
from mjlab.scripts.csv_to_npz import MotionLoader

motion = MotionLoader("run1_subject5.csv", input_fps=30, output_fps=50, device="cpu")
np.savez("motion.npz", **motion.arrays())
```

Train a world model and policy with local motions:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run src/supermimic/scripts/train.py \
  <TASK_ID> --motion-file data/lafan_run1_subject5/motion.npz \
  --runner.run-name <description>
```

`--motion-file` also accepts comma-separated NPZ paths, a quoted glob, or one
packed motion directory. Training and playback do not download motion data.

Each run saves `config.json`, `world_model.pt`, `actor.pt` and, when present,
`critic.pt` in `logs/<TASK_ID>/<timestamp>/`. SwanLab logs are under that run's
`swanlog/` directory. Model checkpoints stay on disk: SwanLab's public cloud does
not provide a replacement for a versioned model registry in this integration.
Snapshots in `checkpoints/<step>/` include their configuration; the latest five
versions and every 25th version are retained. The latest weights are also saved
when training finishes. These weights do not include optimizer/replay state.

Play back a local trained model in the browser (including offline runs):

```bash
uv run src/supermimic/scripts/play.py \
  <TASK_ID> --model-dir logs/<TASK_ID>/<timestamp> --viewer viser
```

Playback restores the network settings from `config.json` and uses the motions
recorded during training. Pass `--motion-file /new/path/motion.npz` after moving
the dataset. A `checkpoints/<step>/` directory can also be passed to `--model-dir`.
For older weights without a saved configuration, supply a matching JSON file
with `--config-file`; the loader never silently substitutes a default network.
Playback currently supports the main SuperMimic actor, not the separate PPO actor.

For CPU playback without using any GPU:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run src/supermimic/scripts/play.py \
  <TASK_ID> --model-dir logs/<TASK_ID>/<timestamp> --device cpu --viewer viser
```

Open the Viser URL printed in the terminal (normally port 8080). For a remote
server, forward the port with `ssh -N -L 8080:127.0.0.1:8080 <user>@<server>` and
open `http://localhost:8080` locally.

Benchmark a local checkpoint with
`uv run src/supermimic/scripts/benchmark.py --model-dir <MODEL_DIR>`.
Use `--runner.world-model-dir <MODEL_DIR>` during training to initialize only the
world-model weights from another local checkpoint; this is not a full resume.

The old remote registry/run arguments have been removed. SwanLab does not mirror
the old motion registry. The upstream `mjlab` dependency may still install its
own tracking dependencies, but these SuperMimic entry points use SwanLab.

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
