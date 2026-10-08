#!/usr/bin/env bash
set -euo pipefail
python3 collect_minigrid_data.py --env MiniGrid-LavaCrossingS9N3-v0 --envs 10000 --H 400 --train "$@"
