#!/usr/bin/env bash
# Run on convir-4090 from a clean checkout. Each launch reserves a fresh folder.
set -euo pipefail
split=${1:?usage: bash tools/run_sots.sh indoor|outdoor GPU NEW_RUN_ROOT [MAX_IMAGES]}
gpu=${2:?missing GPU}
run_root=${3:?missing new run root}
max_images=${4:-0}
case "$run_root" in
  /sda/home/wangyuxin/Dehaze/runs/*) ;;
  *) printf 'Run root must be below /sda/home/wangyuxin/Dehaze/runs/\n' >&2; exit 2 ;;
esac
case "$split" in
  indoor) tag=ITS; weight=its-base.pkl; border=10 ;;
  outdoor) tag=OTS; weight=ots-base.pkl; border=0 ;;
  *) printf 'Unknown SOTS split: %s\n' "$split" >&2; exit 2 ;;
esac
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
mkdir -- "$run_root"
trap 'code=$?; if [ "$code" -ne 0 ]; then printf "FAILED exit=%s\n" "$code" > "$run_root/status.txt"; fi' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee "$run_root/run.log") 2>&1
printf 'PREFLIGHT\n' > "$run_root/status.txt"
cp -- "${BASH_SOURCE[0]}" "$run_root/launcher.sh"
git -C "$repo" rev-parse HEAD > "$run_root/code_commit.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv > "$run_root/gpu_before.csv"
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
runtime=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
assets=/sda/home/wangyuxin/ConvIR-B
command=("$runtime" "$repo/tools/evaluate_wdmamba.py"
  --data-root "$assets/datasets/RESIDE/official/SOTS" --split "$split"
  --dataset-name "SOTS-$split" --expected-count 500 --max-images "$max_images"
  --gt-border "$border" --out-dir "$run_root/evaluation"
  --convir-its-dir "$assets/repos/ConvIR-B-official-arch-anchor/Dehazing/ITS"
  --convir-dataset "$tag" --a0-checkpoint "$assets/checkpoints/official/$weight"
  --wdmamba-repo "$assets/repos/external_experts/WDMamba"
  --wdmamba-checkpoint /sda/home/wangyuxin/Dehaze/assets/checkpoints/wdmamba/reside-6k/reside6k_32.15.pth
  --alphas 0 0.125 0.25 0.375 0.5 0.75 1 --device cuda:0 --print-freq 10 --save-images)
# Exact command is evidence; re-running it intentionally fails if output exists.
printf '#!/usr/bin/env bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1\n' "$gpu" > "$run_root/command.sh"
printf '%q ' "${command[@]}" >> "$run_root/command.sh"
printf '\n' >> "$run_root/command.sh"
printf 'RUNNING\n' > "$run_root/status.txt"
"${command[@]}"
printf 'DEHAZE_SOTS_RUN_OK\n' > "$run_root/status.txt"
printf 'DEHAZE_SOTS_RUN_OK split=%s out=%s\n' "$split" "$run_root"
