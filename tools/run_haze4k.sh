#!/usr/bin/env bash
# Reproduce the historical Haze4K test alpha grid on convir-4090.
set -euo pipefail
gpu=${1:?usage: bash tools/run_haze4k.sh GPU NEW_RUN_ROOT [MAX_IMAGES] [SAVE_IMAGES]}
run_root=${2:?missing new run root}
max_images=${3:-0}
save_images=${4:-0}
case "$run_root" in
  /sda/home/wangyuxin/Dehaze/runs/*) ;;
  *) printf 'Run root must be below /sda/home/wangyuxin/Dehaze/runs/\n' >&2; exit 2 ;;
esac
case "$save_images" in 0|1) ;; *) printf 'SAVE_IMAGES must be 0 or 1\n' >&2; exit 2 ;; esac
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
export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
runtime=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
assets=/sda/home/wangyuxin/ConvIR-B
command=("$runtime" "$repo/tools/evaluate_wdmamba.py"
  --data-root "$assets/datasets/Haze4K/Haze4K" --split test
  --dataset-name Haze4K-test-reproduction --expected-count 1000 --max-images "$max_images"
  --out-dir "$run_root/evaluation"
  --convir-its-dir "$assets/repos/ConvIR-B-official-arch-anchor/Dehazing/ITS"
  --convir-dataset Haze4K --a0-checkpoint "$assets/checkpoints/official/Haze4K/haze4k-base.pkl"
  --wdmamba-repo "$assets/repos/external_experts/WDMamba"
  --wdmamba-checkpoint "$assets/checkpoints/WDMamba_ckpts/haze4k_35.88.pth"
  --alphas 0 0.125 0.25 0.375 0.5 0.75 1 --ssim-reference-factor 32
  --device cuda:0 --print-freq 20)
if test "$save_images" = 1; then command+=(--save-images); fi
printf '#!/usr/bin/env bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=1 CUBLAS_WORKSPACE_CONFIG=:4096:8\n' "$gpu" > "$run_root/command.sh"
printf '%q ' "${command[@]}" >> "$run_root/command.sh"
printf '\n' >> "$run_root/command.sh"
printf 'RUNNING\n' > "$run_root/status.txt"
"${command[@]}"
printf 'DEHAZE_HAZE4K_RUN_OK\n' > "$run_root/status.txt"
printf 'DEHAZE_HAZE4K_RUN_OK out=%s\n' "$run_root"
