#!/usr/bin/env bash
# Evaluate one paired real-haze dataset on convir-4090.
set -euo pipefail
dataset=${1:?usage: bash tools/run_real_dataset.sh DATASET GPU NEW_RUN_ROOT [MAX_IMAGES]}
gpu=${2:?missing GPU}
run_root=${3:?missing new run root}
max_images=${4:-0}
case "$run_root" in
  /sda/home/wangyuxin/Dehaze/runs/*) ;;
  *) printf 'Run root must be below /sda/home/wangyuxin/Dehaze/runs/\n' >&2; exit 2 ;;
esac
case "$dataset" in
  nhhaze)
    source_root=/sda/home/wangyuxin/Dehaze/assets/datasets/NH-HAZE
    conv_tag=NH-HAZE; conv_weight=/sda/home/wangyuxin/Dehaze/assets/checkpoints/convir/nhhaze-base.pkl
    expert_weight=/sda/home/wangyuxin/Dehaze/assets/checkpoints/wdmamba/NH_20.83.pth
    expected=55; flat=1 ;;
  ohaze)
    source_root=/sda/home/wangyuxin/Dehaze/assets/datasets/O-HAZE
    conv_tag=O-HAZE; conv_weight=/sda/home/wangyuxin/Dehaze/assets/checkpoints/convir/ohaze-base.pkl
    expert_weight=/sda/home/wangyuxin/Dehaze/assets/checkpoints/wdmamba/Ohaze_1200_1600_27.22.pth
    expected=45; flat=0 ;;
  densehaze)
    source_root=/sda/home/wangyuxin/Dehaze/assets/datasets/Dense-Haze
    conv_tag=Dense-Haze; conv_weight=/sda/home/wangyuxin/Dehaze/assets/checkpoints/convir/densehaze-base.pkl
    expert_weight=/sda/home/wangyuxin/Dehaze/assets/checkpoints/wdmamba/Dense_17.52.pth
    expected=55; flat=0 ;;
  *) printf 'Unknown real dataset: %s\n' "$dataset" >&2; exit 2 ;;
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
input_dir=$source_root/hazy
gt_dir=$source_root/GT
if test "$flat" = 1; then
  input_dir=$run_root/input
  gt_dir=$run_root/gt
  mkdir -p "$input_dir" "$gt_dir"
  shopt -s nullglob
  for input in "$source_root"/*_hazy.*; do
    name=$(basename -- "$input")
    stem=${name%.*}; suffix=${name##*.}
    ln -s -- "$input" "$input_dir/$name"
    gt_name="${stem/_hazy/_GT}.${suffix}"
    gt_source="$source_root/$gt_name"
    if test ! -f "$gt_source"; then
      gt_source=$(find "$source_root" -maxdepth 1 -type f -iname "${stem/_hazy/_GT}.*" -print -quit)
    fi
    test -n "$gt_source" -a -f "$gt_source"
    ln -s -- "$gt_source" "$gt_dir/$(basename -- "$gt_source")"
  done
  test "$(find "$input_dir" -maxdepth 1 -type l | wc -l)" -eq "$expected"
fi
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
runtime=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
assets=/sda/home/wangyuxin/ConvIR-B
command=("$runtime" "$repo/tools/evaluate_wdmamba.py"
  --data-root "$source_root" --split test --input-dir "$input_dir" --gt-dir "$gt_dir"
  --dataset-name "$dataset" --expected-count "$expected" --max-images "$max_images"
  --out-dir "$run_root/evaluation"
  --convir-its-dir "$assets/repos/ConvIR-B-official-arch-anchor/Dehazing/ITS"
  --convir-dataset "$conv_tag" --a0-checkpoint "$conv_weight"
  --wdmamba-repo "$assets/repos/external_experts/WDMamba"
  --wdmamba-checkpoint "$expert_weight"
  --alphas 0 0.125 0.25 0.375 0.5 0.75 1 --device cuda:0 --print-freq 1)
printf '#!/usr/bin/env bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1\n' "$gpu" > "$run_root/command.sh"
printf '%q ' "${command[@]}" >> "$run_root/command.sh"
printf '\n' >> "$run_root/command.sh"
printf 'RUNNING\n' > "$run_root/status.txt"
"${command[@]}"
printf 'DEHAZE_REAL_DATASET_RUN_OK\n' > "$run_root/status.txt"
printf 'DEHAZE_REAL_DATASET_RUN_OK dataset=%s out=%s\n' "$dataset" "$run_root"
