#!/usr/bin/env bash
#SBATCH --job-name=base_bench
#SBATCH --account=rashidin_1842
#SBATCH --partition=ce_hopper
#SBATCH --time=04:00:00

#SBATCH --cpus-per-task=30
#SBATCH --mem=100G

#SBATCH --gres=gpu:1

#SBATCH --output=logs_slurm_large_train/%x-%j.slurm.out
#SBATCH --error=logs_slurm_large_train/%x-%j.slurm.err


set -euo pipefail

JOBLIST="${1:?Usage: sbatch --array=0-N%K run_base_benchmarks_array.sbatch bench_jobs.tsv}"

WORKDIR="/home1/firouzko/jigsaw"
mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"

# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"
export WANDB_ENTITY="aradfir-usc"

module unload python 2>/dev/null || true
export HF_CACHE_DIR="/project2/rashidin_1793/arad_cache/data/huggingface/hub/"
export HF_DATASETS_DIR="/project2/rashidin_1793/arad_cache/data/huggingface/datasets/"
export STATS_DIR="/project2/rashidin_1793/arad_cache/data/jigsaw_stats/"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found." >&2
  exit 1
fi

conda activate jigsaw3
PY="$(which python || true)"
[ -n "$PY" ] || { echo "ERROR: python not found after conda activate." >&2; exit 1; }

# ---- Read TSV line for this array task ----
line_no=$((SLURM_ARRAY_TASK_ID + 1))
line="$(sed -n "${line_no}p" "$JOBLIST" || true)"
if [ -z "$line" ]; then
  echo "ERROR: No line ${line_no} in ${JOBLIST}" >&2
  exit 1
fi

IFS=$'\t' read -r alg_raw edited_model_dir data_type <<<"$line"
if [ -z "${alg_raw:-}" ] || [ -z "${edited_model_dir:-}" ] || [ -z "${data_type:-}" ]; then
  echo "ERROR: Malformed TSV line ${line_no}: $line" >&2
  exit 1
fi

# ---- Reformat ALG_NAME: METHOD_<data_type>_<thr> --> METHOD_<thr>_<data_type>
# Supports thresholds like 0.5 / 0.95 / 0.999 or 0p95 (will convert p->.)
alg_name="$alg_raw"
if [[ "$alg_raw" =~ ^(.+)_${data_type}_([0-9]+([.p][0-9]+)?)$ ]]; then
  method="${BASH_REMATCH[1]}"
  thr="${BASH_REMATCH[2]}"
  thr="${thr/p/.}"
  alg_name="${method}_${thr}_${data_type}"
fi

RUN_TAG="basebench_${alg_name}_${data_type}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
exec > >(tee -a "logs/${RUN_TAG}.out") 2>&1

echo "=== SLURM ==="
echo "JobID: ${SLURM_JOB_ID}"
echo "ArrayTaskID: ${SLURM_ARRAY_TASK_ID}"
echo "Node(s): ${SLURM_JOB_NODELIST}"
echo "Start: $(date)"
echo "=== JOB ==="
echo "ALG_RAW: ${alg_raw}"
echo "ALG_NAME (used): ${alg_name}"
echo "EDITED_MODEL_DIR: ${edited_model_dir}"
echo "DATA_TYPE: ${data_type}"
# echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"

nvidia-smi || true

CMD=("$PY" run_base_benchmarks.py
  --edited_model_dir "${edited_model_dir}"
  --model_name "qwen2.5-7b"
  --alg_name "${alg_name}"
  --data_type "${data_type}"
  --eval_num 200
  --wandb_project "QWEN_JIGSAW_SWEEP_BASE_EVAL"
)

echo "=== RUN ==="
printf 'Command:'; printf ' %q' "${CMD[@]}"; echo

srun --export=ALL --ntasks=1 "${CMD[@]}"

echo "Done: $(date)"
