#!/usr/bin/env bash
#SBATCH --job-name=edit_bench_loc
#SBATCH --account=rashidin_1842
#SBATCH --partition=ce_hopper
#SBATCH --time=04:00:00

#SBATCH --cpus-per-task=30
#SBATCH --mem=50G

#SBATCH --gres=gpu:1

#SBATCH --output=logs_slurm_large_train/%x-%j.slurm.out
#SBATCH --error=logs_slurm_large_train/%x-%j.slurm.err


set -euo pipefail

JOBLIST="${1:?Usage: sbatch --array=0-(2*M-1)%N run_edited_benchmarks_prompttype_array.sbatch bench_jobs.tsv}"

WORKDIR="/home1/firouzko/jigsaw"
mkdir -p "$WORKDIR/logs2qwen"
cd "$WORKDIR"

# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True"
export WANDB_ENTITY="aradfir-usc"

module unload python 2>/dev/null || true
export HF_CACHE_DIR="/project2/rashidin_1842/arad_cache/data/huggingface/hub/"
export HF_DATASETS_DIR="/project2/rashidin_1793/arad_cache/data/huggingface/datasets/"
export STATS_DIR="/project2/rashidin_1842/arad_cache/data/jigsaw_stats/"
export HF_HUB_CACHE="/project2/rashidin_1842/arad_cache/data/huggingface/hub/"

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
sleep 2
# ---- Only run these prompt types, qa_inst and no_context ----
prompt_types=("qa_inst" "no_context")
num_pt=${#prompt_types[@]}

# ---- Map SLURM_ARRAY_TASK_ID -> (joblist row, prompt_type) ----
# task_id = row_idx * num_pt + pt_idx
row_idx=$((SLURM_ARRAY_TASK_ID / num_pt))
pt_idx=$((SLURM_ARRAY_TASK_ID % num_pt))
prompt_type="${prompt_types[$pt_idx]}"

# ---- Read the TSV line for this model row ----
line_no=$((row_idx + 1))
line="$(sed -n "${line_no}p" "$JOBLIST" || true)"
if [ -z "$line" ]; then
  echo "ERROR: No line ${line_no} in ${JOBLIST} (row_idx=${row_idx})" >&2
  exit 1
fi

IFS=$'\t' read -r alg_raw edited_model_dir data_type <<<"$line"
if [ -z "${alg_raw:-}" ] || [ -z "${edited_model_dir:-}" ] || [ -z "${data_type:-}" ]; then
  echo "ERROR: Malformed TSV line ${line_no}: $line" >&2
  exit 1
fi

# ---- Parse alg_raw into (algName, thr?) given data_type column ----
# Expected patterns:
#   1) METHOD_<data_type>_<thr>   e.g., AlphaEditFT_zsre_0.5
#   2) METHOD_<data_type>         e.g., MEMIT_zsre
# Otherwise fallback:
#   METHOD = alg_raw, thr empty
method=""
thr=""

if [[ "$alg_raw" =~ ^(.+)_${data_type}_([0-9]+([.p][0-9]+)?)$ ]]; then
  method="${BASH_REMATCH[1]}"
  thr="${BASH_REMATCH[2]}"
  thr="${thr/p/.}"
elif [[ "$alg_raw" =~ ^(.+)_${data_type}$ ]]; then
  method="${BASH_REMATCH[1]}"
  thr=""
else
  method="$alg_raw"
  thr=""
fi

# ---- Build alg_name in the requested format: algName_thr_dataType_promptType ----
# If thr is missing: algName_dataType_promptType
if [ -n "$thr" ]; then
  alg_name="${method}_${thr}_${data_type}_${prompt_type}"
else
  alg_name="${method}_${data_type}_${prompt_type}"
fi

RUN_TAG="editbench_${alg_name}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
exec > >(tee -a "logs2qwen/${RUN_TAG}.out") 2>&1

echo "=== SLURM ==="
echo "JobID: ${SLURM_JOB_ID}"
echo "ArrayTaskID: ${SLURM_ARRAY_TASK_ID}"
echo "RowIdx: ${row_idx} (line ${line_no})"
echo "PromptType: ${prompt_type}"
echo "Node(s): ${SLURM_JOB_NODELIST}"
echo "Start: $(date)"
echo "=== JOB ==="
echo "ALG_RAW: ${alg_raw}"
echo "METHOD: ${method}"
echo "THR: ${thr:-<none>}"
echo "ALG_NAME (used): ${alg_name}"
echo "EDITED_MODEL_DIR: ${edited_model_dir}"
echo "DATA_TYPE: ${data_type}"
# echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"

nvidia-smi || true

CMD=("$PY" run_edited_benchmarks.py
  --edited_model_dir "${edited_model_dir}"
  --model_name "qwen2.5-32b"
  --max_length 40
  --context_type "${prompt_type}"
  --alg_name "${alg_name}"
  --data_type "${data_type}"
  --evaluation_criteria "llm_judge"
  --eval_num 3000
  --wandb_project "JIGSAW_LARGE_EVAL_EDITS"
)

echo "=== RUN ==="
printf 'Command:'; printf ' %q' "${CMD[@]}"; echo

srun --export=ALL --ntasks=1 "${CMD[@]}"

echo "Done: $(date)"
