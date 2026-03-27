#!/usr/bin/env bash
#SBATCH --job-name=jigsaw_run
#SBATCH --account=rashidin_1842
#SBATCH --partition=rashidinejad
#SBATCH --time=02:00:00

#SBATCH --cpus-per-task=30
#SBATCH --mem=100G

#SBATCH --gres=gpu:1

#SBATCH --output=logs_slurm2_train/%x-%j.slurm.out
#SBATCH --error=logs_slurm2_train/%x-%j.slurm.err

set -euo pipefail

# ---------------------------
# Required inputs (sbatch --export)
# ---------------------------
: "${DATA_TYPE:?Set DATA_TYPE (wiki|zsre|counterfact|safeedit_train|safeedit_test|zsre10k)}"
: "${ENERGY_THRESHOLD:?Set ENERGY_THRESHOLD (e.g., 0.8)}"
: "${CACHE_SAMPLE_NUM:?Set CACHE_SAMPLE_NUM (e.g., 100000)}"

# ---------------------------
# Optional inputs
# ---------------------------
# NUM_EDITS:
#   if set -> enables sequential edit and sets --num_edits
#   example: NUM_EDITS=100
#
# RECALC_CACHE:
#   if set to 1/true -> enables --recalculate_cache
#
# RECALC_WEIGHT_THR:
#   if set -> enables --recalculate_cache AND sets --recalculate_weight_threshold <num>
#   example: RECALC_WEIGHT_THR=0.4
#
# FT:
#   if set to 1/true -> disables SNAP projection by adding --no_snap (i.e., regular finetuning)
#
# DISABLE_OLD_LOSS_CHECK:
#   if set to 1/true -> adds --disable_old_loss_check
#
# NO_SNAP:
#   alias for FT (either one works)
# LAYERS:
#   if set -> passes --layers "<comma-separated indices>"
#   example: LAYERS="19,20,21"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"
MODEL="${MODEL:-qwen2.5-7b}"
WANDB_PROJECT="${WANDB_PROJECT:-JIGSAW_QWEN_SWEEP}"
BATCH_SIZE="${BATCH_SIZE:-32}"

WORKDIR="/home1/firouzko/jigsaw"
mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"

# Tagging for logs
THR_TAG="${ENERGY_THRESHOLD/./p}"
RUN_TAG="Jigsaw_${DATA_TYPE}_thr${THR_TAG}_cache${CACHE_SAMPLE_NUM}_job${SLURM_JOB_ID}"
exec > >(tee -a "logs/${RUN_TAG}.out") 2>&1

echo "=== SLURM ==="
echo "JobID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_JOB_NODELIST}"
echo "Start: $(date)"
echo "=== INPUTS ==="
echo "MODEL: ${MODEL}"
echo "DATA_TYPE: ${DATA_TYPE}"
echo "ENERGY_THRESHOLD: ${ENERGY_THRESHOLD}"
echo "CACHE_SAMPLE_NUM: ${CACHE_SAMPLE_NUM}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"
echo "NUM_EDITS: ${NUM_EDITS:-<unset>}"
echo "RECALC_CACHE: ${RECALC_CACHE:-<unset>}"
echo "LAYERS: ${LAYERS:-<unset>}"
echo "RECALC_WEIGHT_THR: ${RECALC_WEIGHT_THR:-<unset>}"
echo "FT/NO_SNAP: ${FT:-${NO_SNAP:-<unset>}}"
echo "DISABLE_OLD_LOSS_CHECK: ${DISABLE_OLD_LOSS_CHECK:-<unset>}"

echo "=== GPU ==="
nvidia-smi || true

# ---------------------------
# Environment
# ---------------------------
module unload python 2>/dev/null || true
export HF_CACHE_DIR="/project2/rashidin_1793/arad_cache/data/huggingface/hub/"
export HF_DATASETS_DIR="/project2/rashidin_1793/arad_cache/data/huggingface/datasets/"

export WANDB_ENTITY="aradfir-usc"

# Determine whether this run should NOT reuse the "default" stats/caches
NEEDS_UNIQUE_STATS=0

# Sequential runs -> unique namespace
if [ -n "${NUM_EDITS:-}" ]; then
  NEEDS_UNIQUE_STATS=1
fi

# Recalc threshold implies recalc_cache, also unique namespace
if [ -n "${RECALC_WEIGHT_THR:-}" ]; then
  NEEDS_UNIQUE_STATS=1
fi

# Explicit recalc_cache requested -> unique namespace
case "${RECALC_CACHE:-0}" in
  1|true|TRUE|yes|YES) NEEDS_UNIQUE_STATS=1 ;;
esac

export STATS_DIR="/project2/rashidin_1793/arad_cache/data/jigsaw_stats"
if [ "$NEEDS_UNIQUE_STATS" -eq 1 ]; then
    STATS_DIR="${STATS_DIR}/unique_${RANDOM}"
fi
mkdir -p "$STATS_DIR"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"

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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ---------------------------
# Build args (conditional flags)
# ---------------------------
ARGS=(
  --model "$MODEL"
  --data_type "$DATA_TYPE"
  --cache_sample_num "$CACHE_SAMPLE_NUM"
  --energy_threshold "$ENERGY_THRESHOLD"
  --batch_size "$BATCH_SIZE"
  --wandb_project "$WANDB_PROJECT"
)

# If NUM_EDITS is set -> enable sequential_edit and set num_edits
if [ -n "${NUM_EDITS:-}" ]; then
  ARGS+=( --sequential_edit --num_edits "$NUM_EDITS" --edit_cache_style sequential )
fi

# If RECALC_WEIGHT_THR is set -> enable recalculate_cache AND set weight threshold
if [ -n "${RECALC_WEIGHT_THR:-}" ]; then
  ARGS+=( --recalculate_cache --recalculate_weight_threshold "$RECALC_WEIGHT_THR" )
else
  # Otherwise optionally enable recalculate_cache if RECALC_CACHE=true/1
  case "${RECALC_CACHE:-0}" in
    1|true|TRUE|yes|YES) ARGS+=( --recalculate_cache ) ;;
  esac
fi
# If LAYERS is set -> pass through to python
if [ -n "${LAYERS:-}" ]; then
  ARGS+=( --layers "${LAYERS}" )
fi
ARGS+=( --disable_old_loss_check )
# Disable old loss check if requested
# case "${DISABLE_OLD_LOSS_CHECK:-0}" in
  # 1|true|TRUE|yes|YES) ARGS+=( --disable_old_loss_check ) ;;
# esac

# FT mode (disable SNAP) if requested: FT=1 or NO_SNAP=1
case "${FT:-${NO_SNAP:-0}}" in
  1|true|TRUE|yes|YES) ARGS+=( --no_crisp ) ;;
esac

CMD=( "$PY" run_crispedit.py "${ARGS[@]}" )

echo "=== RUN ==="
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-unset}"
printf 'Command:'; printf ' %q' "${CMD[@]}"; echo
echo "Time: $(date)"

srun --export=ALL --ntasks=1 "${CMD[@]}"

echo "Done: $(date)"
