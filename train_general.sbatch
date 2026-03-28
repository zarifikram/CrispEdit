#!/usr/bin/env bash
#SBATCH --job-name=jigsaw_gen
#SBATCH --account=rashidin_1842
#SBATCH --partition=rashidinejad
#SBATCH --time=02:00:00

#SBATCH --cpus-per-task=30
#SBATCH --mem=200G

#SBATCH --gres=gpu:1

#SBATCH --output=logs/%x-%j.slurm.out
#SBATCH --error=logs/%x-%j.slurm.err

set -euo pipefail

# ---- Inputs (pass via sbatch --export) ----
: "${METHOD:?Set METHOD (MEMIT|UltraEdit|AlphaEdit|Loc-BF-FT|AlphaEditFT|AlphaEdit-100Batch|WISE|LoRA|LoRASeq)}"
: "${DATA_TYPE:?Set DATA_TYPE (wiki|zsre|counterfact|zsre10k)}"

LAYERS="${LAYERS:-}" # <--- Capture optional LAYERS from environment

# ---- Fixed config ----
MODEL="qwen2.5-32b"
WANDB_PROJECT="JIGSAW_LARGE"
export WANDB_ENTITY="aradfir-usc"

WORKDIR="/home1/firouzko/jigsaw"
mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"

RUN_TAG="${METHOD}_${DATA_TYPE}_job${SLURM_JOB_ID}"
exec > >(tee -a "logs/${RUN_TAG}.out") 2>&1

echo "=== SLURM ==="
echo "JobID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_JOB_NODELIST}"
echo "Start: $(date)"
echo "=== CONFIG ==="
echo "WORKDIR: ${WORKDIR}"
echo "METHOD: ${METHOD}"
echo "LAYERS: ${LAYERS:-<default YAML>}"
echo "DATA_TYPE: ${DATA_TYPE}"
echo "MODEL: ${MODEL}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"

echo "=== GPU ==="
nvidia-smi || true

# ---- Environment ----
module unload python 2>/dev/null
export HF_CACHE_DIR="/project2/rashidin_1842/arad_cache/data/huggingface/hub/"
export HF_DATASETS_DIR="/project2/rashidin_1793/arad_cache/data/huggingface/datasets/"
export STATS_DIR="/project2/rashidin_1842/arad_cache/data/jigsaw_stats/"
export HF_HUB_CACHE="/project2/rashidin_1842/arad_cache/data/huggingface/hub/"
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
[ -n "$PY" ] || { echo "ERROR: python not found."; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "$WORKDIR"

# ---- Build Optional Layer Arguments ----
EXTRA_ARGS=()
if [[ -n "$LAYERS" ]]; then
  EXTRA_ARGS+=( --layers "$LAYERS" )
fi

# ---- Choose command by METHOD ----
case "$METHOD" in
  "MEMIT"|"UltraEdit")
    CMD=("$PY" edit.py
      --model "$MODEL"
      --data_type "$DATA_TYPE"
      --editing_method "$METHOD"
      --num_edits 32
      --eval_every 512
      --batch_edit True
      --wandb_project "$WANDB_PROJECT"
      "${EXTRA_ARGS[@]}" # <--- Injects --layers if provided
    )
    ;;

  "AlphaEdit"|"AlphaEdit-100Batch")
    CMD=("$PY" edit.py
      --model "$MODEL"
      --data_type "$DATA_TYPE"
      --editing_method AlphaEdit
      --num_edits 100
      --eval_every 500
      --batch_edit True
      --wandb_project "$WANDB_PROJECT"
      "${EXTRA_ARGS[@]}"
    )
    ;;

  "Loc-BF-FT")
    CMD=("$PY" locft-bf.py
      --model "$MODEL"
      --data_type "$DATA_TYPE"
      --batch_size 32
      --wandb_project "$WANDB_PROJECT"
      "${EXTRA_ARGS[@]}"
    )
    ;;

  "AlphaEditFT")
    CMD=("$PY" alphaedit_ft.py
      --model "$MODEL"
      --data_type "$DATA_TYPE"
      --cache_sample_num 10000
      --energy_threshold 0.5
      --batch_size 32
      --wandb_project "$WANDB_PROJECT"
      "${EXTRA_ARGS[@]}"
    )
    ;;

  "WISE"|"MEND")
    CMD=("$PY" edit.py
      --model "$MODEL"
      --data_type "$DATA_TYPE"
      --editing_method "$METHOD"
      --num_edits 1
      --eval_every 512
      --batch_edit False
      --wandb_project "$WANDB_PROJECT"
      "${EXTRA_ARGS[@]}"
    )
    ;;

  "LoRA")
    CMD=("$PY" edit.py
      --model "$MODEL"
      --data_type "$DATA_TYPE"
      --editing_method LoRA
      --num_edits 100
      --batch_size 32
      --eval_every 100
      --batch_edit True
      --wandb_project "$WANDB_PROJECT"
      "${EXTRA_ARGS[@]}"
    )
    ;;

  "LoRASeq")
    CMD=("$PY" run_crispedit.py
      --model "$MODEL"
      --data_type "$DATA_TYPE"
      --batch_size 32
      --wandb_project "$WANDB_PROJECT"
      --no_crisp
      --perform_lora
      --lora_type adalora
      # Note: run_crispedit.py already natively supported --layers, 
      # so passing it via EXTRA_ARGS maps cleanly.
      "${EXTRA_ARGS[@]}"
    )
    ;;

  *)
    echo "ERROR: Unknown METHOD='$METHOD'." >&2
    exit 1
    ;;
esac

echo "=== RUN ==="
printf 'Command:'; printf ' %q' "${CMD[@]}"; echo
echo "Time: $(date)"

srun --ntasks=1 "${CMD[@]}"

echo "Done: $(date)"