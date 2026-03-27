#!/usr/bin/env bash
#SBATCH --job-name=jigsaw
#SBATCH --account=ywang234_1595
#SBATCH --partition=gpu
#SBATCH --time=4:00:00

#SBATCH --cpus-per-task=15
#SBATCH --mem=50G

#SBATCH --gres=gpu:1
#SBATCH --constraint='a100-80gb|a100-40gb|l40s|a40'

#SBATCH --mail-type=all
#SBATCH --mail-user=firouzko@usc.edu

#SBATCH --output=logs/%x-%j.slurm.out
#SBATCH --error=logs/%x-%j.slurm.err

set -euo pipefail

# ---- Inputs (pass via sbatch --export) ----
: "${METHOD:?Set METHOD (MEMIT|UltraEdit|AlphaEdit|Loc-BF-FT|AlphaEditFT|AlphaEdit-100Batch|WISE|LoRA|LoRASeq)}"
: "${DATA_TYPE:?Set DATA_TYPE (wiki|zsre|counterfact|zsre10k)}"

# ---- Fixed config ----
MODEL="qwen2.5-7b"
WANDB_PROJECT="JIGSAW_QWEN"

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
echo "DATA_TYPE: ${DATA_TYPE}"
echo "MODEL: ${MODEL}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"

echo "=== GPU ==="
nvidia-smi || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# ---- Environment ----
module unload python 2>/dev/null
export HF_CACHE_DIR="/project2/rashidin_1793/arad_cache/data/huggingface/hub/"
export HF_DATASETS_DIR="/project2/rashidin_1793/arad_cache/data/huggingface/datasets/"
export STATS_DIR="/project2/rashidin_1793/arad_cache/data/jigsaw_stats/"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"

# Conda activation (adjust if conda lives elsewhere)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found under ~/miniconda3 or ~/anaconda3. Edit this script." >&2
  exit 1
fi

conda activate jigsaw3

echo "PATH=$PATH"
which python || true
python --version || true

PY="$(which python || true)"
if [ -z "$PY" ]; then
  echo "ERROR: python not found after conda activate. Check conda env or PATH." >&2
  exit 1
fi

# Helpful for CPU-threaded libs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "$WORKDIR"
# ---- Choose command by METHOD ----
case "$METHOD" in
  "MEMIT")
    CMD=("$PY" edit.py
      --model "$MODEL"
      --data_type "$DATA_TYPE"
      --editing_method MEMIT
      --batch_size 32
      --eval_every 512
      --batch_edit True
      --wandb_project "$WANDB_PROJECT"
    )
    ;;

  "UltraEdit")
    CMD=("$PY" edit.py
      --model "$MODEL"
      --data_type "$DATA_TYPE"
      --editing_method UltraEdit
      --batch_size 32
      --eval_every 512
      --batch_edit True
      --wandb_project "$WANDB_PROJECT"
    )
    ;;

  "AlphaEdit")
    CMD=("$PY" edit.py
      --model "$MODEL"
      --data_type "$DATA_TYPE"
      --editing_method AlphaEdit
      --batch_size 32
      --eval_every 512
      --batch_edit True
      --wandb_project "$WANDB_PROJECT"
    )
    ;;

  "Loc-BF-FT")
    CMD=("$PY" locft-bf.py
      --model "$MODEL"
      --data_type "$DATA_TYPE"
      --batch_size 32
      --wandb_project "$WANDB_PROJECT"
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
    )
    ;;
  "AlphaEdit-100Batch")
    CMD=("$PY" edit.py
      --model "$MODEL"
      --data_type "$DATA_TYPE"
      --editing_method AlphaEdit
      --batch_size 100
      --eval_every 500
      --batch_edit True
      --wandb_project "$WANDB_PROJECT"
    )
    ;;
  "WISE")
    CMD=("$PY" edit.py
      --model "$MODEL"
      --data_type "$DATA_TYPE"
      --editing_method WISE
      --batch_size 1
      --eval_every 512
      --batch_edit False
      --wandb_project "$WANDB_PROJECT"
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
    )
    ;;
  "LoRASeq")
  CMD=("$PY" run_snap.py
      --model "$MODEL"
      --data_type "$DATA_TYPE"
      --batch_size 32
      --no_snap
      --perform_lora
      --lora_type adalora
      --wandb_project "$WANDB_PROJECT"
    )
    ;;
  *)
    echo "ERROR: Unknown METHOD='$METHOD'." >&2
    echo "Allowed: MEMIT | UltraEdit | AlphaEdit | Loc-BF-FT | AlphaEditFT" >&2
    exit 1
    ;;
esac

echo "=== RUN ==="
printf 'Command:'; printf ' %q' "${CMD[@]}"; echo
echo "Time: $(date)"

# Use srun so Slurm tracks the job step correctly
srun --ntasks=1 "${CMD[@]}"

echo "Done: $(date)"
