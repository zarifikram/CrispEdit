#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./submit_general_with_eval.sh <METHOD> <DATA_TYPE> [LAYERS]
# Example:
#   ./submit_general_with_eval.sh AlphaEdit zsre "19,20,21"

METHOD="${1:?METHOD required (MEMIT|UltraEdit|AlphaEdit|Loc-BF-FT|AlphaEditFT|WISE|LoRA)}"
DATA_TYPE="${2:?DATA_TYPE required (wiki|zsre|counterfact|zsre10k)}"
LAYERS="${3:-}" # <--- Optional 3rd argument for layers

MODEL="qwen2.5-32b"

# ---- Predict the Exact Save Name from Python Logic ----
if [[ "$METHOD" == "Loc-BF-FT" ]]; then
  # locft-bf.py logic: f"{args.model}_{hparams.alg_name}_{args.data_type}" -> alg_name is "FT"
  EDITED_MODEL_DIR="${MODEL}_FT_${DATA_TYPE}"

elif [[ "$METHOD" == "AlphaEditFT" ]]; then
  # alphaedit_ft.py logic: hardcodes energy_threshold to 0.5 in train_general.sbatch
  EDITED_MODEL_DIR="${MODEL}_AlphaEditFT_${DATA_TYPE}_0.5"

elif [[ "$METHOD" == "AlphaEdit-100Batch" ]]; then
  # edit.py logic: passes --editing_method AlphaEdit, so the name drops the "100Batch" part
  EDITED_MODEL_DIR="${MODEL}_AlphaEdit_${DATA_TYPE}"

elif [[ "$METHOD" == "LoRASeq" ]]; then
  # Custom run_snap.py fallback (ensure this matches your snap script)
  EDITED_MODEL_DIR="${MODEL}_LoRASeq_${DATA_TYPE}"

else
  # edit.py standard logic: f"{args.model}_{args.editing_method}_{args.data_type}"
  # Covers: MEMIT, UltraEdit, AlphaEdit, WISE, LoRA
  EDITED_MODEL_DIR="${MODEL}_${METHOD}_${DATA_TYPE}"
fi

# ---- Append Layer Suffix if specified ----
if [[ -n "$LAYERS" ]]; then
  # Replace commas with dashes to match Python's '-'.join()
  LAYER_STR="${LAYERS//,/-}"
  EDITED_MODEL_DIR="${EDITED_MODEL_DIR}_L${LAYER_STR}"
fi

RUN_NAME="${EDITED_MODEL_DIR}"

# ---- Setup TSV for Evaluation ----
TSV_DIR="eval_joblists"
mkdir -p "$TSV_DIR"
TSV_FILE="${TSV_DIR}/${RUN_NAME}.tsv"
printf "%s\t%s\t%s\n" "$RUN_NAME" "$EDITED_MODEL_DIR" "$DATA_TYPE" > "$TSV_FILE"

echo "=== Pipeline Config ==="
echo "METHOD:            $METHOD"
echo "LAYERS:            ${LAYERS:-<default YAML>}"
echo "RUN_NAME:          $RUN_NAME"
echo "EDITED_MODEL_DIR:  $EDITED_MODEL_DIR"
echo "TSV FILE:          $TSV_FILE"
echo

# ---- Submit Training Job ----
EXPORTS="ALL,METHOD=${METHOD},DATA_TYPE=${DATA_TYPE}"
if [[ -n "$LAYERS" ]]; then
  # Add layers to the SLURM export without internal quotes
  EXPORTS="${EXPORTS},LAYERS=${LAYERS}"
fi

TRAIN_JOBID="$(sbatch --parsable --export="${EXPORTS}" train_general.sbatch)"
echo "Submitted training job: ${TRAIN_JOBID}"

# ---- Submit Eval Jobs (Chained) ----
EVAL_JOBID="$(sbatch --parsable --dependency=afterok:${TRAIN_JOBID} --array=0-1 run_edited_benchmarks_prompttype_array.sbatch "${TSV_FILE}")"
echo "Submitted  edit eval job: ${EVAL_JOBID} (afterok:${TRAIN_JOBID})"

EVAL_JOBID="$(sbatch --parsable --dependency=afterok:${TRAIN_JOBID} --array=0-0 run_base_benchmarks_array.sbatch "${TSV_FILE}")"
echo "Submitted base eval job: ${EVAL_JOBID} (afterok:${TRAIN_JOBID})"