#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./submit_snap_with_eval.sh <DATA_TYPE> <ENERGY_THRESHOLD> <CACHE_SAMPLE_NUM> [NUM_EDITS] [RECALC_WEIGHT_THR] [FT]
#
# FT can be: 1 / true / FT / ft
#
# Examples:
#   ./submit_snap_with_eval.sh zsre 0.99 500 100 0.10
#   ./submit_snap_with_eval.sh zsre 0.99 500 100 0.10 ft
# DATA_TYPE="${1:?DATA_TYPE required (wiki|zsre|counterfact|zsre10k)}"
# ENERGY_IN="${2:?ENERGY_THRESHOLD required (e.g. 0.99)}"
# CACHE_SAMPLE_NUM="${3:?CACHE_SAMPLE_NUM required (e.g. 1000)}"
# NUM_EDITS="${4:-}"              # optional -> enables sequential if provided
# # FT_MODE="${5:-0}"               # optional -> enables FT/no_snap
# # RECALC_WEIGHT_THR="${6:-}"      # optional -> enables recalc if provided
# LAYERS="${7:-}"                 # optional -> comma-separated layers "19,20,21"
DATA_TYPE="${1:?DATA_TYPE required (wiki|zsre|counterfact|zsre10k)}"
ENERGY_IN="${2:?ENERGY_THRESHOLD required (e.g. 0.99)}"
CACHE_SAMPLE_NUM="${3:?CACHE_SAMPLE_NUM required (e.g. 1000)}"

# New order: layers before num_edits
LAYERS="${4:-}"     # optional -> comma-separated layers "19,20,21"
NUM_EDITS="${5:-}"  # optional -> enables sequential if provided

# Force finetuning + recalc OFF in this interface
FT_MODE=0
RECALC_WEIGHT_THR=""
MODEL="qwen2.5-7b"

# ---- Normalize energy threshold like Python float -> str(float) ----
ENERGY_THRESHOLD="$(python -c 'import sys; print(float(sys.argv[1]))' "$ENERGY_IN")"

# ---- FT toggle ----
FT_ENABLED=0
case "${FT_MODE}" in
  1|true|TRUE|FT|ft|yes|YES) FT_ENABLED=1 ;;
esac

# Base hparams.alg_name is JIGSAW, but for FT it becomes JigsawFT per your function
BASE_ALG="CRISPEDIT"
if [[ "$FT_ENABLED" -eq 1 ]]; then
  BASE_ALG="CRISPEDITFT"
fi

# ---- Build alg_name exactly like calculate_model_name() ----
ALG_NAME="${BASE_ALG}"
ALG_NAME+="Cache${CACHE_SAMPLE_NUM}"

if [[ -n "${NUM_EDITS}" ]]; then
  ALG_NAME+="Seq${NUM_EDITS}"
fi

if [[ -n "${RECALC_WEIGHT_THR}" ]]; then
  # Python logic: f"{thr:.2f}".replace("0.","")  => 0.10 -> "10"
  RECALC_TAG="$(python -c 'import sys; print(f"{float(sys.argv[1]):.2f}".replace("0.",""))' "$RECALC_WEIGHT_THR")"
  ALG_NAME+="Recalc${RECALC_TAG}"
fi

# Folder name from your python:
# name = f"{args.model}_{alg_name}_{args.data_type}_{args.energy_threshold}"
RUN_NAME="${ALG_NAME}_${DATA_TYPE}_${ENERGY_THRESHOLD}"
if [[ -n "${LAYERS}" ]]; then
  LAYER_DASHED="${LAYERS//,/-}"
  RUN_NAME+="_L${LAYER_DASHED}"
fi
EDITED_MODEL_DIR="${MODEL}_${RUN_NAME}"

# TSV:
TSV_DIR="eval_joblists"
mkdir -p "$TSV_DIR"
TSV_FILE="${TSV_DIR}/${RUN_NAME}.tsv"
printf "%s\t%s\t%s\n" "$RUN_NAME" "$EDITED_MODEL_DIR" "$DATA_TYPE" > "$TSV_FILE"

echo "=== Derived names ==="
echo "FT_ENABLED:        $FT_ENABLED"
echo "RUN_NAME:          $RUN_NAME"
echo "EDITED_MODEL_DIR:  $EDITED_MODEL_DIR"
echo "TSV:               $TSV_FILE"
echo

# ---- Submit training ----
EXPORTS="ALL,DATA_TYPE=${DATA_TYPE},ENERGY_THRESHOLD=${ENERGY_THRESHOLD},CACHE_SAMPLE_NUM=${CACHE_SAMPLE_NUM}"

if [[ -n "${NUM_EDITS}" ]]; then
  EXPORTS+=",NUM_EDITS=${NUM_EDITS}"
fi

if [[ -n "${RECALC_WEIGHT_THR}" ]]; then
  EXPORTS+=",RECALC_WEIGHT_THR=${RECALC_WEIGHT_THR}"
fi

# Enable FT in the sbatch environment (your snap.sbatch should map this to --no_snap)
if [[ "$FT_ENABLED" -eq 1 ]]; then
  EXPORTS+=",FT=1,NO_SNAP=1"
fi
if [[ -n "${LAYERS:-}" ]]; then
  # quote the comma-containing value so Slurm doesn't split it
  EXPORTS+=",LAYERS='${LAYERS}'"
fi

TRAIN_JOBID="$(sbatch --parsable --export="${EXPORTS}" snap.sbatch)"
echo "Submitted training job: ${TRAIN_JOBID}"
echo "Final command: sbatch --parsable --export="${EXPORTS}" snap.sbatch"

# ---- Submit eval after training completes successfully ----
# You currently do --array=0-0 manually; keep that behavior:
EVAL_JOBID="$(sbatch --parsable --dependency=afterok:${TRAIN_JOBID} --array=0-1 run_edited_benchmarks_prompttype_array.sbatch "${TSV_FILE}")"
echo "Submitted  edit eval job: ${EVAL_JOBID} (afterok:${TRAIN_JOBID})"
# EVAL_JOBID="$(sbatch --parsable --dependency=afterok:${TRAIN_JOBID} --array=0-0 run_edited_benchmarks_QA_array.sbatch "${TSV_FILE}")"
# echo "Submitted QA context edit eval job: ${EVAL_JOBID} (afterok:${TRAIN_JOBID})"
EVAL_JOBID="$(sbatch --parsable --dependency=afterok:${TRAIN_JOBID} --array=0-0 run_base_benchmarks_array.sbatch "${TSV_FILE}")"
echo "Submitted base eval job: ${EVAL_JOBID} (afterok:${TRAIN_JOBID})"
