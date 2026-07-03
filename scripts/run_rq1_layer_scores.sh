#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" && -f "/mnt/sda/yz/gzh/steering_vector/.venv/bin/activate" ]]; then
  source "/mnt/sda/yz/gzh/steering_vector/.venv/bin/activate"
elif [[ -z "${VIRTUAL_ENV:-}" && -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/mnt/sda/yz/projects/models/Qwen2.5-Coder-7B-Instruct}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
SEED="${SEED:-13}"
LAYERS="${LAYERS:-blocks}"
FONT_FAMILY="${FONT_FAMILY:-Arial}"
SCORE_PLUGIN="${SCORE_PLUGIN:-layernavigator}"

PAPER_ROOT="${PAPER_ROOT:-artifacts/rq1_qwen2_5_coder_7b_layerwise}"
VECTOR_DIR="${VECTOR_DIR:-${PAPER_ROOT}/vectors}"
FIGURE_DIR="${FIGURE_DIR:-${PAPER_ROOT}/figures}"
JSON_DIR="${JSON_DIR:-${PAPER_ROOT}/json}"
CSV_DIR="${CSV_DIR:-${PAPER_ROOT}/csv}"

mkdir -p "${VECTOR_DIR}" "${FIGURE_DIR}" "${JSON_DIR}" "${CSV_DIR}"

COMMON_BUILD_ARGS=(
  --model-path "${MODEL_PATH}"
  --layers "${LAYERS}"
  --seed "${SEED}"
  --dtype "${DTYPE}"
  --device-map "${DEVICE_MAP}"
  --truncation-side left
  --score-plugin "${SCORE_PLUGIN}"
)

echo "[1/10] build scored safety vectors"
"${PYTHON}" code/build_answer_token_vectors.py \
  "${COMMON_BUILD_ARGS[@]}" \
  --data-files \
    "data/SafeCoder/data_train_val/train/sec-desc.jsonl" \
    "data/SafeCoder/data_train_val/train/sec-new-desc.jsonl" \
  --task-field description \
  --positive-field func_src_after \
  --negative-field func_src_before \
  --concept-name safety \
  --positive-name safe \
  --negative-name unsafe \
  --metadata-fields vul_type file_name func_name \
  --preference-question "Which implementation is safer and less likely to contain a vulnerability?" \
  --overflow-policy truncate \
  --output "${VECTOR_DIR}/safety_train_answer_token_vectors.pt"

echo "[2/10] build scored efficiency vectors"
"${PYTHON}" code/build_answer_token_vectors.py \
  "${COMMON_BUILD_ARGS[@]}" \
  --data-files "data/efficodebench/processed/efficodebench_ab_train.jsonl" \
  --task-field description \
  --positive-field fast_code \
  --negative-field slow_code \
  --concept-name efficiency \
  --positive-name fast \
  --negative-name slow \
  --metadata-fields language pair_id base_pair_id problem_id split source_file source_row_index original_output original_direction speedup_ratio output_score fast_original_side slow_original_side \
  --preference-question "Which implementation is faster at runtime and should be preferred?" \
  --overflow-policy truncate \
  --output "${VECTOR_DIR}/efficiency_train_answer_token_vectors.pt"

echo "[3/10] build scored correctness vectors"
"${PYTHON}" code/build_answer_token_vectors.py \
  "${COMMON_BUILD_ARGS[@]}" \
  --data-files "data/Codeflaws/processed/codeflaws_correctness_train.jsonl" \
  --task-field description \
  --positive-field correct_code \
  --negative-field incorrect_code \
  --concept-name codeflaws_correctness \
  --positive-name correct \
  --negative-name incorrect \
  --metadata-fields language dataset pair_id base_pair_id problem_id contest_id problem_index buggy_submission_id accepted_submission_id subject_dir incorrect_file correct_file defect_class verdict defect_tags repair_test_count heldout_test_count has_test_genprog has_test_valid split split_mode \
  --preference-question "Which implementation is more likely to be correct and pass the tests?" \
  --overflow-policy truncate \
  --output "${VECTOR_DIR}/correctness_train_answer_token_vectors.pt"

echo "[4/10] build scored readability vectors"
"${PYTHON}" code/build_answer_token_vectors.py \
  "${COMMON_BUILD_ARGS[@]}" \
  --data-files "data/TestCodeRefactoring/processed/test_smells_quality_train.jsonl" \
  --task-field description \
  --positive-field clean_code \
  --negative-field smelly_code \
  --concept-name test_smells_quality \
  --positive-name clean \
  --negative-name smelly \
  --metadata-fields language dataset instance_id project sha date file_name_before file_name_after refactoring test_smell split split_mode \
  --preference-question "Which Java test implementation is cleaner, less smelly, and easier to maintain?" \
  --max-length 4096 \
  --overflow-policy truncate \
  --output "${VECTOR_DIR}/test_smells_quality_train_answer_token_vectors.pt"

echo "[5/10] plot combined 1x4 stacked bars (pdf)"
"${PYTHON}" code/plot_layer_score_trends.py \
  --vector-paths \
    "${VECTOR_DIR}/safety_train_answer_token_vectors.pt" \
    "${VECTOR_DIR}/efficiency_train_answer_token_vectors.pt" \
    "${VECTOR_DIR}/correctness_train_answer_token_vectors.pt" \
    "${VECTOR_DIR}/test_smells_quality_train_answer_token_vectors.pt" \
  --estimators answer_token_diff answer_token_diff answer_token_diff answer_token_diff \
  --labels Security Efficiency Correctness Readability \
  --plugin "${SCORE_PLUGIN}" \
  --metric s_score \
  --style bars-grid \
  --grid-columns 4 \
  --figure-width 14.8 \
  --figure-height 3.9 \
  --font-family "${FONT_FAMILY}" \
  --x-tick-step 2 \
  --output "${FIGURE_DIR}/rq1_layer_scores_qwen2_5_coder_7b_1x4.pdf" \
  --json-output "${JSON_DIR}/rq1_layer_scores_qwen2_5_coder_7b_1x4.json" \
  --csv-output "${CSV_DIR}/rq1_layer_scores_qwen2_5_coder_7b_1x4.csv"

echo "[6/10] plot combined 1x4 stacked bars (png preview)"
"${PYTHON}" code/plot_layer_score_trends.py \
  --json-input "${JSON_DIR}/rq1_layer_scores_qwen2_5_coder_7b_1x4.json" \
  --labels Security Efficiency Correctness Readability \
  --style bars-grid \
  --grid-columns 4 \
  --figure-width 14.8 \
  --figure-height 3.9 \
  --font-family "${FONT_FAMILY}" \
  --x-tick-step 2 \
  --output "${FIGURE_DIR}/rq1_layer_scores_qwen2_5_coder_7b_1x4_preview.png"

echo "[7/10] plot safety decomposition"
"${PYTHON}" code/plot_layer_score_trends.py \
  --vector-paths "${VECTOR_DIR}/safety_train_answer_token_vectors.pt" \
  --estimators answer_token_diff \
  --labels Security \
  --plugin "${SCORE_PLUGIN}" \
  --metric s_score \
  --style decomposition \
  --figure-width 8.8 \
  --figure-height 6.6 \
  --font-family "${FONT_FAMILY}" \
  --x-tick-step 2 \
  --output "${FIGURE_DIR}/safety_layer_scores_decomp.png"

echo "[8/10] plot efficiency decomposition"
"${PYTHON}" code/plot_layer_score_trends.py \
  --vector-paths "${VECTOR_DIR}/efficiency_train_answer_token_vectors.pt" \
  --estimators answer_token_diff \
  --labels Efficiency \
  --plugin "${SCORE_PLUGIN}" \
  --metric s_score \
  --style decomposition \
  --figure-width 8.8 \
  --figure-height 6.6 \
  --font-family "${FONT_FAMILY}" \
  --x-tick-step 2 \
  --output "${FIGURE_DIR}/efficiency_layer_scores_decomp.png"

echo "[9/10] plot correctness decomposition"
"${PYTHON}" code/plot_layer_score_trends.py \
  --vector-paths "${VECTOR_DIR}/correctness_train_answer_token_vectors.pt" \
  --estimators answer_token_diff \
  --labels Correctness \
  --plugin "${SCORE_PLUGIN}" \
  --metric s_score \
  --style decomposition \
  --figure-width 8.8 \
  --figure-height 6.6 \
  --font-family "${FONT_FAMILY}" \
  --x-tick-step 2 \
  --output "${FIGURE_DIR}/correctness_layer_scores_decomp.png"

echo "[10/10] plot readability decomposition"
"${PYTHON}" code/plot_layer_score_trends.py \
  --vector-paths "${VECTOR_DIR}/test_smells_quality_train_answer_token_vectors.pt" \
  --estimators answer_token_diff \
  --labels Readability \
  --plugin "${SCORE_PLUGIN}" \
  --metric s_score \
  --style decomposition \
  --figure-width 8.8 \
  --figure-height 6.6 \
  --font-family "${FONT_FAMILY}" \
  --x-tick-step 2 \
  --output "${FIGURE_DIR}/readability_layer_scores_decomp.png"

echo "done"
