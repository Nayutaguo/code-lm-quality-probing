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

PAPER_ROOT="${PAPER_ROOT:-artifacts/rq1_qwen2_5_coder_7b}"
VECTOR_DIR="${VECTOR_DIR:-${PAPER_ROOT}/vectors}"
REPORT_DIR="${REPORT_DIR:-${PAPER_ROOT}/reports}"
SCORE_DIR="${SCORE_DIR:-${PAPER_ROOT}/scores}"
FIGURE_DIR="${FIGURE_DIR:-${PAPER_ROOT}/figures}"

mkdir -p "${VECTOR_DIR}" "${REPORT_DIR}" "${SCORE_DIR}" "${FIGURE_DIR}"

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --layers "${LAYERS}"
  --seed "${SEED}"
  --dtype "${DTYPE}"
  --device-map "${DEVICE_MAP}"
  --truncation-side left
  --font-family "${FONT_FAMILY}"
)

echo "[1/6] RQ1 safety: train -> val"
"${PYTHON}" code/rq1_projection_margin.py \
  "${COMMON_ARGS[@]}" \
  --fit-data-files \
    "data/SafeCoder/data_train_val/train/sec-desc.jsonl" \
    "data/SafeCoder/data_train_val/train/sec-new-desc.jsonl" \
  --eval-data-files \
    "data/SafeCoder/data_train_val/val/sec-desc.jsonl" \
    "data/SafeCoder/data_train_val/val/sec-new-desc.jsonl" \
  --task-field description \
  --positive-field func_src_after \
  --negative-field func_src_before \
  --concept-name safety \
  --positive-name safe \
  --negative-name unsafe \
  --metadata-fields vul_type file_name func_name \
  --preference-question "Which implementation is safer and less likely to contain a vulnerability?" \
  --overflow-policy truncate \
  --fit-output "${VECTOR_DIR}/safety_rq1.fit.pt" \
  --output "${REPORT_DIR}/safety_rq1.json" \
  --csv-output "${SCORE_DIR}/safety_rq1.csv" \
  --summary-output "${SCORE_DIR}/safety_rq1.summary.json" \
  --figure-output "${FIGURE_DIR}/safety_rq1.png"

echo "[2/6] RQ1 efficiency: train -> val"
"${PYTHON}" code/rq1_projection_margin.py \
  "${COMMON_ARGS[@]}" \
  --fit-data-files "data/efficodebench/processed/efficodebench_ab_train.jsonl" \
  --eval-data-files "data/efficodebench/processed/efficodebench_ab_val.jsonl" \
  --task-field description \
  --positive-field fast_code \
  --negative-field slow_code \
  --concept-name efficiency \
  --positive-name fast \
  --negative-name slow \
  --metadata-fields language pair_id base_pair_id problem_id split source_file source_row_index original_output original_direction speedup_ratio output_score fast_original_side slow_original_side \
  --preference-question "Which implementation is faster at runtime and should be preferred?" \
  --overflow-policy truncate \
  --fit-output "${VECTOR_DIR}/efficiency_rq1.fit.pt" \
  --output "${REPORT_DIR}/efficiency_rq1_val.json" \
  --csv-output "${SCORE_DIR}/efficiency_rq1_val.csv" \
  --summary-output "${SCORE_DIR}/efficiency_rq1_val.summary.json" \
  --figure-output "${FIGURE_DIR}/efficiency_rq1_val.png"

echo "[3/6] RQ1 correctness: train -> test"
"${PYTHON}" code/rq1_projection_margin.py \
  "${COMMON_ARGS[@]}" \
  --fit-data-files "data/Codeflaws/processed/codeflaws_correctness_train.jsonl" \
  --eval-data-files "data/Codeflaws/processed/codeflaws_correctness_test.jsonl" \
  --task-field description \
  --positive-field correct_code \
  --negative-field incorrect_code \
  --concept-name codeflaws_correctness \
  --positive-name correct \
  --negative-name incorrect \
  --metadata-fields language dataset pair_id base_pair_id problem_id contest_id problem_index buggy_submission_id accepted_submission_id subject_dir incorrect_file correct_file defect_class verdict defect_tags repair_test_count heldout_test_count has_test_genprog has_test_valid split split_mode \
  --preference-question "Which implementation is more likely to be correct and pass the tests?" \
  --overflow-policy truncate \
  --fit-output "${VECTOR_DIR}/correctness_rq1.fit.pt" \
  --output "${REPORT_DIR}/correctness_rq1.json" \
  --csv-output "${SCORE_DIR}/correctness_rq1.csv" \
  --summary-output "${SCORE_DIR}/correctness_rq1.summary.json" \
  --figure-output "${FIGURE_DIR}/correctness_rq1.png"

echo "[4/6] RQ1 readability: train -> test"
"${PYTHON}" code/rq1_projection_margin.py \
  "${COMMON_ARGS[@]}" \
  --fit-data-files "data/TestCodeRefactoring/processed/test_smells_quality_train.jsonl" \
  --eval-data-files "data/TestCodeRefactoring/processed/test_smells_quality_test.jsonl" \
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
  --fit-output "${VECTOR_DIR}/test_smells_quality_rq1.fit.pt" \
  --output "${REPORT_DIR}/test_smells_quality_rq1.json" \
  --csv-output "${SCORE_DIR}/test_smells_quality_rq1.csv" \
  --summary-output "${SCORE_DIR}/test_smells_quality_rq1.summary.json" \
  --figure-output "${FIGURE_DIR}/test_smells_quality_rq1.png"

echo "[5/6] plot combined RQ1 profile grid"
"${PYTHON}" code/plot_rq1_profiles_grid.py \
  --report-inputs \
    "${REPORT_DIR}/safety_rq1.json" \
    "${REPORT_DIR}/efficiency_rq1_val.json" \
    "${REPORT_DIR}/correctness_rq1.json" \
    "${REPORT_DIR}/test_smells_quality_rq1.json" \
  --labels Security Efficiency Correctness Readability \
  --columns 4 \
  --figure-width 14.8 \
  --figure-height 3.9 \
  --font-family "${FONT_FAMILY}" \
  --output "${FIGURE_DIR}/rq1_profiles_qwen2_5_coder_7b_1x4.pdf"

echo "[6/6] render png preview"
"${PYTHON}" code/plot_rq1_profiles_grid.py \
  --report-inputs \
    "${REPORT_DIR}/safety_rq1.json" \
    "${REPORT_DIR}/efficiency_rq1_val.json" \
    "${REPORT_DIR}/correctness_rq1.json" \
    "${REPORT_DIR}/test_smells_quality_rq1.json" \
  --labels Security Efficiency Correctness Readability \
  --columns 4 \
  --figure-width 14.8 \
  --figure-height 3.9 \
  --font-family "${FONT_FAMILY}" \
  --output "${FIGURE_DIR}/rq1_profiles_qwen2_5_coder_7b_1x4_preview.png"

echo "done"
