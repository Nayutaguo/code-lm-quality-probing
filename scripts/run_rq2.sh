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
LAYER="${LAYER:-20}"
RUN_JUDGE="${RUN_JUDGE:-0}"

RQ1_ROOT="${RQ1_ROOT:-artifacts/rq1_qwen2_5_coder_7b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/rq2_qwen2_5_coder_7b}"
REPORT_DIR="${REPORT_DIR:-${OUTPUT_ROOT}/reports}"

mkdir -p "${REPORT_DIR}"

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --dtype "${DTYPE}"
  --device-map "${DEVICE_MAP}"
  --seed "${SEED}"
)

echo "[1/5] RQ2 ours: safety @ layer ${LAYER}"
"${PYTHON}" code/rq2_binary_selection.py \
  "${COMMON_ARGS[@]}" \
  --vector-path "${RQ1_ROOT}/vectors/safety_rq1.fit.pt" \
  --data-files \
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
  --layer "${LAYER}" \
  --truncation-side left \
  --overflow-policy truncate \
  --output "${REPORT_DIR}/safety_train_to_val_rq2_layer${LAYER}.json"

echo "[2/5] RQ2 ours: efficiency @ layer ${LAYER}"
"${PYTHON}" code/rq2_binary_selection.py \
  "${COMMON_ARGS[@]}" \
  --vector-path "${RQ1_ROOT}/vectors/efficiency_rq1.fit.pt" \
  --data-files "data/efficodebench/processed/efficodebench_ab_test.jsonl" \
  --task-field description \
  --positive-field fast_code \
  --negative-field slow_code \
  --concept-name efficiency \
  --positive-name fast \
  --negative-name slow \
  --metadata-fields language pair_id base_pair_id problem_id split source_file source_row_index original_output original_direction speedup_ratio output_score fast_original_side slow_original_side \
  --preference-question "Which implementation is faster at runtime and should be preferred?" \
  --layer "${LAYER}" \
  --max-length 8192 \
  --truncation-side left \
  --overflow-policy drop \
  --output "${REPORT_DIR}/efficiency_train_to_test_rq2_layer${LAYER}_max8192.json"

echo "[3/5] RQ2 ours: correctness @ layer ${LAYER}"
"${PYTHON}" code/rq2_binary_selection.py \
  "${COMMON_ARGS[@]}" \
  --vector-path "${RQ1_ROOT}/vectors/correctness_rq1.fit.pt" \
  --data-files "data/Codeflaws/processed/codeflaws_correctness_test.jsonl" \
  --task-field description \
  --positive-field correct_code \
  --negative-field incorrect_code \
  --concept-name codeflaws_correctness \
  --positive-name correct \
  --negative-name incorrect \
  --metadata-fields language dataset pair_id base_pair_id problem_id contest_id problem_index buggy_submission_id accepted_submission_id subject_dir incorrect_file correct_file defect_class verdict defect_tags repair_test_count heldout_test_count has_test_genprog has_test_valid split split_mode \
  --preference-question "Which implementation is more likely to be correct and pass the tests?" \
  --layer "${LAYER}" \
  --truncation-side left \
  --overflow-policy truncate \
  --output "${REPORT_DIR}/correctness_train_to_test_rq2_layer${LAYER}.json"

echo "[4/5] RQ2 ours: readability @ layer ${LAYER}"
"${PYTHON}" code/rq2_binary_selection.py \
  "${COMMON_ARGS[@]}" \
  --vector-path "${RQ1_ROOT}/vectors/test_smells_quality_rq1.fit.pt" \
  --data-files "data/TestCodeRefactoring/processed/test_smells_quality_test.jsonl" \
  --task-field description \
  --positive-field clean_code \
  --negative-field smelly_code \
  --concept-name test_smells_quality \
  --positive-name clean \
  --negative-name smelly \
  --metadata-fields language dataset instance_id project sha date file_name_before file_name_after refactoring test_smell split split_mode \
  --preference-question "Which Java test implementation is cleaner, less smelly, and easier to maintain?" \
  --layer "${LAYER}" \
  --max-length 4096 \
  --truncation-side left \
  --overflow-policy truncate \
  --output "${REPORT_DIR}/test_smells_quality_train_to_test_rq2_layer${LAYER}.json"

if [[ "${RUN_JUDGE}" == "1" ]]; then
  echo "[5/5] LLM judge baselines"
  "${PYTHON}" code/llm_judge_binary_selection.py \
    "${COMMON_ARGS[@]}" \
    --data-files \
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
    --max-new-tokens 8 \
    --truncation-side left \
    --overflow-policy truncate \
    --output "${REPORT_DIR}/safety_train_to_val_llm_judge.json"

  "${PYTHON}" code/llm_judge_binary_selection.py \
    "${COMMON_ARGS[@]}" \
    --data-files "data/efficodebench/processed/efficodebench_ab_test.jsonl" \
    --task-field description \
    --positive-field fast_code \
    --negative-field slow_code \
    --concept-name efficiency \
    --positive-name fast \
    --negative-name slow \
    --metadata-fields language pair_id base_pair_id problem_id split source_file source_row_index original_output original_direction speedup_ratio output_score fast_original_side slow_original_side \
    --preference-question "Which implementation is faster at runtime and should be preferred?" \
    --max-length 8192 \
    --max-new-tokens 8 \
    --truncation-side left \
    --overflow-policy drop \
    --output "${REPORT_DIR}/efficiency_train_to_test_max8192_llm_judge.json"

  "${PYTHON}" code/llm_judge_binary_selection.py \
    "${COMMON_ARGS[@]}" \
    --data-files "data/Codeflaws/processed/codeflaws_correctness_test.jsonl" \
    --task-field description \
    --positive-field correct_code \
    --negative-field incorrect_code \
    --concept-name codeflaws_correctness \
    --positive-name correct \
    --negative-name incorrect \
    --metadata-fields language dataset pair_id base_pair_id problem_id contest_id problem_index buggy_submission_id accepted_submission_id subject_dir incorrect_file correct_file defect_class verdict defect_tags repair_test_count heldout_test_count has_test_genprog has_test_valid split split_mode \
    --preference-question "Which implementation is more likely to be correct and pass the tests?" \
    --max-new-tokens 8 \
    --truncation-side left \
    --overflow-policy truncate \
    --output "${REPORT_DIR}/correctness_train_to_test_llm_judge.json"

  "${PYTHON}" code/llm_judge_binary_selection.py \
    "${COMMON_ARGS[@]}" \
    --data-files "data/TestCodeRefactoring/processed/test_smells_quality_test.jsonl" \
    --task-field description \
    --positive-field clean_code \
    --negative-field smelly_code \
    --concept-name test_smells_quality \
    --positive-name clean \
    --negative-name smelly \
    --metadata-fields language dataset instance_id project sha date file_name_before file_name_after refactoring test_smell split split_mode \
    --preference-question "Which Java test implementation is cleaner, less smelly, and easier to maintain?" \
    --max-length 4096 \
    --max-new-tokens 8 \
    --truncation-side left \
    --overflow-policy truncate \
    --output "${REPORT_DIR}/test_smells_quality_train_to_test_llm_judge.json"

  "${PYTHON}" code/summarize_rq2_binary_selection.py \
    --dimensions Safety Efficiency Correctness Readability \
    --ours-reports \
      "${REPORT_DIR}/safety_train_to_val_rq2_layer${LAYER}.json" \
      "${REPORT_DIR}/efficiency_train_to_test_rq2_layer${LAYER}_max8192.json" \
      "${REPORT_DIR}/correctness_train_to_test_rq2_layer${LAYER}.json" \
      "${REPORT_DIR}/test_smells_quality_train_to_test_rq2_layer${LAYER}.json" \
    --judge-reports \
      "${REPORT_DIR}/safety_train_to_val_llm_judge.json" \
      "${REPORT_DIR}/efficiency_train_to_test_max8192_llm_judge.json" \
      "${REPORT_DIR}/correctness_train_to_test_llm_judge.json" \
      "${REPORT_DIR}/test_smells_quality_train_to_test_llm_judge.json" \
    --output-json "${OUTPUT_ROOT}/rq2_layer${LAYER}_summary_with_ci.json" \
    --output-csv "${OUTPUT_ROOT}/rq2_layer${LAYER}_summary_with_ci.csv"
else
  echo "[5/5] skipped LLM judge (RUN_JUDGE=${RUN_JUDGE})"
  echo "Only our layer-${LAYER} RQ2 reports were produced."
fi

echo "done"
