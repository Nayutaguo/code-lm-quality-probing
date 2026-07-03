# Hidden-State Code Quality Experiments

Minimal release for the main RQ1 and RQ2 experiments on four code-quality dimensions:

- `SafeCoder` for security
- `efficodebench` for efficiency
- `Codeflaws` for correctness
- `TestCodeRefactoring` for readability

This curated version keeps only:

- processed datasets under [data](/mnt/sda/yz/gzh/steeringvector%20for%20code/data)
- core experiment code under [code](/mnt/sda/yz/gzh/steeringvector%20for%20code/code)
- three entry scripts under [scripts](/mnt/sda/yz/gzh/steeringvector%20for%20code/scripts)
- retained RQ1/RQ2 results and the final paper figures under [artifacts](/mnt/sda/yz/gzh/steeringvector%20for%20code/artifacts)

Main scripts:

- [scripts/run_rq1.sh](/mnt/sda/yz/gzh/steeringvector%20for%20code/scripts/run_rq1.sh)
- [scripts/run_rq1_layer_scores.sh](/mnt/sda/yz/gzh/steeringvector%20for%20code/scripts/run_rq1_layer_scores.sh)
- [scripts/run_rq2.sh](/mnt/sda/yz/gzh/steeringvector%20for%20code/scripts/run_rq2.sh)

Default settings target `Qwen2.5-Coder-7B-Instruct`. Override with environment variables when needed, for example:

```bash
MODEL_PATH=/path/to/model bash scripts/run_rq1.sh
MODEL_PATH=/path/to/model bash scripts/run_rq2.sh
```

Current retained result folders:

- [artifacts/rq1_llama3_1_8b](/mnt/sda/yz/gzh/steeringvector%20for%20code/artifacts/rq1_llama3_1_8b)
- [artifacts/rq1_qwen2_5_coder_7b](/mnt/sda/yz/gzh/steeringvector%20for%20code/artifacts/rq1_qwen2_5_coder_7b)
- [artifacts/rq2_llama3_1_8b](/mnt/sda/yz/gzh/steeringvector%20for%20code/artifacts/rq2_llama3_1_8b)
- [artifacts/rq2_qwen2_5_coder_7b](/mnt/sda/yz/gzh/steeringvector%20for%20code/artifacts/rq2_qwen2_5_coder_7b)
- [artifacts/rq1_layerwise_figures](/mnt/sda/yz/gzh/steeringvector%20for%20code/artifacts/rq1_layerwise_figures)

Only one main layer-wise figure layout and one main Llama-3.1-8B RQ1 profile layout are retained; duplicate previews, alternative panel layouts, per-dimension figure variants, and intermediate vector artifacts were removed.

Exploratory scripts, intermediate samples, case-study notes, and unrelated benchmarks were removed to keep the repository submission-focused.
