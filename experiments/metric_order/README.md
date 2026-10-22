# Metric order changes scores for frozen answers

This experiment accompanies [LM Evaluation Harness #4449](https://github.com/EleutherAI/lm-evaluation-harness/pull/4449). It replays synthetic prediction text through the real `ConfigurableTask.process_results` and `_compute_task_aggregations` methods. It does not run a language model.

For a task with targets `["answer", "alias"]`, the three frozen predictions are `"answer"`, `"alias"`, and `"wrong"`. Two of three are correct. On the baseline, putting `exact_match` first changes the input received by a later text-equality metric from a string to a list. Its aggregate score falls from 2/3 to 0 solely because the metric order changed.

| Source | Metric order | exact_match | text_equal |
| --- | --- | ---: | ---: |
| Baseline | exact_match, text_equal | 0.666667 | 0 |
| Baseline | text_equal, exact_match | 0.666667 | 0.666667 |
| Fixed | exact_match, text_equal | 0.666667 | 0.666667 |
| Fixed | text_equal, exact_match | 0.666667 | 0.666667 |

The table rounds the raw floating-point values for display. The fixed source preserves the scalar prediction and constructs the repeated list only inside the `exact_match` call. No matching rules, normalization, thresholds, or aggregation rules change.

## Pinned comparison

- Baseline: [`d6de81643928d653435c431bae19945d41d32520`](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520)
- Fixed: [`80b17f4a32b1aebb12302a2262962376e9d29d80`](https://github.com/justavibedev/lm-evaluation-harness/tree/80b17f4a32b1aebb12302a2262962376e9d29d80)
- Python: 3.12.11; exact installed dependency versions: [requirements.txt](requirements.txt)
- Inputs: [fixtures.json](fixtures.json)
- Replay and assertions: [replay.py](replay.py)

The replay uses four target cases: two answers, those answers reversed, a singleton list, and a scalar. Each case uses both two-metric orders and each metric alone. Each configuration scores the same three frozen predictions. That is 16 configurations and 48 scoring calls per source version, repeated in two fresh processes per version. These are variations of one synthetic regression, not independent benchmark evaluations.

The singleton-list case also reproduces the order dependence. Scalar targets and single-metric configurations are controls. Every per-document score, recorded callback input, and aggregate is checked. Aggregation uses the library's mean implementation with bootstrap disabled; no confidence intervals are estimated.

## Reproduce

Clone this experiment branch and, when using a commit permalink, check out that experiment commit to preserve the exact replay and fixtures. Keep full history: both source commits must be available locally.

```bash
git clone --branch experiment/metric-order-replay-20261022 \
  https://github.com/justavibedev/lm-evaluation-harness.git
cd lm-evaluation-harness
uv venv --python 3.12.11 .venv-replay
uv pip install --python .venv-replay/bin/python \
  -r experiments/metric_order/requirements.txt
uv pip install --python .venv-replay/bin/python \
  --no-deps --no-build-isolation -e .
.venv-replay/bin/python experiments/metric_order/replay.py \
  --output /tmp/lm-eval-metric-order-replay
diff experiments/metric_order/results/baseline-1.jsonl \
  /tmp/lm-eval-metric-order-replay/baseline-1.jsonl
diff experiments/metric_order/results/fixed-1.jsonl \
  /tmp/lm-eval-metric-order-replay/fixed-1.jsonl
```

Choose a new output directory if it already exists. Do not use Python `-O`, which disables assertions. Setup requires obtaining the repository and Python dependencies. Scoring uses no model, tokenizer, dataset download, provider credential, or GPU.

Each child imports a separate `git archive` snapshot of the pinned package. The runner checks the origin of every loaded `lm_eval` module and the SHA-256 digest of the archived package. Child processes receive a fresh home directory and a minimal environment, with Hugging Face offline settings and a Python socket guard that blocks network connections. This guard is a reproducibility aid, not an operating-system security sandbox.

## Raw results

- [Baseline repeat 1](results/baseline-1.jsonl) and [repeat 2](results/baseline-2.jsonl)
- [Fixed repeat 1](results/fixed-1.jsonl) and [repeat 2](results/fixed-2.jsonl)
- [All aggregates](results/aggregates.json), computed from the raw scores
- [Environment, source pins, hashes, commands, and exit codes](results/environment.json)
- Per-process import identities and stdout/stderr are alongside those files.

Both repeats produce byte-identical JSONL within each source version. Results retain the metric order explicitly. Environment metadata is separate from deterministic observations and includes no local usernames or machine hostnames.

## Limits

This demonstrates an API scoring defect with a deliberately simple custom metric. It does not measure model quality, throughput, a leaderboard change, or the impact on a published evaluation. Source-only inspection did not confirm an affected bundled benchmark; dataset-dependent target types were not fetched or evaluated. The replay does not exercise generation, distributed evaluation, task groups, model integrations, or other metric semantics.

The dependency pins record one environment, not a cross-platform compatibility claim or a package security audit. The upstream PR remains subject to maintainer review. AI assistance was used for implementation, analysis, testing, and review.
