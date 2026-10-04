# Final evaluation readiness checklist

Complete these checks before starting paid runs. The source-level evidence is listed; runtime boxes should be checked only after executing the commands in the target environment.

- [ ] Install and test: `python -m pip install -e '.[all]' && python -m pytest -q`.
- [ ] Build the evaluator: `docker build -f docker/evaluator.Dockerfile -t facets-evaluator:revised-v1 .`.
- [ ] Run `facets doctor --config experiments/final-study.yaml` and resolve every error.
- [ ] Confirm all intended provider IDs/tags. In particular, edit unresolved `claude-haiku-4-6` and `qwen3.6`; verify account access for `gpt-5.4` and `gpt-5-codex`. Do not substitute models silently.
- [ ] Confirm required keys and local Ollama/embedding models are present; doctor never downloads them.
- [ ] Confirm the smoke manifest records 345 derived records per model before limits/samples, six dataset hashes, `revised-v1`, and one frozen profile hash.
- [ ] Confirm all sweep manifests share dataset, prompt, evaluator, and profile hashes.

Objective source/test evidence:

- RTS invocation: `reference_test_success.py` imports separate candidate/reference modules and explicitly calls `_facets_reference_module.check(_candidate)`; `test_reference_test_success.py` covers correct/wrong/missing/empty/syntax/exception/timeout/multiline/definition-only cases.
- Actual-candidate generated tests: `generated_test_effectiveness.py` requires `candidate` imports or wraps the complete HumanEval checker, rejects shadowing, records pytest node IDs, and detects per-ID regressions. Controlled tests cover strong/weak/wrong/empty/collection/multiline/parameterized/mutation/unequal-ID/infrastructure cases.
- Workload runtime: `runtime_analysis.py` executes the same harness for trusted baseline and candidate, gates on expected output, times repeated in-child workload execution, retains samples/ratios, and tests formulas without flaky timing thresholds.
- Consistent aggregation: `revised-v1.yaml` is the single weight/formula profile; loader rejects non-unit sums; infrastructure errors make the run incomplete; functional pass@k uses actual functional metadata.
- Reproducibility: each response precedes evaluation on disk; JSONL results are fsynced; checkpoints/manifests are atomic and include config/Git/profile/prompt/dataset/runtime/tool provenance.
- Resume: orchestration tests interrupt after response saving and assert resume makes zero provider calls and appends no duplicate sample.

Small live smoke test (paid for hosted models):

```bash
facets run claude-sonnet-4-6 \
  --config experiments/final-study.yaml \
  --limit 2 \
  --run-name sonnet-smoke
```

Inspect `manifest.json`, `summary.json`, `results.jsonl`, and `responses/`, then resume once to confirm no duplicate calls:

```bash
facets resume reports/runs/sonnet-smoke
```

Complete sweep only after smoke inspection:

```bash
facets sweep --config experiments/final-study.yaml --dry-run
facets sweep --config experiments/final-study.yaml --resume
facets summarize reports/runs
facets analyze reports/runs --output reports/analysis/revised-v1
```

Supply: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, exact available hosted IDs, installed Ollama tags, a running Ollama service at the configured URL, `nomic-embed-text:latest` (or an edited embedding model), Docker/Podman, and the built evaluator image.

