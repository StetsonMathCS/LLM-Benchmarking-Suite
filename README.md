# FACETS

FACETS evaluates language models on six Python-focused software-engineering tasks: bug fixing, code generation, code review, refactoring, test generation, and translation. The revised evaluator emphasizes observed correctness, reproducible configuration, and response-first resumability.

The corrected evaluator is `revised-v1`; its report schema is `2.0`. Do not combine its scores with legacy reports.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all]'
docker build -f docker/evaluator.Dockerfile -t facets-evaluator:revised-v1 .
```

The container backend is the real-run default. It disables networking, uses a read-only root/work mount, drops capabilities, and limits memory, CPU, and process count. `execution.backend: local` is available only for controlled offline fixtures; a subprocess is not a filesystem or network sandbox.

## Configure

Edit [`experiments/final-study.yaml`](experiments/final-study.yaml) and [`config/models.yaml`](config/models.yaml). No secrets belong in either file.

```bash
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
```

Ollama uses the configured local base URL and exact installed tag. The registry preserves aliases for all 13 historically evaluated models. `claude-haiku-4-6` and `qwen3.6` are deliberately marked unresolved because no matching official identifier was verified; `gpt-5.4` and `gpt-5-codex` require an account capability check. FACETS will not replace them with another model.

CLI values override YAML only when explicitly supplied; omitted CLI values preserve YAML values. Provider extras use `--provider-extra key=value` and unsupported keys fail preflight.

## Preflight and run

```bash
facets --help
facets models list
facets doctor --config experiments/final-study.yaml
facets run claude-sonnet-4-6 --config experiments/final-study.yaml --dry-run
facets run claude-sonnet-4-6 --config experiments/final-study.yaml --limit 2 --run-name sonnet-smoke
facets run claude-sonnet-4-6 --config experiments/final-study.yaml --run-name sonnet-final
facets run --provider anthropic --model EXACT_PROVIDER_ID --config experiments/final-study.yaml
facets run --provider ollama --model 'mistral:7b' --config experiments/final-study.yaml
facets sweep --config experiments/final-study.yaml --dry-run
facets sweep --config experiments/final-study.yaml --resume
facets resume reports/runs/sonnet-smoke
```

`doctor` and `--dry-run` do not infer, pull models, or download weights. They validate profiles, CSV schemas/IDs, derived counts and hashes, required tools, the configured container image, credentials, embedding service, and model-registry status. A live provider check is never implicit.

Runs are noninteractive. Each response is atomically saved before evaluation, results are durably appended, and checkpoints are atomic. Resume reuses saved responses after an evaluation crash and skips completed identities. A changed model, dataset, sample, evaluator, or nonsecret config produces a different fingerprint and cannot be mixed into an existing run.

`run_experiment.py` is a compatibility entry point to the same CLI:

```bash
python run_experiment.py run claude-sonnet-4-6 --config experiments/final-study.yaml --dry-run
```

## Reevaluate and analyze

```bash
facets reevaluate OLD_REPORT_OR_RUN_DIRECTORY --profile revised-v1 --output reports/reevaluated/MODEL
facets summarize reports/runs
facets analyze reports/runs --output reports/analysis/revised-v1
```

Reevaluation never calls a generation provider. It can run deterministic code/static evaluators and the explicitly configured embedding service. Legacy repr fields are parsed with a restricted AST adapter, never `eval`; unrecoverable responses stop reevaluation instead of triggering generation.

Analysis exports tidy record/dimension CSVs, task/model summaries, correctness-conditioned quality, raw runtime ratios, prespecified weight sensitivity/rank changes, rank correlations, paired record-level bootstrap intervals, JSON summaries, and charts. Incompatible evaluator/profile/dataset manifests are rejected unless an explicitly descriptive mixed analysis is requested. Bootstrap intervals describe benchmark-record variation, not repeated-inference uncertainty or guaranteed generalization.

## Revised dimensions

| Stable ID | Display name | Interpretation |
|---|---|---|
| `reference_test_success` | Reference Test Success (RTS) | Binary success after explicitly invoking the complete trusted `check(candidate)` |
| `generated_test_effectiveness` | Generated Test Effectiveness (GTE) | generated-test validity × trusted-mutant detection |
| `structural_similarity` | Structural Similarity (SS) | structural heuristic; not semantic equivalence |
| `reference_review_similarity` | Reference Review Similarity (RRS) | reference-review embedding similarity; not expert-review accuracy |
| `functional_correctness` | Functional Correctness | observed behavior on supplied tests/inputs |
| `code_consistency` | Code Consistency | style heuristic |
| `linting` | Linting | findings from the named lint tool |
| `vulnerabilities` | Vulnerabilities | findings from Bandit/cppcheck/ESLint as recorded |
| `runtime_analysis` | Runtime Analysis | correctness-gated measured workload time/memory, not asymptotic complexity |

Exact formulas, weights, failure semantics, and limitations are in [`docs/METHODS.md`](docs/METHODS.md). Legacy handling is in [`docs/MIGRATION.md`](docs/MIGRATION.md), and final-sweep checks are in [`docs/READINESS.md`](docs/READINESS.md).

## Dataset inventory

Counts are derived by parsing the current CSVs; `doctor` records them and their hashes. The checked-in Python files currently contain 164 generation, 31 bug-fixing, 20 review, 20 refactoring, 80 test-generation, and 30 translation records (345 total). IDs and source rows are retained. Per-record license/source provenance is incomplete and is reported as such rather than inferred.

## Historical results — not revised-v1 results

The prior leaderboard and charts are preserved as historical artifacts under `reports/`. They are affected by evaluator/configuration defects: legacy CGT did not invoke HumanEval `check(candidate)`, legacy TPR inferred kills from aggregate pass counts, runtime often measured definitions/startup, and historical generation weights differ from the later matrix. These values must not be presented as verified functional results or mixed with revised-v1 rankings.

| Model | Provider | Historical score |
|---|---|---:|
| claude-opus-4.6 | Anthropic | 89.33% |
| claude-sonnet-4.6 | Anthropic | 88.40% |
| gpt-5-codex | OpenAI | 88.62% |
| gpt-5.4 | OpenAI | 88.11% |
| claude-haiku-4.6 | Anthropic | 86.85% |
| gpt-5-nano | OpenAI | 86.36% |
| deepseek-r1 | Ollama | 85.13% |
| qwen3.6 | Ollama | 85.20% |
| gpt-oss | Ollama | 83.24% |
| ministral-3 | Ollama | 79.48% |
| llama3 | Ollama | 79.29% |
| mistral | Ollama | 70.70% |
| phi4-mini | Ollama | 64.56% |

