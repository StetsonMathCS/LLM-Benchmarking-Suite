# CLI reference

Every `facets` subcommand, flag by flag. Verified against `facets/cli.py`.

Entry point is `facets` (installed by `pip install -e .`). `run_experiment.py` is a
compatibility wrapper that calls the same `main()`, so `python run_experiment.py <args>`
accepts everything below.

For config key semantics see [`EXPERIMENTS.md`](EXPERIMENTS.md); for score
interpretation see [`METHODS.md`](METHODS.md).

## Contents

- [Setup](#setup)
- [Global conventions](#global-conventions)
- [`facets models list`](#facets-models-list)
- [`facets doctor`](#facets-doctor)
- [`facets run`](#facets-run)
- [`facets resume`](#facets-resume)
- [`facets sweep`](#facets-sweep)
- [`facets queue`](#facets-queue)
- [`facets reevaluate`](#facets-reevaluate)
- [`facets summarize`](#facets-summarize)
- [`facets analyze`](#facets-analyze)
- [Exit codes](#exit-codes)
- [Which command to use](#which-command-to-use)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[all]'
docker build -f docker/evaluator.Dockerfile -t facets-evaluator:revised-v1 .
```

The container image is required: `execution.backend` defaults to `container` and
doctor errors if the image is absent.

All configured models are local Ollama models and need no API keys. Every registry entry
carries its own `base_url`; nothing has to be exported.

## Global conventions

**CLI overrides YAML only when explicitly supplied.** Every flag defaults to `None`;
`merge_cli` (`facets/config.py:35`) skips `None` values, so an omitted flag preserves the
YAML value. There is no way to clear a YAML value from the CLI except `--provider-extra`,
which replaces the whole map.

**Task ids** are exactly: `bug_fixing`, `code_generation`, `code_review`, `refactoring`,
`test_generation`, `translation`.

**Registry** defaults to `config/models.yaml`. Every command that resolves a model takes
`--registry PATH` to point elsewhere.

**Output is JSON** on stdout, pretty-printed and key-sorted. Diagnostics go to stderr.
Events during a queue run are streamed as one JSON object per line, so you can pipe
through `jq`.

**Two `--dry-run` behaviours.** On `run` and `sweep` it means "run the full doctor
preflight and stop". On `queue run` it means "report what would run and stop" using already
frozen queue state. They are not the same check.

## `facets models list`

Prints the model registry. Takes no positionals.

```bash
facets models list
facets models list --registry config/models.yaml
```

`--registry PATH` — alternate registry file.

Availability flags (`unresolved`, `verify_with_provider`) are recorded for the reader but
never used to substitute a different model.

## `facets doctor`

Offline preflight. Validates config, dataset CSVs and derived counts, credential presence,
evaluator tools, the container image, and the embedding service. Makes no generation calls.

```bash
facets doctor --config experiments/final-study.yaml
```

| Flag | Default | Notes |
|---|---|---|
| `--config PATH` | required | Experiment YAML. |
| `--registry PATH` | `config/models.yaml` | Model registry. |
| `--live-provider-check` | off | Contacts each provider to verify the model id is reachable. |

Every model in the config's `models:` list is doctored. The first runs with
`mutation_preflight: full`; the rest get `already_validated_for_shared_dataset` so the
shared dataset is validated once, not once per model (`facets/cli.py:359`).

Exit `0` if all models pass, `2` if any fails.

**Never downloads, pulls a model, or contacts a provider** unless
`--live-provider-check` is passed.

## `facets run`

One model, one run directory, no queue.

```bash
facets run qwen3.8 --config experiments/final-study.yaml --dry-run
facets run qwen3.8 --config experiments/final-study.yaml --limit 2 --run-name qwen38-smoke
facets run --provider ollama --model 'qwen3.8:27b' --config experiments/final-study.yaml
```

### Model selection

Positional `alias` (optional) resolves through the registry's `aliases` map to a canonical
entry. Alias chains are followed; cycles are rejected. Alternatively name the provider and
exact model id directly, bypassing the registry.

| Flag | Notes |
|---|---|
| `alias` (positional, optional) | Registry alias or canonical key. |
| `--provider NAME` | `anthropic`, `openai`, or `ollama`. Use with `--model`. |
| `--model ID` | Exact provider identifier. Use with `--provider`. |

### Config overrides

| Flag | YAML key | Notes |
|---|---|---|
| `--config PATH` | — | Required. |
| `--registry PATH` | — | Alternate registry. |
| `--tasks A B C` | `tasks` | Space-separated list of task ids. |
| `--language LANG` | `language` | Must be `python` for `revised-v1`. |
| `--translation-target T` | `translation_target` | |
| `--limit N` | `limit` | Per-task record cap. Prefix of stable CSV order. Omit for all records. |
| `--samples N` | `samples` | Generations per record. |
| `--temperature F` | `temperature` | |
| `--max-tokens N` | `max_tokens` | |
| `--provider-extra K=V` | `provider_extras` | Repeatable. Value parsed as JSON, falling back to a bare string. Unsupported keys fail preflight. |
| `--provider-timeout S` | `timeouts.provider_s` | |
| `--evaluator-timeout S` | `timeouts.evaluator_s` | |
| `--retries N` | `retries.attempts` | Overrides attempts only; backoff values come from YAML. |
| `--generation-concurrency N` | `generation_concurrency` | Defaults to `1`. Raise to parallelize generation. |
| `--evaluator-concurrency N` | `evaluator_concurrency` | Must be `1`. `revised-v1` requires sequential isolated evaluation. |
| `--output-dir PATH` | `output_dir` | Run root. |
| `--run-name NAME` | `run_name` | Subdirectory name. Defaults to the model alias. |
| `--dry-run` | — | Print resolved config plus doctor report, then stop. |

**Output location:** `<output_dir>/<run_name>/`. The directory path is printed to stdout
on success; the run's exit status is the process exit code.

**Preflight is not skippable.** `validate_resolved` errors abort with exit `2` before any
provider call (`facets/cli.py:198`).

## `facets resume`

Continues an interrupted run directory. Takes exactly one positional, no flags.

```bash
facets resume reports/runs/qwen38-smoke
```

Reconstructs config and model from the run's own `manifest.json` — no `--config` needed,
because the resolved config was frozen into the manifest. Reuses saved responses after an
evaluation crash and skips identities already completed, so resume makes zero provider
calls for finished work.

Use this for a single interrupted run. For a cohort, use `queue run` or `sweep --resume`.

## `facets sweep`

Runs every model in a config's `models:` list through the queue system. Config-addressed:
it re-reads the YAML on every invocation.

```bash
facets sweep --config experiments/final-study.yaml --dry-run
facets sweep --config experiments/final-study.yaml --resume
facets sweep --config experiments/final-study.yaml --retry-failed --on-error continue
```

| Flag | Notes |
|---|---|
| `--config PATH` | Required. Read on every invocation. |
| `--registry PATH` | Alternate registry. |
| `--dry-run` | Doctor every model in the list. Does not create or touch a queue. |
| `--resume` | Load the existing queue instead of re-freezing. |
| `--retry-failed` | Re-queue items in `failed` state. |
| `--on-error MODE` | `stop` (default) or `continue`. Overrides the queue's stored policy. |
| `--queue-name NAME` | Queue name. Defaults to `sweep-<config stem>`. |
| `--limit N` | Must match the frozen cohort when resuming. |
| `--output-dir PATH` | Run root. |

**Queue name is derived from the config path** (`facets/queue.py:573`), so
`experiments/final-study.yaml` maps to queue `sweep-final-study`. Rename or move the config
and `--resume` loses the queue.

**Without `--resume`, a bare `sweep` re-freezes the cohort** from the current YAML. Editing
the config and re-sweeping starts a fresh cohort rather than continuing the old one.

**On `--resume`, refuses to proceed** if the config no longer matches the frozen cohort
(`facets/cli.py:232`). Create a new queue instead of editing one in place.

Exit code comes from the queue summary.

## `facets queue`

Durable multi-model cohort. Split into create/run so state is inspectable between passes.
Config is frozen at `create` and read from `queue.json` at `run` time — `queue run`
deliberately takes no `--config`, so a later YAML edit cannot silently redefine a cohort.

### `facets queue create`

```bash
facets queue create study --config experiments/final-study.yaml
facets queue create smoke --config experiments/smoke.yaml --models qwen38 phi4 --on-error continue
```

| Flag | Default | Notes |
|---|---|---|
| `name` (positional) | required | Queue name. Becomes `<output_dir>/<name>/` for run directories. |
| `--config PATH` | required | Source of the resolved config. |
| `--models A B` | config's `models:` | Override the model list without editing the config. |
| `--registry PATH` | `config/models.yaml` | |
| `--on-error MODE` | `stop` | `stop` or `continue`, fixed at creation. |
| `--output-dir PATH` | config's `output_dir` | Run root. |
| `--limit N` | — | Applied before freezing. |

Freezes into `queue.json`: `resolved_config`, `config_fingerprint`, `cohort`,
`registry_fingerprint`, and the resolved model list with each item's `run_directory`.

The cohort is `tasks`, `language`, `translation_target`, `samples`, `limit`, `profile`,
`max_tokens`, `temperature`, `provider_extras` (`facets/queue.py:46`).

Rejects a duplicate model: if two requested aliases resolve to the same canonical key, it
errors rather than running one model twice.

Prints a summary with `dry_run: true` and exits `0`. Errors if the queue already exists —
use a new name, do not delete and recreate under the same name.

### `facets queue run`

```bash
facets queue run study
facets queue run study --retry-failed
facets queue run study --on-error continue
facets queue run study --dry-run
```

| Flag | Default | Notes |
|---|---|---|
| `name` (positional) | required | Existing queue. |
| `--output-dir PATH` | `reports/runs` | Must match where the queue lives. |
| `--registry PATH` | `config/models.yaml` | |
| `--on-error MODE` | queue's stored policy | Per-invocation override. |
| `--retry-failed` | off | Re-queue `failed` items. |
| `--dry-run` | off | Report the plan from frozen state; run nothing. |

On every invocation this verifies:

- **Registry fingerprint** unchanged (`facets/queue.py:376`) — else refuses.
- **Per-item preflight** before each model runs (`facets/queue.py:387`) — a failure marks
  that item `failed` with its `issues`, it does not raise.

Holds an OS file lock for the duration, so two drivers cannot execute the same queue.

Item states: `pending` → `running` → `completed`, or → `failed`, `interrupted`, `cancelled`.
An item left `running` by a killed process is recovered to `interrupted` on load. Ctrl-C
marks the current item `interrupted` and exits `130`; re-running resumes it, and a completed
model is never re-requested.

### `facets queue status`

```bash
facets queue status study --output-dir reports/runs
```

Derives the summary from item state and recomputes it rather than trusting a stored field.
`name` positional, `--output-dir PATH` (default `reports/runs`). Runs nothing.

### `facets queue list`

```bash
facets queue list
facets queue list --output-dir reports/runs
```

`--output-dir PATH` only. Lists queues under that root.

### Queue persistence

```
<output_dir>/queues/<name>/queue.json    frozen state, replaced atomically
<output_dir>/queues/<name>/events.jsonl  append-only lifecycle log
<output_dir>/<name>/<model-alias>/      the actual run directory
```

## `facets reevaluate`

Rescores saved responses under a new profile. Never calls a generation provider — a
`_NoGenerationProvider` guard raises if one is attempted. Deterministic code/static
evaluators and the configured embedding service still run.

```bash
facets reevaluate reports/runs/old-run --profile revised-v1 --output reports/reevaluated/qwen38
facets reevaluate reports/legacy.json --output reports/reevaluated/legacy
```

| Flag | Default | Notes |
|---|---|---|
| `source` (positional) | required | A run directory or a legacy report JSON. |
| `--output PATH` | required | Destination run directory. |
| `--profile ID` | `revised-v1` | Overrides the config's profile. |
| `--config PATH` | `experiments/final-study.yaml` | Supplies tasks, limits, and settings. |

**Reads the config**, unlike `queue run` — it needs the task plan to know what to rescore.
Legacy `repr` fields are parsed with a restricted AST adapter, never `eval`.

If any response cannot be recovered, it writes `reevaluation-error.json` listing the
missing keys and exits `4` rather than regenerating.

## `facets summarize`

One flat row per discovered run. No figures, no bootstrap, no ranks.

```bash
facets summarize reports/runs
facets summarize reports/runs --output reports/summary.json
```

| Flag | Notes |
|---|---|
| `results_directory` (positional) | Required. Scanned recursively for run directories. |
| `--output PATH` | Also writes JSON to this file. Still prints to stdout. |

Per run: `run_directory`, `model`, `status`, `evaluator_version`, `profile`, `planned`,
`completed`, `mean_composite_score`, `composite_pass_rate` (score ≥ 0.5), and
`infrastructure_failures` (records with a null `combined_score`).

Queue, checkpoint, and analysis directories are skipped when discovering runs.

## `facets analyze`

Full export: tidy CSVs, dimension tables, weight sensitivity, rank correlations, paired
bootstrap intervals, JSON summaries, and figures.

```bash
facets analyze reports/runs --output reports/analysis/latest --latest-complete
facets analyze reports/runs --output reports/analysis/one --run study/qwen3.8
facets analyze reports/runs --output reports/analysis/all --allow-mixed
```

| Flag | Default | Notes |
|---|---|---|
| `results_directory` (positional) | required | Root scanned for run directories. |
| `--output PATH` | required | Destination for all artefacts. |
| `--run RUN_ID` | — | Repeatable. Analyse only these run ids, relative to `results_directory`. |
| `--latest-complete` | off | For models with several runs, take the newest complete one. |
| `--allow-mixed` | off | Permit incompatible manifests; output degrades to descriptive. |

Reads saved records only. Never calls a provider.

**Selection is explicit by default.** If a model appears in more than one run, the command
refuses rather than guessing. `--latest-complete` and `--run` are mutually exclusive and
error with exit `2` if both are given.

**Incompatible evaluator, profile, or dataset manifests are rejected** unless
`--allow-mixed` is passed, which downgrades the output to descriptive and suppresses ranks
and bootstrap intervals.

**Figures** are written as PNG, PDF, and SVG, with captions in `figures.json` and an
`index.md` gallery: record-level score distributions, per-task radar, model/task heatmap,
grouped task bars, score vs published parameter size, score vs recorded generation latency.
Models with undisclosed parameter size are excluded from the size scatter rather than
assumed. A figure that cannot be drawn is recorded as a warning instead of aborting the run.

Bootstrap intervals describe benchmark-record variation — not repeated-inference
uncertainty and not guaranteed generalization.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. All items completed. |
| `2` | Usage or config error. Doctor failure, preflight error, unresolvable flags, missing queue. |
| `3` | Finished with failures or incomplete items. |
| `4` | Reevaluation could not recover saved responses. |
| `130` | Interrupted (Ctrl-C). Resumable. |

## Which command to use

| Situation | Command |
|---|---|
| Check config before spending money | `facets doctor` |
| One model, one-off | `facets run MODEL` |
| One model, interrupted | `facets resume RUN_DIR` |
| Whole config's model list | `facets sweep --resume` |
| Cohort you want to inspect between passes | `facets queue create` + `queue run` |
| Model not in the registry | `facets run --provider P --model ID` |
| Quick per-run table | `facets summarize` |
| Figures, ranks, bootstrap intervals | `facets analyze` |
| Rescore old responses under a new profile | `facets reevaluate` |

Prefer `queue` over `sweep` for anything you intend to compare across models: the name is
stable rather than derived from a config path, and you can check `queue status` between
passes without re-freezing.

## Gotchas

- `evaluator_concurrency` must stay `1`. `revised-v1` requires sequential isolated evaluation.
- Changing model, dataset, `samples`, `limit`, `profile`, `max_tokens`, `temperature`, or
  `provider_extras` changes the cohort fingerprint. A frozen queue will refuse to resume
  against a config that no longer matches. Create a new queue instead.
- Changing `config/models.yaml` changes the registry fingerprint. Every existing queue
  refuses to run until you either restore the registry or create a new queue.
- Secrets belong only in the environment variable named by the model's `credential_env`. A
  missing variable is a hard error, never a fallback to another key.
- `execution.backend: local` is not a sandbox — a subprocess is neither a filesystem nor a
  network boundary. The container backend is the real-run default.
- `limit` is a prefix of stable CSV order, not a random sample. Omitting it loads every
  record.