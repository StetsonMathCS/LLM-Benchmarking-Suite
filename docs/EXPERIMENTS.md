# Experiments and queues

Two ways to run FACETS: `facets run` for a single model, and `facets queue` for a
durable multi-model cohort. Both call the same `execute_run` pipeline, so a queued
model and a directly-run model produce identical run directories.

## Experiment YAML

An experiment config is a flat mapping of run-shaping keys plus a `models` list.
Only the keys below are read; anything else in the file is inert, so a typo'd key
silently does nothing rather than failing loudly.

| Key | Meaning |
|---|---|
| `profile` | Scoring profile id. Defines task weights and per-dimension weights; changing it changes the composite. |
| `language`, `translation_target` | `language` must be `python`; revised-v1 is Python-focused and preflight rejects anything else. `translation_target` is the target for the `translation` task. |
| `tasks` | Subset of the six task ids. Omitted tasks are never planned. |
| `samples`, `limit` | Sample count and per-task record cap. `limit` is a prefix of the dataset's stable CSV order; omit it for the whole dataset. Both feed the cohort fingerprint. |
| `temperature`, `max_tokens` | Requested generation settings. Prefer omitting both so a provider default is used and recorded as effective. |
| `provider_extras` | Provider-specific extras; keys unsupported by the provider fail preflight. |
| `timeouts` | `provider_s` and `evaluator_s` walls. |
| `retries` | `attempts`, `initial_backoff_s`, `max_backoff_s`. |
| `execution` | Evaluator sandbox: `backend`, `image`, `memory_mb`, `cpus`, `pids_limit`. |
| `mutation_preflight` | `full` (the default) makes `doctor` validate the shared dataset once. Later models in a multi-model pass use `already_validated_for_shared_dataset` to skip the repeat; any other value skips it too, so this key is trusted rather than validated. |
| `embedding` | `base_url` and `model` for reference-review similarity. Omit to disable the dimension. |
| `generation_concurrency`, `evaluator_concurrency` | `evaluator_concurrency` must be `1`; revised-v1 requires sequential isolated evaluation. `generation_concurrency` defaults to `1` and may be raised, since usage attribution is thread-bound per record. |
| `output_dir` | Run root. `queue` keeps queue bookkeeping under `<output_dir>/queues/`. |
| `expected_counts` | Per-task record counts asserted by `doctor`; a mismatch is reported, never silently accepted. |

Secrets never belong in a config. Credentials come from the environment variable
named by the model's `credential_env`, and a missing one is a hard error rather
than a fallback to another key.

## Model aliases

[`config/models.yaml`](../config/models.yaml) has a canonical `models` map and a
short `aliases` map. An alias resolves to a canonical model key, and the canonical
key stays the model identity in manifests, so `sonnet` and `claude-sonnet-4-6`
cannot fork a cohort. Alias chains are followed, and cycles are rejected.

`availability` is recorded, never used to substitute a model. `unresolved` and
`verify_with_provider` entries still run if you ask for them by exact name; they
are flagged so an unverified result is not mistaken for a verified one.

## Queues

```bash
facets queue create study --config experiments/final-study.yaml
facets queue run study
facets queue run study --retry-failed
facets queue status study
facets queue list
```

`create` freezes the cohort: the resolved config, its fingerprint, the model order,
and a fingerprint of the registry (models *and* aliases). `run` refuses to proceed
if that fingerprint changed or if `--limit`/`--samples` no longer match the frozen
cohort. Create a new queue instead of editing one in place.

`sweep` is a thin wrapper: it derives a queue named `sweep-<config stem>` and runs
it, so both paths share one implementation and one set of guarantees.

### Item states

`pending` → `running` → `completed`, or → `failed`, `interrupted`, `cancelled`.

* Default `--on-error stop` halts before the next model; remaining items stay `pending`.
* `--on-error continue` runs the rest and still exits non-zero if anything failed.
* Ctrl-C marks the current item `interrupted`, saves, and exits `130`. Re-running
  resumes it; a completed model is never re-requested.
* An item left `running` by a killed process is recovered to `interrupted` on load.
* A preflight failure is recorded as `failed` with its `issues`, not raised.

### Exit codes

`0` all completed · `2` usage/config error · `3` finished with failures or
incomplete · `4` reevaluation could not recover responses · `130` interrupted.

### Persistence

```
<output_dir>/queues/<name>/queue.json   frozen state, replaced atomically
<output_dir>/queues/<name>/events.jsonl append-only lifecycle log
<output_dir>/<name>/<model-alias>/     the actual run directory
```

An OS file lock is held for the duration of a run, so two drivers cannot execute
the same queue. Summary is derived from item state and recomputed rather than
trusted from a stale field.

## Analysis

```bash
facets summarize reports/runs
facets analyze reports/runs --output reports/analysis/revised-v1
facets analyze reports/runs --output reports/analysis/latest --latest-complete
facets analyze reports/runs --output reports/analysis/one --run study/sonnet
```

Analysis reads saved records only. It never calls a provider. Queue, checkpoint,
and analysis directories are skipped when discovering runs.

Selection is explicit by default: if a model appears in more than one run the
command refuses rather than guessing. `--latest-complete` takes the newest
complete run per model, and `--run` selects exactly the given run ids. Incompatible
evaluator/profile/dataset manifests are rejected unless `--allow-mixed` is passed,
which downgrades the output to descriptive and suppresses ranks and bootstrap
intervals.

Alongside the tidy CSVs, `analyze` writes each figure as PNG, PDF, and SVG with a
caption in `figures.json` and an `index.md` gallery. A figure that cannot be drawn
is recorded as a warning instead of aborting the run, so one empty panel cannot
cost the whole set.
