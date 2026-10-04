# FACETS revised-v1 methods and scoring

## Study design

The frozen default retains six task weights: code generation 0.27, translation 0.16, bug fixing 0.15, code review 0.15, refactoring 0.15, and test generation 0.12. The default is one sample per record. Deterministic selection uses stable CSV order; a limit selects the prefix before sampling. The manifest records selection, hashes, requested/effective provider settings, and sample identity.

Within-task weights use stable IDs:

- Bug fixing: functional correctness .70, linting .10, code consistency .10, vulnerabilities .05, structural similarity .05.
- Code generation: code consistency .40, RTS .60.
- Code review: RRS 1.00.
- Refactoring: functional correctness .40, runtime analysis .35, code consistency .10, linting .10, vulnerabilities .05.
- Test generation: code consistency .25, GTE .75.
- Translation: code consistency .30, linting .10, vulnerabilities .10, functional correctness .50.

Every declared weight set must sum to exactly one within numerical tolerance; values such as .90 are rejected, never silently normalized. For an explicitly selected subset of tasks, task weights are deterministically normalized over that predeclared subset and recorded. Dimensions may be not applicable only where declared by the frozen profile. Missing mandatory correctness evidence is an infrastructure failure, not a style-only score.

The composite threshold is one configurable value, default .5. “Composite pass rate” means the proportion meeting that threshold and is not classification accuracy or functional correctness.

## Reference Test Success

RTS supports the current Python HumanEval format. Candidate and trusted tests are separate modules. Preflight requires an identifier entry point and exactly one complete `def check(candidate)` body. The harness verifies that the candidate entry point exists and is callable, emits an invocation marker, and explicitly calls the trusted checker. Setup code, imports, loops, and multiline assertions remain intact. Candidate definitions cannot replace the trusted `check` binding.

RTS is binary per record. A completed checker is 1; missing functions, empty/malformed candidate output, syntax/runtime/assertion failure, or candidate timeout is 0. Missing/malformed trusted tests and unsupported mandatory formats are infrastructure errors. Reports retain invocation, entry point, reference/dataset hashes, exit status, timeout, and bounded diagnostics. HumanEval exposes suite success, so FACETS does not invent an assertion/test count.

## Generated Test Effectiveness

Generated pytest/unittest suites must import the implementation from `candidate`; suites that define/shadow the entry point are rejected. HumanEval-shaped generated tests retain the complete checker and receive a wrapper that calls it. Pytest hooks record stable node IDs including parameter IDs, phase outcomes, collection errors, skips, expected failures, unexpected passes, exceptions, and durations. Pytest collects unittest suites through the explicit adapter.

Validity is `V = baseline-passing executable tests / collected executable tests`. Skips and expected failures are reported but excluded from the executable denominator; failures, errors, and unexpected passes are executable. No executable tests yields V=0. Collection/malformed-suite failures do not pass.

A deterministic maximum-five pool uses bounded first-site AST operators: return `None`, return zero, return false, flip a comparison, and negate a condition. Duplicates and syntax-invalid mutants are excluded. The known-correct baseline must pass trusted reference tests. Each mutant must be distinguished by those same trusted tests; otherwise it is excluded with a reason, not called equivalent. Pool member and aggregate hashes are stable across models.

For each eligible mutant, only exact test IDs that passed on the baseline can detect it. Detection requires one such ID to fail/error against the mutant. Changed collections, harness errors, and suite-level timeouts are infrastructure diagnostics and never automatic kills. Mutant runtime errors observed by an otherwise intact test ID count as detection; unattributable timeouts do not. `M = detected eligible mutants / evaluated eligible mutants`; `GTE = V × M`. No eligible pool for a mandatory record invalidates the run and must be repaired before the sweep.

## Functional behavior and execution

Functional Correctness is observed behavior on supplied tests/inputs, not a proof. Expected values use presence checks, so `""`, `0`, and `False` remain valid. Failed executions never compare equal merely because both stdout strings are empty. Translation executes source and target using their actual languages and applies only line-ending/final-newline normalization plus JavaScript scalar spelling; it does not broadly rewrite output.

Real-run candidate execution uses the configured container: networking disabled, read-only mounts/root, dropped capabilities, no-new-privileges, resource/process limits, bounded output, minimal environment, and process-group/container cleanup on timeout. Provider secrets are not passed into the container. The local backend is explicitly non-isolating and only for controlled offline regression tests.

## Runtime analysis

Refactoring uses the trusted expected refactored implementation as baseline. Both baseline and candidate execute the same dataset workload. Definitions/setup occur before timing; the workload is warmed up, invoked repeatedly with fresh workload-local inputs, and timed inside the child using `perf_counter_ns`. The median is reported. Peak `ru_maxrss` is measured in separate baseline/candidate processes. Raw samples, units, ratios, invocation count, method, and environment are retained.

Both baseline and candidate must satisfy the expected workload output before performance scoring. The trusted baseline failing its contract is infrastructure failure; an incorrect candidate receives zero and no efficiency credit. Each component is `min(1, baseline/generated)`. The runtime score is the geometric mean of time and memory components. Equal or better use earns 1; regressions earn less. Raw ratios remain uncapped. Zero/unavailable time or memory makes the measurement unavailable/infrastructure-failed, never full credit. This is measured workload performance, not asymptotic complexity.

Refactoring record 1 previously compared the exact `OrderResult(...)` repr. The revised dataset sidecar uses explicit attribute assertions (`order_id`, `total`, and `user_id`) and a success marker, so an otherwise valid result is not rejected merely for repr formatting while the requested structured contract remains enforced. The sidecar participates in dataset and record hashes.

## Other dimensions and failure policy

Code Consistency is a style heuristic. SS is AST/tree structural similarity and does not establish semantic equivalence. RRS is embedding similarity to a reference review and does not establish expert-review accuracy. Linting and vulnerability dimensions record findings and exact named tools (Pylint, Bandit, cppcheck, or ESLint); missing tools are infrastructure failures.

Candidate failures—refusal/empty or malformed output, failed checks, and candidate timeouts—score zero for applicable criteria and stay in the planned denominator. Infrastructure failures—provider outage, missing tool, broken trusted data, embedding failure, or evaluator exception—may be retried only when transient and make an unresolved official run incomplete with no official final score. Not-applicable dimensions are frozen in advance.

Functional pass@k groups per-problem/per-sample actual functional outcomes for applicable tasks and uses the standard unbiased estimator. It never substitutes composite score. When `n < k`, the value is unavailable (`null`), not zero. Composite success remains separate.

## Analysis and limitations

Prespecified sensitivity profiles are frozen default, equal task weights, equal dimension weights, code-generation +20%, and review +20%, with normalization defined before ranking inspection. Paired deterministic bootstrap intervals resample shared record IDs within task strata; they quantify benchmark-record variation, not repeated-inference uncertainty or guaranteed generalization. Runtime raw ratios are analyzed separately and only after correctness gating. Parameter-size plots include known sizes only; undisclosed sizes remain a separate category. FACETS scores are not numerically equated with LiveCodeBench or other benchmarks.
