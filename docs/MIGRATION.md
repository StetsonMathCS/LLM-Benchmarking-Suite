# Legacy report migration

Legacy “Code Generation Tests”/“Code Completion Tests” concatenated candidate and trusted checker definitions but did not call `check(entry_point)`. A script that only defined `check` exited successfully, so those scores are not evidence that reference assertions ran. The historical evaluator in Git has the same defect.

Legacy “Test Pass Rate” was a different mechanism: it reran the whole generated suite against a return-None override and inferred kills from aggregate passing-count differences. It did not preserve test identities, validate a diverse frozen mutant pool, or separate collection/harness failures. It must not be described as generated-test mutation effectiveness.

Legacy runtime often executed definitions without the record workload and used a logistic score where parity was .5. Historical generation reports also reflect effective CC/CGT/lint/vulnerability weights of 4/9, 4/9, 1/18, and 1/18, while the later matrix contained .40/.60. These configurations cannot be silently mixed.

`facets reevaluate` recovers saved response text with a restricted AST adapter (never `eval`) and runs the revised deterministic evaluators. It stops if a response cannot be verified and never calls a generation provider. It writes a new directory with source path/hash, revised profile/evaluator hashes, and new record identities. Old dimension values remain labeled `legacy_unverified`; old CGT is never relabeled RTS. Original reports are preserved unchanged.

Expected score differences arise from actual checker invocation, identity-based mutation detection, correctness-gated workload timing, capped relative efficiency, strict infrastructure-failure handling, structured output comparison, and a frozen profile. Revised and historical rankings must be presented in separate tables/figures.

