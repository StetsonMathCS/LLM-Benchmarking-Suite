# Datasets

This document describes the source, structure, and record counts for each
benchmark dataset used in the LLM Testing Suite.

## Summary

| Task | Python | C++ | JavaScript | Primary Source |
|------|--------|-----|------------|----------------|
| Bug Fixing | 31 | 50 | 50 | QuixBugs, Defects4C, FixJS |
| Code Completion | 164 | 34 | 32 | OpenAI HumanEval |
| Code Generation | 164 | 24 | 36 | OpenAI HumanEval |
| Code Review | 20 | 20 | 20 | Microsoft CodeReviewer |
| Partial Transform | 30 | -- | -- | JetBrains LCA |
| Refactoring | 20 | 20 | 19 | Synthetically generated |
| Test Generation | 80 | -- | -- | Derived from HumanEval |
| Translation | 30 | 30 | 30 | Synthetically generated |

---

## Bug Fixing

**Task:** Given buggy code, produce a corrected version that compiles and
produces the expected output.

**Columns:** `id, buggy_code, fixed_code, expected_output`

| Language | Records | Source |
|----------|---------|--------|
| Python | 31 | **QuixBugs** (Lin et al., 2017) |
| C++ | 50 | **Defects4C** |
| JavaScript | 50 | **FixJS** / **TFix** |

### Python — QuixBugs

The Python bug-fixing dataset is drawn from the
[QuixBugs](https://github.com/jkoppel/QuixBugs) benchmark, a collection of 40
classic algorithm programs each containing a single known bug. 31 self-contained
programs (those not requiring external data structures like `Node`) were
converted into runnable code with test-call print statements. Expected outputs
were verified by executing the correct versions.

**Citation:**
> D. Lin, J. Koppel, A. Chen, and A. Solar-Lezama, "QuixBugs: A Multi-Lingual
> Program Repair Benchmark Set Based on the Quixey Challenge," in Proceedings of
> the ACM SIGPLAN International Conference on Systems, Programming, Languages,
> and Applications: Software for Humanity (SPLASH Companion), 2017.

### C++

Source: [Defects4C](https://github.com/defects4c/defects4c) — a curated
collection of real bugs from C/C++ projects with verified patches.

### JavaScript

Sources:
- [FixJS](https://github.com/AAI-USZ/FixJS) — bug-fix pairs mined from
  JavaScript repositories.
- [TFix](https://github.com/eth-sri/TFix) — a learning-based JavaScript repair
  dataset from ETH Zurich.

---

## Code Completion

**Task:** Given a partial function signature and docstring, complete the
implementation.

**Columns (Python):** `id, prompt, completed, test, entry_point`
**Columns (C++/JS):** `id, prompt, completed`

| Language | Records | Source |
|----------|---------|--------|
| Python | 164 | **OpenAI HumanEval** (Chen et al., 2021) |
| C++ | 34 | Synthetically generated |
| JavaScript | 32 | Synthetically generated |

### Python — HumanEval

The Python dataset uses
[HumanEval](https://github.com/openai/human-eval), a hand-crafted set of 164
programming problems with function signatures, docstrings, reference
implementations, and unit tests created by OpenAI researchers.

**Citation:**
> M. Chen et al., "Evaluating Large Language Models Trained on Code," arXiv
> preprint arXiv:2107.03374, 2021.

---

## Code Generation

**Task:** Generate a complete function from a natural-language specification or
docstring prompt.

**Columns (Python):** `id, prompt, completed, test, entry_point`
**Columns (C++/JS):** `id, instruction, generated_code, expected_output`

| Language | Records | Source |
|----------|---------|--------|
| Python | 164 | **OpenAI HumanEval** (Chen et al., 2021) |
| C++ | 24 | HumanEval problems translated to C++ |
| JavaScript | 36 | Synthetically generated |

### Python — HumanEval

Same dataset as Code Completion (see citation above). The distinction is in how
the benchmark uses the data: Code Generation evaluates from-scratch generation,
while Code Completion evaluates partial-program completion.

---

## Code Review

**Task:** Given a code snippet, produce a review identifying bugs, security
issues, design problems, and other quality concerns.

**Columns:** `id, code_snippet, review, expected_reviews`

| Language | Records | Source |
|----------|---------|--------|
| Python | 20 | **Microsoft CodeReviewer** |
| C++ | 20 | **Microsoft CodeReviewer** |
| JavaScript | 20 | **Microsoft CodeReviewer** |

### All Languages — Microsoft CodeReviewer

Adapted from the
[CodeReviewer](https://github.com/microsoft/CodeBERT/tree/master/CodeReviewer)
dataset released as part of Microsoft's CodeBERT project. The `expected_reviews`
field contains pipe-separated issue categories (e.g.,
`SECURITY|BUGS|PERFORMANCE`) used for scoring review completeness.

**Citation:**
> S. Lu et al., "CodeXGLUE: A Machine Learning Benchmark Dataset for Code
> Understanding and Generation," in Proceedings of NeurIPS Datasets and
> Benchmarks Track, 2021.

---

## Partial Transform

**Task:** Transform all instances of a specific code pattern (e.g., replace
`range(len(...))` with `enumerate()`) without leaving any untransformed.

**Columns:** `id, code_snippet, from, to, final_output, verify_ast_nodes_removed, verify_ast_nodes_added, verify_strategy, verify_note`

| Language | Records | Source |
|----------|---------|--------|
| Python | 30 | **JetBrains LCA** |

### Python — JetBrains LCA

Derived from the [LCA Project-Level Code Completion](https://huggingface.co/datasets/JetBrains-Research/lca-project-level-code-completion)
dataset on HuggingFace, published by JetBrains Research. Records include AST
verification metadata (`verify_ast_nodes_removed`, `verify_ast_nodes_added`) to
enable automated checking that all pattern instances were transformed.

---

## Refactoring

**Task:** Improve code structure (extract functions, simplify logic, use modern
idioms) while preserving identical behavior.

**Columns (Python):** `id, original_code, refactoring_task, output_expected, expected_console_output`
**Columns (C++/JS):** `id, original_code, refactoring_task, output_expected`

| Language | Records | Source |
|----------|---------|--------|
| Python | 20 | Synthetically generated |
| C++ | 20 | Synthetically generated |
| JavaScript | 19 | Synthetically generated |

These datasets were synthetically generated to cover common refactoring
patterns: extracting validation logic, replacing nested conditionals, using
dataclasses, improving error handling, and converting to generator expressions.
Each record includes a refactoring task description and an expected refactored
output.

---

## Test Generation

**Task:** Given a function implementation and description, generate
comprehensive unit tests.

**Columns:** `id, code_snippet, description`

| Language | Records | Source |
|----------|---------|--------|
| Python | 80 | **Derived from OpenAI HumanEval** |

### Python — HumanEval-Derived

The 80 function implementations are taken directly from the HumanEval benchmark
(Chen et al., 2021). The `code_snippet` field contains the complete reference
implementation, and `description` is extracted from the original docstring. This
reuse ensures the functions are well-specified, non-trivial, and have known
correct behavior — ideal targets for test generation evaluation.

See Code Generation section above for citation.

---

## Translation

**Task:** Translate code from one programming language to another while
preserving behavior.

**Columns (Python):** `id, code_snippet, expected_output`
**Columns (C++/JS):** `id, code_snippet, code_output`

| Language | Records | Source |
|----------|---------|--------|
| Python | 30 | Synthetically generated |
| C++ | 30 | Synthetically generated |
| JavaScript | 30 | Synthetically generated |

These datasets were synthetically generated with parallel implementations
demonstrating common language features: basic functions, list/array operations,
class definitions, and standard library usage. Each record includes the source
code and its expected console output for behavioral verification after
translation.

---

## Notes on Synthetic Data

Several datasets (refactoring, translation, and the C++/JavaScript subsets of
some tasks) are synthetically generated. These are acknowledged as a limitation.
The primary contribution of this work is the evaluation framework and
multi-dimensional scoring methodology, not the datasets themselves. Where
established benchmarks exist (HumanEval, QuixBugs, CodeReviewer, LCA), they are
used. For tasks without widely-adopted standard benchmarks (refactoring,
translation), synthetic data demonstrates the framework's extensibility.

---

## References

1. Chen, M. et al. (2021). "Evaluating Large Language Models Trained on Code."
   arXiv:2107.03374.
2. Lin, D. et al. (2017). "QuixBugs: A Multi-Lingual Program Repair Benchmark
   Set Based on the Quixey Challenge." SPLASH Companion 2017.
3. Lu, S. et al. (2021). "CodeXGLUE: A Machine Learning Benchmark Dataset for
   Code Understanding and Generation." NeurIPS Datasets and Benchmarks.
4. JetBrains Research. "LCA Project-Level Code Completion." HuggingFace Datasets.
