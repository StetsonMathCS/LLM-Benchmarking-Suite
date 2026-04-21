# Test Generation Dataset & Mutation-Based Scoring

## Overview

The test_generation dataset contains 80 Python functions from OpenAI's HumanEval benchmark, each with:

- **code_snippet**: The reference function implementation
- **description**: A natural-language description of what the function does
- **test**: The ground-truth test cases (assertions) from HumanEval
- **entry_point**: The function name being tested

## Mutation-Based Scoring

The test pass rate dimension now uses **mutation testing** to measure test quality:

### Scoring Algorithm

1. **Pass Rate** — Run LLM-generated tests against the *correct* code
   - `pass_rate = tests_passed / total_tests`
   - Tests that fail on correct code are wrong (score 0)

2. **Kill Rate** — Run the *same* tests against a null mutant (function returns None)
   - Tests that still pass on the mutant are trivial (no real assertions)
   - Tests that fail on the mutant actually detect the bug
   - `kill_rate = tests_killed / tests_that_passed_on_original`

3. **Final Score** — Both must be high
   - `score = pass_rate * kill_rate`
   - All assertions + mutant detection = 1.0
   - No assertions or tautologies = 0.0
   - Mixed quality = partial credit

### Examples

| Scenario | Pass Rate | Kill Rate | Score | Interpretation |
|----------|-----------|-----------|-------|-----------------|
| Real assertions (catches bug) | 1.00 | 1.00 | **1.00** | ✅ Good tests |
| No assertions (trivial) | 1.00 | 0.00 | **0.00** | ❌ Tests are useless |
| Mixed (2 real + 2 trivial) | 1.00 | 0.50 | **0.50** | ⚠️ Partial credit |
| Tests fail on correct code | 0.00 | — | **0.00** | ❌ Tests are wrong |

## Running Tests Locally

### Quick Test (10 records)

```bash
cd "d:/STETSON CLASSES/SENIOR RESEARCH/codebase"
.venv/Scripts/python.exe scripts/test_mutation_standalone.py --limit 10
```

Output shows:
- **Score** for each record (0.0–1.0)
- **Pass rate** (how many tests passed on correct code)
- **Kill rate** (how many of those tests fail on the mutant)

### Check a Specific Record

```bash
.venv/Scripts/python.exe scripts/test_mutation_standalone.py --limit 1
```

### Increase Test Count

```bash
.venv/Scripts/python.exe scripts/test_mutation_standalone.py --limit 80
```

### Run Through the Framework

When integrated into the full benchmark suite:

```bash
python run_experiment.py --tasks test_generation --limit 5
```

This will:
1. Load the test_generation CSV
2. For each record, have an LLM generate tests for the code_snippet
3. Score the LLM-generated tests using mutation-based scoring
4. Report pass_rate × kill_rate as the final score

## Test Field Format

The **test** column contains HumanEval-style assertions:

```python
METADATA = {
    'author': 'openai',
    'dataset': 'humaneval'
}

def check(candidate):
    assert candidate([1.0, 2.0, 3.0], 0.5) == False
    assert candidate([1.0, 2.8, 3.0, 4.0], 0.3) == True
    # ... more assertions
```

The dimension automatically converts this to pytest format:

```python
def test_has_close_elements_1():
    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False
def test_has_close_elements_2():
    assert has_close_elements([1.0, 2.8, 3.0, 4.0], 0.3) == True
# ...
```

## Why Mutation Testing?

Without mutation testing, trivial tests score 100%:

```python
# This passes without running any code
def test_add():
    result = add(2, 3)
    # no assertion!
```

With mutation testing, this scores 0.0:

```python
# LLM code (original): returns correct result ✓
# LLM code (mutant):   returns None
# The trivial test passes on both → kill_rate = 0.0 → score = 0.0
```

## References

- **HumanEval**: Chen et al., "Evaluating Large Language Models Trained on Code" (2021)
- **Mutation Testing**: Jia & Harman, "An Analysis and Survey of the Development of Mutation Testing" (IEEE TSE, 2011)
