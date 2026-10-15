# 🧪 tests/

This directory contains unit tests for the `src/` package, written with **pytest**.

> Tests verify that the core utilities (corruption functions, evaluation metrics, etc.)
> behave correctly — catching bugs before they silently corrupt experiment results.

---

## Why Tests Matter for Research Code

Research code is notorious for subtle bugs that produce plausible-looking but
wrong numbers. A few targeted unit tests can catch:

- Off-by-one errors in corruption intensity calculations
- Incorrect F1 averaging modes (`macro` vs. `weighted`)
- Edge cases in corruption functions (empty strings, single-character words)
- Silent import errors that break notebook execution

---

## Running the Test Suite

```bash
# From the project root, with .venv active:
pytest tests/ -v

# With coverage report:
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Planned Test Files

```
tests/
├── README.md                        ← This file
├── test_corruption.py               ← Tests for src/corruption.py
│       - test that output length changes correctly per corruption type
│       - test edge cases: empty string, single character, full sentence
│       - test that intensity=0 returns the original string unchanged
├── test_preprocessing.py            ← Tests for src/preprocessing.py
│       - test TF-IDF vectorizer produces expected output shapes
│       - test text cleaning removes expected noise
└── test_evaluation.py               ← Tests for src/evaluation.py
        - test accuracy / F1 match sklearn on known inputs
        - test robustness_drop returns 0 when clean == corrupted accuracy
        - test edge case: all predictions correct, all predictions wrong
```

---

## Testing Guidelines

| Guideline | Detail |
|-----------|--------|
| **One assert per test** | Keeps failures easy to pinpoint |
| **Test edge cases** | Empty strings, single tokens, 100% corruption intensity |
| **Use fixed seeds** | Corruption functions that use `random` must accept a `seed` param |
| **Name tests clearly** | `test_deletion_returns_shorter_string` beats `test_1` |
| **No network calls in tests** | Mock or skip tests that would download data |
| **Run tests before committing** | Make `pytest tests/` pass before every `git commit` |

---

## Adding a New Test

1. Create `tests/test_<module_name>.py`.
2. Import from `src`:
   ```python
   from src.corruption import corrupt_text
   ```
3. Write functions starting with `test_`:
   ```python
   def test_deletion_reduces_length():
       original = "hello world"
       corrupted = corrupt_text(original, strategy="deletion", intensity=0.5, seed=42)
       assert len(corrupted) < len(original)
   ```
4. Run `pytest tests/ -v` to confirm the test passes.
