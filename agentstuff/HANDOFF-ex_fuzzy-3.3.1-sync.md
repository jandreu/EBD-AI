# Handoff: ex_fuzzy 3.3.1 sync and the temporal move to `ebdai`

Written 2026-11-05. Supersedes nothing; read `LLM/STATUS.md` first, then this.

## Objective

Port every upstream Ex-Fuzzy change made after the 3.3.0 merge into EBD-AI,
and keep temporal fuzzy sets — but re-homed in `ebdai`, because upstream
removed them.

User decisions taken during the work:

1. **Scope: everything upstream has** (fixes, batched fitness, growing data),
   not just the bug fixes.
2. **Temporal stays, as a BDI feature**: "port the temporal.py fuzzy sets into
   EBD-toolbox and keep it in this rebase, it will part of the EBDAI import and
   not ex-fuzzy anymore".

## Where the work is

| What | Where |
| --- | --- |
| Branch (pushed) | `sync/upstream-post-3.3.0`, commit `468af94`, based on `main` `7e7d72d` |
| Worktree | `/tmp/opencode/ebdai-sync` (shares the git dir of `~/github/EBD-AI`) |
| Venv used | `/tmp/opencode/venv-sync` (`pip install -e "./ebdai-sync[test]"`) |
| Upstream clone (bare) | `/tmp/opencode/exfuzzy-upstream`, `main` = `fac6c0e`, `v3.3.0` = `5fbfa…` |
| Patch applied | `/tmp/opencode/up-sync.patch` (`v3.3.0..main` minus excluded files) |
| Tree exports used for diffs | `/tmp/opencode/compare/{v330,ebdai-main,up-main,d330}` |

`/tmp` is disposable: if it is gone, re-create with

```bash
cd /tmp/opencode
git clone --bare https://github.com/Fuminides/ex-fuzzy.git exfuzzy-upstream
git -C ~/github/EBD-AI worktree add /tmp/opencode/ebdai-sync sync/upstream-post-3.3.0
```

(the branch is on `origin`, so a plain clone + `git switch` works too).

## What changed

`ex_fuzzy/` is now **3.3.1** and carries every upstream commit after 3.3.0:

- batched fitness: `batch_size` (`_streamed_fitness.py`, `FitRuleBase` hook)
- growing training subset: `initial_data_fraction` (`_growing_data.py`,
  pymoo backend and `FitRuleBase`)
- `explainable_predict` returns the winner's association degree
- term genes bounded by the largest partition and remapped proportionally
  (`_fitness.py`, `_population_fitness.py`, `_torch_fitness.py`,
  `evolutionary_fit.py`)
- pickling: `release_problem_data` drops pymoo's problem, and `_problem.py`
  keeps fits loadable in a new process
- `MasterRuleBase.pruning_tolerance`
- `utils.construct_breakpoint_partition`
- regression `unique_antecedents`
- `pattern_stability` rule-set stability diagnostics
- training base rates predicted where no rule fires

`ex_fuzzy/temporal.py` was **moved** (`git mv`) to `ebdai/temporal.py` and
adapted to the post-removal API:

- `temporalFS.type()` returns the wrapped set's family (`t1`/`t2`/`gt2`), so
  generic `ex_fuzzy` code (fast paths, plotting, rule-base construction) works
  unchanged. New `temporalFS.temporal_type()` returns the temporal flavour
  (`TMP_FUZZY_SETS.temporal`, `temporal_t2`, `temporal_gt2`).
- `temporalMasterRuleBase.bind_time_moments(moments)` stores per-sample
  moments, and `compute_firing_strengths`, `compute_association_degrees`,
  `winning_rule_predict` and `_winning_rules` take `time_moments=None`,
  falling back to the bound moments. This is what lets the stock
  `ex_fuzzy.eval_rules.evalRuleBase` (whose `time_moments` parameter upstream
  deleted) score a temporal rule base.
- `_winning_rules` now returns the winner's association degree, matching the
  `ex_fuzzy` fix.
- The partition helpers removed from `ex_fuzzy.utils` moved here with the same
  names: `construct_conditional_frequencies`, `classify_temp`, `assign_time`,
  `create_tempVariables`, `create_multi_tempVariables`, `temporal_cuts`,
  `temporal_assemble`.

Registration and docs:

- `ebdai/__init__.py`: `'temporal'` added to `_SUBMODULES`; 14 names added to
  `_EXPORTS` (`TemporalFuzzyRulesClassifier`, `temporalFS`, …). `import ebdai`
  stays lazy: it does not import numpy/sklearn.
- `tests/test_temporal.py` → `tests/test_ebdai_temporal.py`, imports
  `from ebdai import temporal`, type assertions updated, helper calls moved to
  `temporal.*`. 15 tests.
- `tests/test_lazy_imports.py`: `ex_fuzzy.__version__` assertion 3.3.0 → 3.3.1.
- New `docs/source/temporal.rst`; `docs/source/index.rst` toctree `tmpfs` →
  `temporal`; `docs/source/api/ebdai.rst` automodule for `ebdai.temporal`;
  `docs/source/user-guide/ebdai.rst` version strings; changelog entry;
  `LLM/STATUS.md` and `LLM/DEVELOPMENT.md` updated.

## Deliberately not ported

- `README.md` (upstream branding/labs) — EBD-AI keeps its own.
- `LLM/**` from upstream — they are that repo's agent notes.
- `pyproject.toml` — EBD-AI's packaging differs.
- `docs/source/changelog.rst`, `docs/source/index.rst` — EBD-AI's own; only a
  hand-written entry and the toctree edit were applied.
- `docs/source/optimize.rst` and `docs/source/user-guide/training-performance.rst`
  — the upstream hunks were **skipped, not merged**. `training-performance.rst`
  is where upstream documents `batch_size`; that section still needs porting
  (see below).
- The temporal `FUZZY_SETS` members are **not** coming back to `ex_fuzzy`. If
  code needs the temporal flavour, use `ebdai.temporal`.

## Verification already done

- Full suite, at the code state committed (`tests/`, excluding the slow KEEL
  benchmark): **1567 passed, 134 skipped** in ~227 s.
  `pytest tests/ -q --ignore=tests/test_keel_benchmark.py`
- `tests/test_ebdai_temporal.py`: 15 passed.
- `tests/test_ebdai_temporal.py`, `test_lazy_imports.py`, `test_model_pickle.py`
  together: 24 passed after the docs pass.
- The documented install path works: `pip install "ebdai @ git+…"` yields
  `ex_fuzzy 3.3.1`.
- **Not** verified: Sphinx build, `tests/test_keel_benchmark.py`, demo
  notebooks, and the two large CPU-bound benchmarks (`benchmark_streamed_memory.py`,
  `benchmark_regression_unique_antecedents.py`).

## Remaining work

1. **Review and merge** `sync/upstream-post-3.3.0` (single commit `468af94`).
   Nothing is on `main` yet.
2. Port the upstream `batch_size` documentation into
   `docs/source/user-guide/training-performance.rst`, and any still-relevant
   hunks of `docs/source/optimize.rst`. Compare with
   `git -C /tmp/opencode/exfuzzy-upstream diff v3.3.0 main -- docs/source`.
3. Build the docs to check the new page and the `tmpfs` → `temporal` toctree
   change (`sphinx-build -b html docs/source /tmp/docs`), since the deleted
   `docs/source/tmpfs.rst` used to satisfy that toctree.
4. Optional: run `python Demos/run_notebooks.py` (no notebook was edited, but
   the version string appears in outputs).
5. When merged: `git worktree remove /tmp/opencode/ebdai-sync`.

## Gotchas

- Project policy (`LLM/STATUS.md`): do **not** `git merge upstream/main`; port
  by patch or re-implementation. This sync was applied with `git apply -3` from
  `v3.3.0..main` (excluding the files listed above), not a merge.
- One conflict occurred, in `evolutionary_fit.py`'s `allow_unknown` docstring:
  upstream's wording was kept and EBD-AI's `'ebdai[evox]'` install hint
  preserved.
- Several docs files use **CRLF** line endings (`docs/source/index.rst`); a
  plain `sed 's/^    tmpfs$/…/'` silently does nothing. Use `\r`-aware edits or
  the edit tool.
- Bit-for-bit parity: the term-gene remapping changes which rule bases a given
  genome decodes to, which is the upstream intent. No arithmetic in the
  objective changed; `_fitness.py` still returns `None` for unsupported cases,
  and the parity tests in `tests/test_array_evaluation.py`,
  `test_genetic_fitness_semantics.py`, `test_fast_fitness.py`,
  `test_population_evaluation.py`, `test_route_dispatch.py` and
  `test_evox_population.py` pass.
- Repo state from earlier in the session: EBD-AI now lives at
  `https://github.com/jandreu/EBD-AI` (public, `origin` rewritten). The
  classroom app is private at `jandreu/EBD-AI-Classroom` and cloned at
  `~/github/EBD-AI-Classroom`; do not re-add `Demos/fuzzy_classroom/` here.
