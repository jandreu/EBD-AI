# Agent handoff

Last updated 2026-10-15. Check the working tree and `git log` before treating
this as current.

## What this repository is

[HAISymbiosis/EBD-AI](https://github.com/HAISymbiosis/EBD-AI) is an
**independent** public copy of [Fuminides/ex-fuzzy](https://github.com/Fuminides/ex-fuzzy),
not a GitHub-linked fork. Product name in the README: **EBD Toolbox**.
Author: Prof. Javier Andreu-Perez. It is a rebase of Ex-Fuzzy; acknowledge
that in user-facing text. Older docs and comments may still say BDI.

Remote `upstream` (if configured) points at `Fuminides/ex-fuzzy`. Bring
library updates across, but do not take upstream's README, package name, or
agent notes, and do not convert this repo into a GitHub fork. If a change
touches `FitRuleBase`, `_fitness.py`, `_array_fitness.py`, or firing kernels,
the evaluators must still agree with each other bit for bit.

As of 2026-10-03, `upstream/main` is `93bb437` (Ex-Fuzzy 3.3.0: macro F1
objective, honoured `tournament_size`, `algorithm='nsga2'`). That is merged
into `main`. The temporal library code in `ex_fuzzy/temporal.py` stays. Do
not merge `fix/python39-pymoo-install` (Python 3.9) or `gh-pages` into `main`.

## Two packages, two versions

`pip install ebdai` installs both. They are versioned separately.

| Import | Path | Version | Role |
| --- | --- | --- | --- |
| `ebdai` | `ebdai/` | **1.0.1** (`ebdai/_version.py`, pip metadata) | BDI expansions |
| `ex_fuzzy` | `ex_fuzzy/` | **3.3.0** (`ex_fuzzy/_version.py`) | Rebased Ex-Fuzzy API |

Do **not** re-export Ex-Fuzzy names from `ebdai`. Classes and enums must exist
once per process. Register new `ebdai` modules in `ebdai/__init__.py`
`_SUBMODULES` / `_EXPORTS`, same lazy pattern as `ex_fuzzy`.

## Bias work (on `main`)

Workshop source:
[rferper/WorkshopIgualdad2025](https://github.com/rferper/WorkshopIgualdad2025)
(Ex-Fuzzy 2.1.3 notebooks). Ported to current `ex_fuzzy` 3.3.0.

- Library: `ebdai/bias.py`, `ebdai/datasets.py`, CSVs in `ebdai/data/`
- Public names: `outcome_rates_by_group`, `fairness_report`,
  `winning_rules_by_group`, `reweigh_weights`, `weighted_mcc_loss`,
  `fairness_regularized_loss`, `load_titanic`, `load_heart_failure`,
  `load_loan_approval`
- Demos: `Demos/09_bias_titanic.ipynb`, `10_bias_heart_failure.ipynb`,
  `11_bias_loan_fairness.ipynb` (executed outputs stored). Each starts with the
  workshop Colab install cell, unchanged:

  ```
  !git clone -q https://github.com/HAISymbiosis/EBD-AI.git
  %cd EBD-AI
  !pip install -q .
  ```

  Open-in-Colab links are in `Demos/README.md`.
- Tests: `tests/test_ebdai_bias.py`
- Docs: `docs/source/user-guide/ebdai.rst`, `docs/source/api/ebdai.rst`

Verified in the `developer` mamba env: unit tests passed; all three notebooks
ran with `python Demos/run_notebooks.py`.

## Fuzzy Classroom teaching site (moved out)

The classroom website, its exported models, the training server and the tests
that check the JavaScript engine against `ex_fuzzy` moved to the separate
private repository `jandreu/EBD-AI-Classroom` on 2026-11-05, and were removed
from this repository's history. The deployment notes went with them: the live
site runs that checkout, so nothing under `Demos/` here is served.

If the classroom needs changing, clone `jandreu/EBD-AI-Classroom` and install
this package into its environment; it imports `ex_fuzzy` from here.

## fNIRS analysis demo (since 2026-10-15)

`Demos/fnirs_analysis/`: the "fNIRS Fuzzy Rule Explainer" from
[arkabite/Dissertation](https://github.com/arkabite/Dissertation) (FastAPI
backend, React/Vite page), imported with its
legacy commented-out code removed, file paths moved to `data/`, the API under
`/api`, the built page served by `backend.py` from `site/` (gitignored; build
with `npm ci && npm run build` in `frontend/`), and `MainFold.py` ported to
`train_model.py` on this toolbox. The shipped rules are the dissertation's own
(Ex-Fuzzy 2.1.5); the toolbox rerun is reported in the demo README, not shipped,
because `regression_test.py` checks facts about those rules. Written answers
come from the Claude Code CLI logged in on the server (`claude_cli.py`: `claude -p`
with no tools, MCP servers, plugins, hooks or saved sessions; text protocol for
the visualisation tool calls), replacing the original Ollama Cloud client; visitors
whose gate role (`X-Gate-Role`) is not admin get the Venice API instead
(`venice_client.py`, same interface); without a model the answers are rule lists. Answers also get passages from 16 open-access
(CC BY) n-back/fNIRS papers in `data/papers.json` (built by `build_paper_corpus.py`
from Europe PMC; BM25 search in `paper_search.py`), cited and listed under
"Background reading"; citations of papers not given are caught as before. A
"Your data" toggle lets a visitor upload an Ex-Fuzzy rule base (printed text or
JSON), notes and reference documents into an in-memory per-tab workspace
(`workspace.py`, `X-Workspace` header); `pypdf` reads uploaded PDFs. Needs `fastapi uvicorn` (and `httpx` for the
tests) beside the toolbox; the `developer` env has them. Tests: `tests/test_fnirs_analysis.py`. On the public
host it runs as the `fnirs-demo` unit on localhost:8010 behind the same Caddy
password gate; its settings (`CLAUDE_MODEL`, `CLAUDE_MAX_PARALLEL`, ...) are in
that unit's environment file outside the repository.

## Environment

Install libraries in the **mamba env `developer`**
(`/home/ubuntu/miniforge3/envs/developer`). As of 2026-10-07 it holds
Python 3.13, numpy, pandas, scikit-learn, matplotlib, pymoo, pytest and an
editable `ebdai`, plus Node.js and Playwright with headless Chromium for
checking the web demos in a browser. `gh` (GitHub login **jandreu**) is not
installed on the host at present.

```bash
conda activate developer
pip install -e ".[test]"
python -m pytest tests/test_ebdai_bias.py
```

## User-facing docs vs this folder

Sphinx and README are for humans. Keep agent conventions here. User-facing
ebdai guide: `docs/source/user-guide/ebdai.rst`.

## Reasonable next work (not promised)

- More EBD method families under `ebdai.<name>` (concept bottlenecks,
  computing with words, prototypes) without renaming `ex_fuzzy`
- Publish `ebdai` 1.0.1 to PyPI when asked
- GitHub Settings → social preview: `docs/source/_static/og-image.png`
- After Pages rebuild, confirm `/user-guide/ebdai.html` and `/api/ebdai.html`
