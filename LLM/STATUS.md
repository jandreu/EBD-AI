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

## Fuzzy Classroom teaching site (on `main`)

`Demos/fuzzy_classroom/`: static teaching website plus `build_models.py`
(exports fitted models to `site/models/*.js`) and `serve.py` (stdlib server
with `/api/train`). Datasets: penguins (default), iris, football positions
(FIFA 22, 2,650 players, 7 classes; `data/make_football_csv.py`) and breast
cancer. Each dataset can
override `DEFAULT_SETTINGS` in `DATASET_SETTINGS` (football: 20 rules, 3 ants,
300 gens, pop 100, seed 12, `tolerance=0.01`; cancer: 2 ants, seed 8). Breast
cancer (UCI Wisconsin Original, `data/make_cancer_csv.py`) sets `'scale': [1, 10]`
in its loader: its sets are built from an evenly spaced grid, not data
quantiles, because most scores are 1. Retraining accepts `n_partitions`
2-5, `shape` trapezoid/triangular/gaussian (triangular needs 3 sets) and
`tolerance`. `site/fuzzy.js` ports ex-fuzzy's Type-1 inference (clipping,
trapezoid epsilon, Gaussian `exp(-z*z)`, product t-norm, `ds_mode` 0/1, first-max
winner).
If inference semantics change, rebuild the models and keep
`tests/test_fuzzy_classroom.py` passing; its JS parity test needs `node` on PATH
and skips otherwise. The page also reports parity live in its header badge.
`serve.py` trains in a spawn-context process pool (workers = cores - 1,
admission limit max(64, 4 x workers), 503 when full, 504 after `--timeout`,
SIGTERM stops the workers); the server and page keep no state between visitors.
`load_check.py` simulates a class against a running server.

### Public site (since 2026-10-07)

Since 2026-10-15 the homepage of https://human-centred.ai is the fNIRS analysis
demo (below), not the classroom; the classroom units are stopped and disabled,
and the notes in this section describe how to bring them back. The development
host served this checkout's Fuzzy Classroom as the homepage of
https://human-centred.ai (`www` redirects to it). The server configuration is
outside the repository and, by the user's choice, not versioned:

- systemd units run `serve.py` on localhost with the `developer` env's Python
  (`fuzzy-classroom` on port 8000; their worker counts and CPU pinning are server
  configuration).
- Caddy (`/etc/caddy/Caddyfile`) terminates HTTPS (Let's Encrypt, automatic
  renewal) and reverse-proxies to it. Keep `serve.py` on localhost; it has no
  TLS or authentication of its own.
- Visitors can retrain (`/api/train`), bounded by the worker pool and admission
  limit. To disable it, add `respond /api/train 403` to the Caddyfile.

Consequences for agents working here: edits under `site/` are live as soon as
they are saved; changes to `serve.py`, `build_models.py` or `ex_fuzzy` need
`sudo systemctl restart fuzzy-classroom`; switching branches changes the live
site. Logs: `journalctl -u fuzzy-classroom` and `journalctl -u caddy`.

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
"Background reading"; citations of papers not given are caught as before. Needs `fastapi uvicorn` (and `httpx` for the
tests) beside the toolbox; the `developer` env has them. Tests: `tests/test_fnirs_analysis.py`. On the public
host it runs as the `fnirs-demo` unit on localhost:8010 behind the same Caddy
password gate; its settings (`CLAUDE_MODEL`, `CLAUDE_MAX_PARALLEL`, ...) are in
that unit's environment file outside the repository.

## Environment

Install libraries in the **mamba env `developer`**
(`/home/ubuntu/miniforge3/envs/developer`). As of 2026-10-07 it holds
Python 3.13, numpy, pandas, scikit-learn, matplotlib, pymoo, pytest and an
editable `ebdai`, plus Node.js (so the JS parity tests in
`tests/test_fuzzy_classroom.py` run) and Playwright with headless Chromium (for
checking the classroom page in a browser). The live site runs from it, so test
changes to it before restarting the service. `gh` (GitHub login **jandreu**)
is not installed on the host at present.

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
