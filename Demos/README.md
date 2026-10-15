# BDI demos

Ten notebooks walk through the library, from a first classifier to
uncertainty and bias in data and inference. They are
executed with their outputs stored, so they read well on GitHub, and
every one of them runs in well under a minute on a laptop. Open them
with Jupyter after `pip install ebdai` (or `pip install -e .` from a
checkout), or read them online.

| Notebook | What it shows |
| --- | --- |
| [01 Getting started](01_getting_started.ipynb) | Fit, score, read the rules, probabilities, per-sample explanations, partition plots. |
| [02 Scikit-learn integration](02_scikit_learn_integration.ipynb) | Titanic data with categorical columns and missing values: a `Pipeline` with imputation, cross-validation, `GridSearchCV`. |
| [03 Rules and partitions](03_rules_and_partitions.ipynb) | Fuzzy sets and variables by hand, handmade rules, fixed versus optimised partitions, Type-2 sets, validation, inference modes, LaTeX export, saving and loading. |
| [04 Controlling the search](04_controlling_the_search.ipynb) | Budget and early stopping, custom objectives, checkpoints, mined candidate rules, and a comparison of every classifier on the Wine data. |
| [05 Regression](05_regression.ipynb) | `BaseFuzzyRulesRegressor` with crisp and Mamdani consequents on California housing, then inference by hand. |
| [06 Uncertainty](06_uncertainty.ipynb) | Conformal prediction sets with coverage evaluation and rule-level explanations, next to FERL and DeepFERL evidential outputs. |
| [07 Robustness](07_robustness.ipynb) | Pattern stability over repeated fits, and permutation and bootstrap validation of a fitted rule base. |
| [09 Bias in the data (Titanic)](09_bias_titanic.ipynb) | `ebdai` outcome rates and winning-rule firings by sex on the workshop Titanic table. |
| [10 Bias in heart-failure labels](10_bias_heart_failure.ipynb) | Same bias tools on the heart-failure death data. |
| [11 Bias in inference and mitigation (loans)](11_bias_loan_fairness.ipynb) | Demographic parity, Kamiran–Calders reweighing, and a fairness-regularised genetic loss. |

Notebooks 1 and 5 download Titanic and California housing through scikit-learn
on the first run and cache them in your home directory. Notebooks 9–11 load
tables shipped in `ebdai/data/` from the
[WorkshopIgualdad2025](https://github.com/rferper/WorkshopIgualdad2025) workshop.

Those three bias notebooks start with the workshop install cell for Google Colab:

```
!git clone -q https://github.com/HAISymbiosis/EBD-AI.git
%cd EBD-AI
!pip install -q .
```

Open them from GitHub:

- [09 Bias in the data (Titanic)](https://colab.research.google.com/github/HAISymbiosis/EBD-AI/blob/main/Demos/09_bias_titanic.ipynb)
- [10 Bias in heart-failure labels](https://colab.research.google.com/github/HAISymbiosis/EBD-AI/blob/main/Demos/10_bias_heart_failure.ipynb)
- [11 Bias in inference and mitigation (loans)](https://colab.research.google.com/github/HAISymbiosis/EBD-AI/blob/main/Demos/11_bias_loan_fairness.ipynb)

[`fuzzy_classroom/`](fuzzy_classroom/README.md) is an interactive teaching website: pick a
Palmer penguin, an iris, a football player or a breast cancer sample and watch fuzzification, rule firing, dominance weighting, aggregation
and the winning rule step by step, switch rules off, and retrain with ex-fuzzy. Open
`fuzzy_classroom/site/index.html`, or run `python Demos/fuzzy_classroom/serve.py` to enable retraining.

[`fnirs_analysis/`](fnirs_analysis/README.md) is a web app that explains a fuzzy
rule classifier trained on fNIRS brain recordings from an n-back memory task: ask
about the rules in plain English and see the channels on a scalp map, a 3D brain
and a connectivity diagram. It comes from the
[arkabite/Dissertation](https://github.com/arkabite/Dissertation) project; run
`python Demos/fnirs_analysis/backend.py` after building its page (see its README).

## Run the EBD Classroom on your own computer

The classroom in [`fuzzy_classroom/`](fuzzy_classroom/README.md) runs on any
Windows or Linux computer; on macOS, follow the Linux steps. There are two
ways to use it:

- **Just open the page.** No installation: everything works except retraining.
- **Run the small server.** Needs Python, and adds the "Train your own rules"
  panel, which reruns ex-fuzzy's genetic search on your computer.

### What you need

| What | Needed for | Windows | Linux |
| --- | --- | --- | --- |
| A web browser (Chrome, Edge, Firefox or Safari) | Always | Already installed | Already installed |
| The repository files | Always | Git, or **Code → Download ZIP** on GitHub | `git`, or the ZIP |
| Python 3.11 or newer (3.12 or 3.13 recommended) | Retraining, rebuilding models | [python.org](https://www.python.org/downloads/) installer (tick **Add python.exe to PATH**), or `winget install Python.Python.3.12` | Your distribution's `python3` and `python3-venv`, if new enough |
| The toolbox and its libraries (numpy, pandas, scikit-learn, pymoo, matplotlib) | Retraining, rebuilding models | Installed by `pip install -e .` below | Installed by `pip install -e .` below |
| Node.js 18 or newer | Optional: the tests that compare the page's engine with ex-fuzzy | [nodejs.org](https://nodejs.org/) LTS installer | Package manager or nodejs.org |
| An internet connection | Installing; the page's fonts (it falls back to system fonts offline); rebuilding the data tables | | |

The toolbox itself supports Python 3.10, but the classroom server needs 3.11 or
newer: its pool of training processes uses a Python 3.11 feature. Check with
`python --version` (Windows) or `python3 --version` (Linux). Ubuntu 22.04, for
example, ships Python 3.10; there, install a newer Python first, for instance
with [Miniforge](https://github.com/conda-forge/miniforge)
(`conda create -n ebd python=3.12`, then `conda activate ebd`).

`pip install ebdai` alone is not enough for the classroom: the website and its
data live in this repository's `Demos/` folder, so you need a copy of the
repository.

Any recent laptop will do. By default the server starts one training process
per CPU core, minus one, and each uses about 175 MB of memory; pass
`--workers 2` on a small machine.

### Option 1: open the page, without installing anything

Download the repository (or clone it) and open
`Demos/fuzzy_classroom/site/index.html` in your browser, for example by
double-clicking it. The five steps, the rule lab, the decision map and the
exercises all work. The training panel shows the equivalent Python code but
cannot run it.

### Option 2: run the server, with retraining

**Windows** (PowerShell):

```powershell
git clone https://github.com/HAISymbiosis/EBD-AI.git
cd EBD-AI
py -3.12 -m venv .venv                  # or: py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python Demos\fuzzy_classroom\serve.py
```

If PowerShell refuses to run `Activate.ps1`, allow local scripts once with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt and
`.venv\Scripts\activate.bat` instead.

**Linux** (bash):

```bash
sudo apt install git python3 python3-venv   # Debian and Ubuntu; use dnf, pacman, ... elsewhere
git clone https://github.com/HAISymbiosis/EBD-AI.git
cd EBD-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python Demos/fuzzy_classroom/serve.py
```

Then open <http://127.0.0.1:8000>. The server takes a few seconds to start,
while its training processes load ex-fuzzy. Stop it with **Ctrl+C**. Next time,
activate the environment again (`.venv\Scripts\Activate.ps1` or
`source .venv/bin/activate`) from the `EBD-AI` folder and run `serve.py`.

Useful options (`python Demos/fuzzy_classroom/serve.py --help` lists them all):

- `--port 8080` if port 8000 is taken.
- `--workers 2` to use fewer CPU cores.
- `--host 0.0.0.0` to let others on the same network open
  `http://YOUR-COMPUTER'S-IP:8000`. Windows will ask whether to let Python
  through its firewall. The server has no login and no encryption: use it on a
  classroom or home network, never on the open internet.

Open a dataset directly by adding its name to the address:
`http://127.0.0.1:8000/#penguins`, `#iris`, `#football` or `#cancer`.

### Optional: rebuild the models and run the tests

With the environment active, from the `EBD-AI` folder:

```bash
python Demos/fuzzy_classroom/build_models.py            # retrain the shipped models (a few seconds)
pip install -e ".[test]"                                # adds pytest
python -m pytest tests/test_fuzzy_classroom.py          # checks the models, the engine and the server
```

The tests that compare the page's JavaScript engine with ex-fuzzy run only when
Node.js is installed (`node --version` works); otherwise they are skipped. The
data tables ship with the repository. `Demos/fuzzy_classroom/data/make_football_csv.py`
and `make_cancer_csv.py` rebuild them from their original sources, which needs
an internet connection.

`evox_backend_demo.py` compares the PyMoo and EvoX backends and reports
whether EvoX and CUDA are available; run it as `python Demos/evox_backend_demo.py`.

To refresh the stored outputs after changing the library, run
`python Demos/run_notebooks.py` from the repository root.
