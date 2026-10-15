# fNIRS analysis demo

A web app that explains a fuzzy rule classifier trained on brain recordings.
Twenty-six people did an n-back working-memory task while functional
near-infrared spectroscopy (fNIRS) measured oxygenated (HbO) and deoxygenated
(HbR) haemoglobin at 36 scalp channels. A Type-1 fuzzy rule base learns to tell
the easy 0-back blocks from the harder 2/3-back blocks. The page lets you ask
questions about those rules in plain English, and shows the channels involved
on a 2D scalp map, a 3D brain and a connectivity diagram.

The code comes from the dissertation project
[arkabite/Dissertation](https://github.com/arkabite/Dissertation) ("fNIRS Fuzzy
Rule Explainer"). Here it runs on the EBD Toolbox's `ex_fuzzy` and is served as a
single page plus API. The recordings come from the open-access dataset of Shin
et al. (2018), *Simultaneous acquisition of EEG and NIRS during cognitive tasks
for an open access dataset*, Scientific Data 5, 180003,
<https://doi.org/10.1038/sdata.2018.3>.

## How it works

```
data/block_avg_tmb.csv ──train_model.py──▶ data/extracted_rules_final_raw.json ─┐
                                           data/grouped_kfold_results_raw.csv    │
data/brain_mapping.json ──build_channel_atlas.py──▶ data/channel_atlas.json ─────┤
                                                                                 ▼
browser ◀── site/ (React build of frontend/) ◀── backend.py (FastAPI) ── rag_core.py ── claude CLI (optional)
                                                     │                      └─ viz_tools.py
                                                     └─ /api/health, /api/rules, /api/atlas,
                                                        /api/visualize/{topomap,chord}, POST /api/ask
```

- **Rules and evaluation** (`train_model.py`, ported from `MainFold.py`):
  five-fold cross-validation grouped by subject, so no person appears in both
  training and test data. Each fold fits three seeds and votes. A final rule
  set is then fitted on all subjects; the seed closest to the median training
  MCC is kept.
- **Answers** (`rag_core.py`): a question is first sorted into one of three
  kinds. Questions about this model get the best-matching rules (keyword
  retrieval, top 5). General fNIRS questions get curated background from
  `data/literature.json`. Anything else is refused without calling a model.
  When the Claude Code CLI is installed and logged in on the server, Claude
  words the answer from the retrieved rules, may request a visualisation, and
  a consistency check repairs or flags answers that contradict the rules.
  Without it, the answer lists the matching rules as they are.
- **Claude** (`claude_cli.py`): each model call runs `claude -p` with the
  CLI's own login, so the demo holds no API key. Visitors' questions reach the
  model, so every call is locked down: no tools (no shell, files or web), no
  MCP servers or connectors, no skills, hooks or CLAUDE.md (`--safe-mode`), no
  saved session, and an empty working folder. The original project used Ollama
  Cloud; `claude_cli.py` keeps its chat interface, and asks for pictures in
  text because the CLI cannot call the app's functions.
- **Venice** (`venice_client.py`): the same interface over the Venice API
  (OpenAI-compatible; default model `zai-org-glm-4.7-flash`, Venice's own
  system prompt and the model's reasoning trace switched off). Behind a
  password gate that sends `X-Gate-Role` (as on human-centred.ai), the admin
  password's visitors get the Claude CLI and every other visitor gets Venice,
  with no crossover; run locally without a gate, the Claude CLI is used if it
  is available, else Venice. `/api/health` reports the model for the visitor
  who asks, and the question log records which one answered.
- **Consistency check**: besides wrong claims about which rules use a channel,
  it catches an answer that gives a rule channel the wrong haemoglobin signal
  (HbO for a channel the rules use as HbR, or the reverse); the answer is then
  rewritten once, or flagged.
- **Background from papers** (`paper_search.py`, `data/papers.json`): every
  answer also gets the four or five best-matching passages from 16 open-access
  papers on the n-back task, working memory and fNIRS (BM25 keyword search,
  with British and American spellings and HbO/HbR synonyms unified). Claude
  explains the brain signals with them and cites them as (Author et al., year);
  a citation of any paper it was not given is caught, and the answer rewritten
  once or flagged. The papers appear under "Background reading" below the
  answer. For questions about the model they are explicitly general
  background, never findings of this model.
- **Pictures** (`viz_tools.py`): the plotted points always come from the rules
  and the channel atlas, never from the language model.

## Run it

From the repository root, in an environment with the toolbox installed
(`pip install -e .`), and for written answers with the
[Claude Code CLI](https://docs.claude.com/en/docs/claude-code) installed and
logged in (`claude auth status`) for the user that runs the server:

```bash
pip install fastapi uvicorn
cd Demos/fnirs_analysis/frontend && npm ci && npm run build && cd -   # Node.js 18+; writes Demos/fnirs_analysis/site/
python Demos/fnirs_analysis/backend.py                                # http://127.0.0.1:8000/
```

The API documentation is at `/api/docs`. For frontend work, run
`npm run dev` in `frontend/` as well: it serves the page on port 5173 and
forwards `/api` to the backend on port 8000.

### Optional settings (environment variables)

| Variable | Effect |
| --- | --- |
| `CLAUDE_MODEL` | Model alias or ID for the answers (default `haiku`: about 4 s per call, 7 to 10 s per answer) |
| `CLAUDE_MAX_PARALLEL` | Claude calls at once (default 4); a question that waits longer than `CLAUDE_TIMEOUT` gets the rules-only answer |
| `CLAUDE_TIMEOUT` | Seconds per call (default 60) |
| `CLAUDE_BIN` | Path of the `claude` CLI if it is not on `PATH` or in `~/.local/bin` |
| `VENICE_API_KEY` | Venice API key; without it visitors who are not the admin get rule-list answers. Keep it on the server only |
| `VENICE_MODEL` | Venice model ID (default `zai-org-glm-4.7-flash`: about 4 to 8 s per answer) |
| `VENICE_MAX_PARALLEL`, `VENICE_TIMEOUT` | Requests at once (default 8) and seconds per request (default 60) |
| `LIVE_LITERATURE=1`, `OPENALEX_API_KEY` | Also look up papers on OpenAlex for general questions. Off by default; these papers are marked as unvetted. |
| `INTERACTIONS_DB_PATH` | SQLite file that logs each question, answer and latency (default `interactions.db` in this folder, not tracked by git). |

## Retrain

```bash
python Demos/fnirs_analysis/train_model.py --out /tmp/fnirs-results   # about 40 s on one core
```

Without `--out` it overwrites the results in `data/`, which the web app reads.
The shipped results are the dissertation's own, fitted with Ex-Fuzzy 2.1.5:
6 rules, test accuracy 0.691 ± 0.070 and MCC 0.408 ± 0.111 over the five
folds. With the EBD Toolbox (ex_fuzzy 3.3.0, whose search maximises macro F1
rather than MCC) the same protocol gave 0.671 ± 0.036 and MCC 0.364 ± 0.075,
with 5 final rules (15 October 2026). The genetic search is random, so rule
sets differ from seed to seed; `rule_stability_across_seeds` in the results
records how often each rule recurred.

`build_channel_atlas.py` rebuilds `data/channel_atlas.json` and needs
`pip install mne`.

## Background papers

`build_paper_corpus.py` downloads the papers listed in its `PAPERS` table from
Europe PMC and writes `data/papers.json` (abstract, introduction, discussion
and conclusion paragraphs; in-text citations removed). It keeps only papers
licensed CC BY, CC BY-SA or CC0, and every passage carries its paper's
reference, link and licence. To add a paper, put its PMCID in `PAPERS` and run

```bash
python Demos/fnirs_analysis/build_paper_corpus.py
```

The current set (15 October 2026, all CC BY): Shin et al. 2018 (the source
dataset); Huang et al. 2025 (review of the n-back task); Schmiedek et al. 2014;
Lara & Wallis 2015; Fishburn et al. 2014; Herff et al. 2013; Yeung & Han 2023;
Vermeij et al. 2012; Csipo et al. 2021; Unni et al. 2017; Miri Ashtiani &
Daliri 2023; Liu et al. 2017; Naseer & Hong 2015; Pinti et al. 2020; Yücel et
al. 2021; Kirilina et al. 2012. Full references are in `data/papers.json`.

## Tests

```bash
python Demos/fnirs_analysis/regression_test.py   # offline, no key needed; also run by pytest tests/test_fnirs_analysis.py
```

## Files

| Path | Purpose |
| --- | --- |
| `backend.py` | FastAPI app: the API under `/api` and the built page at `/` |
| `claude_cli.py` | Locked-down `claude -p` calls behind the chat interface `rag_core.py` uses |
| `venice_client.py` | The same chat interface over the Venice API |
| `paper_search.py` | BM25 search over the background papers' passages |
| `build_paper_corpus.py` | Downloads the open-access papers and writes `data/papers.json` |
| `rag_core.py` | Retrieval, scope check, prompts, consistency check, fallback answers |
| `viz_tools.py` | Topomap, chord and neighbour data from rules and atlas |
| `literature_live.py` | Optional OpenAlex / Semantic Scholar lookup |
| `train_model.py` | Grouped cross-validation and the final rule set |
| `build_channel_atlas.py` | Channel positions and regions (needs MNE) |
| `regression_test.py` | Offline checks of the backend with a fake language model |
| `frontend/` | React and Vite source of the page |
| `data/block_avg_tmb.csv` | One row per trial: subject, trial, event, 72 task-minus-baseline features, `stimulus` |
| `data/*.json`, `data/*.md` | Rules, cross-validation results, channel atlas, region glossary, curated literature and background |

## Limits

This is a research prototype, not a medical tool. Per-rule accuracy and
dominance describe the fitted rule set, not independent tests. A rule is an
association in this dataset, not evidence that a brain region causes the task
state. Language-model answers can be wrong, so the rules behind each answer
stay visible under "Rules used".
