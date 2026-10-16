# Domain background (from the project's research proposal literature review)

This is context from the dissertation's own literature review. It explains
what fNIRS is and why the project looks the way it does, and can be used to
explain the patterns in the model's rules. Which channels the rules use comes
only from the rules themselves.

## What fNIRS measures

Functional near-infrared spectroscopy (fNIRS) is a non-invasive
neuroimaging technique that shines near-infrared light into the scalp and
measures the reflected signal to track changes in oxygenated (HbO) and
deoxygenated (HbR) haemoglobin in the cerebral cortex. Because active brain
regions consume more oxygen, these haemoglobin changes are used as an
indirect proxy for local neural activity. Recordings are time-series in
nature, span many channels tied to different scalp locations, and are
commonly affected by noise, motion artefacts, and physiological variation —
which is why preprocessing (filtering, normalisation, feature extraction)
happens before model training.

## What "tmb" means (this project's own feature engineering, not the source dataset)

Every feature name in this model is prefixed `tmb_s1_ch*` or `tmb_s2_ch*`
(e.g. `tmb_s1_chAF7`). **`tmb` stands for Task Minus Baseline.** This is
defined directly in this project's own preprocessing code
(`002_Block_Averaging.ipynb`):

    new_row['tmb'] = task_mean - rest_mean

For each trial and each channel, it is the mean signal recorded during the
task period minus the mean signal recorded during that trial's rest
(baseline) period — computed separately for `s1` (HbO) and `s2` (HbR).
This is NOT a term from Shin et al. (2018), the source dataset paper: their
own file-naming convention uses `cnt`/`mrk`/`mnt` with no "tmb" anywhere.
It was introduced later, specifically in this project's own data-processing
pipeline, as the block-averaging/feature-engineering step that turns raw
continuous recordings into the one-row-per-trial features the fuzzy rules
are actually fitted on.

## Why this project uses an interpretable model instead of a black box

fNIRS analysis is frequently carried out with black-box models (deep
networks, ensembles) that give little insight into which channels or
signal properties actually drove a prediction. This project's motivation is
to use an interpretable fuzzy rule model (ExFuzzy) instead, so predictions
can be inspected and questioned rather than accepted at face value, and to
let non-expert users query those rules conversationally rather than reading
a static plot or table.

## Prior related work (cited context, not this project's own finding)

The proposal cites earlier fNIRS work on cognitive workload / mental-state
classification using prefrontal signals, and on combining EEG with fNIRS
for workload measurement. This establishes fNIRS as a workable input for
workload-style classification *in general* — it does not specify which
individual channel matters for *this* project's dataset or classifier, and
should not be cited as if it did.
