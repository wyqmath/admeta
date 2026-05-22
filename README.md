# APOE ε4-Associated Hippocampal Atrophy Trajectories Across the Alzheimer's Disease Continuum: A Systematic Review, Meta-Analysis, and Longitudinal Validation

## Project Structure

```
├── meta/
│   └── meta.R
├── ADNI/
│   └── adni_analysis.py
├── NACC/
│   └── nacc_analysis.py
└── data/                (not included)
```

## Code Description

### `meta/meta.R`

R script for systematic review and meta-analysis of APOE ε4 effects on hippocampal volume.

- Random-effects meta-analysis (REML, Knapp-Hartung adjustment)
- Subgroup analyses by diagnosis (AD, MCI, CN) and ε4 dosage (homozygotes, heterozygotes)
- ICV-corrected vs uncorrected stratification
- Meta-regression (age, sex, CN-subgroup age)
- Publication bias assessment (trim-and-fill, funnel plots)
- Sensitivity analyses (leave-one-out, Baujat diagnostics, multiverse, segmentation method subgroup, ADNI-overlap exclusion)

### `NACC/nacc_analysis.py`

Python script for longitudinal analysis of the NACC cohort.

- Gene-dose linear mixed-effects model (Time × APOE4 dosage) with Time × Diagnosis interaction
- Random slopes and intercepts by subject
- Cross-sectional and longitudinal quality control
- CN-only sensitivity analysis
- Trajectory and atrophy rate visualizations

### `ADNI/adni_analysis.py`

Python script for ADNI cohort analysis.

- Gene-dose linear mixed-effects model with Time × Diagnosis interaction
- Two-cohort fixed-effect meta-analysis (NACC + ADNI pooled estimates)
- CSF biomarker interaction analysis (Aβ42, p-Tau, t-Tau × APOE4)
- Single-factor and joint models with FDR correction
- Forest plots, trajectory plots, and biomarker distribution figures

## Dependencies

### R

- tidyverse, meta, data.table, gridExtra, ggrepel

### Python

- pandas, numpy, statsmodels, matplotlib, seaborn, scipy

## Data

Data are derived from public databases (ADNI and NACC) and require application for access. Processed datasets will be made available upon publication.

**Status:** Coming soon

**Contact:** Minnuo Cai (cmn@stu.xju.edu.cn)
