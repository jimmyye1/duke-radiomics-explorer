# Radiomics Explorer

An interactive local dashboard and reproducible conventional feature-selection pipeline for radiomics cohorts. The bundled sample dataset is the Duke breast MRI radiomics cohort, but the dashboard can load any dataset produced by this repo's analysis scripts (see "Load a different dataset" below).

## What the analysis does

- Joins 772 unique radiomics patients to the Duke clinical workbook by exact patient ID.
- Removes all clinical fields from the predictor matrix.
- Evaluates three endpoints supplied by the clinical data, shown in this order:
  - overall survival (time to death from any cause; disease-specific/disease-free/progression-free fields exist in the workbook but are empty for every patient, so only overall survival is modeled);
  - four-class molecular subtype;
  - pathologic complete response using the workbook's strict overall-response field as a proxy.
- Screens radiomic features inside training folds — ANOVA F ranking for the two classification endpoints, univariate concordance-index ranking for survival.
- Removes redundancy at absolute Spearman correlation of 0.90 or greater.
- Fits a class-weighted L1 logistic regression (subtype/pCR) or an elastic-net Cox proportional-hazards model (survival), with inner-loop tuning.
- Estimates performance with repeated nested cross-validation and bootstrap intervals.
- Ranks features by outer-fold selection frequency, coefficient magnitude, and univariate effect (or hazard ratio, for survival).

The dashboard contains only aggregate results. It does not contain patient identifiers or patient-level predictions.

## Current result (bundled sample dataset)

- Overall survival: a real, moderate association (concordance index 0.665, 95% CI 0.581–0.740; log-rank p = 0.001 between predicted risk groups) from 560 evaluable patients with only 50 observed deaths — treat as hypothesis-generating given that event count.
- Molecular subtype: weak discrimination (macro out-of-fold ROC AUC 0.548); four features were selected in at least half of outer folds.
- pCR proxy: no reproducible radiomics-only signature (ROC AUC 0.428; no feature passed FDR; the final full-cohort sparse model selected zero active features).

These are exploratory internal-validation results, not evidence of clinical utility. Numbers shown in the dashboard update automatically if you load a different dataset (see below).

## Run the dashboard locally

Prerequisites: Node.js 22.13 or later.

```bash
npm install
npm run dev
```

Open the local URL shown by the development server.

## Load a different dataset

Click **Load dataset** in the top bar of either page to view a different cancer's cohort without rebuilding the app. Run `run_analysis.py` (and optionally `run_feature_explorer.py`) locally against a different dataset in the same format, then upload the resulting `analysis.json` (required) and `feature-explorer.json` (optional) through the dialog.

- Files are parsed entirely in your browser — nothing is sent to a server.
- The uploaded dataset is kept in `localStorage` and persists across reloads and navigation between the dashboard and the Feature Statistics page, until you click "Reset to sample dataset" or clear it yourself.
- The dashboard always expects the same three endpoints — `survival`, `subtype`, `pcr` — regardless of the cancer type; only the underlying numbers, feature names, and clinical group labels change.

## Re-run the analysis

The Python environment needs `numpy`, `pandas`, `scipy`, `scikit-learn`, and `lifelines` (the last for the survival endpoint's Cox proportional-hazards model, concordance index, and Kaplan–Meier curves).

```bash
python analysis/run_analysis.py \
  --radiomics /path/to/merged_patient_Data_filtered.csv \
  --clinical /path/to/duke_brca_clin.xlsx \
  --output-dir app/data
```

This takes a few minutes — the survival endpoint's nested Cox cross-validation is the slow part. Then copy the three generated ranking CSVs from `app/data/` to `public/data/` so the dashboard download links use the refreshed files.

## Feature Statistics page

A companion page at `/feature-explorer` audits data quality across all 2,064 radiomic features and gives a full distribution, group-comparison, and univariate-discrimination breakdown for the top-50 stability-ranked features per endpoint (the same features shown in the main dashboard's Feature registry). It reads `app/data/feature_rankings_survival.csv` / `feature_rankings_subtype.csv` / `feature_rankings_pcr.csv`, so regenerate those first with `run_analysis.py`, then run:

```bash
python analysis/run_feature_explorer.py \
  --radiomics /path/to/merged_patient_Data_filtered.csv \
  --clinical /path/to/duke_brca_clin.xlsx \
  --output-dir app/data
```

This writes `app/data/feature-explorer.json`. It depends only on `numpy` and `pandas` for the subtype/pCR endpoints (no `scipy`/`scikit-learn` needed there); the survival endpoint's per-feature hazard-ratio confidence interval uses `lifelines`, same as `run_analysis.py`.

## Key files

- `analysis/run_analysis.py`: auditable analysis pipeline.
- `analysis/run_feature_explorer.py`: cohort/data-quality audit and per-feature statistics for the Feature Statistics page.
- `app/data/analysis.json`: aggregate dashboard data.
- `app/data/feature-explorer.json`: data for the Feature Statistics page.
- `app/dashboard.tsx`: dashboard interactions and charts.
- `app/feature-explorer/`: the Feature Statistics page.
- `app/dataset-context.tsx`: loads/validates/persists the active dataset (sample or uploaded) and shares it via React Context.
- `app/ui.tsx`: small UI pieces (`InfoTip`, chart colors, the dataset-upload dialog) shared across pages.
- `app/globals.css`: responsive visual design.
- `public/data/feature_rankings_*.csv`: full feature-ranking downloads.

## Interpretation limits

The cohort has no external validation set, repeat scans, repeat segmentations, scanner/protocol metadata, or acquisition-site variables. ICC filtering, perturbation robustness, and ComBat harmonization therefore cannot be estimated. The pCR proxy should be checked against the original pathology definition before any downstream study. The survival endpoint has only 50 observed deaths, which limits how precisely the concordance index and hazard ratios can be estimated regardless of sample size, and only overall survival could be modeled since the workbook's disease-specific/disease-free/progression-free fields are empty for every patient.
