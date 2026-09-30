# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A read-only research dashboard ("Radiomics Explorer") that presents the results of a leakage-aware radiomics feature-selection pipeline. The bundled sample dataset is the Duke breast MRI cohort, but the UI is dataset-agnostic. It has two independent halves:

- **`analysis/`** — offline Python scripts that read raw radiomics + clinical data and emit aggregate JSON/CSV into `app/data/`.
  - `run_analysis.py` → `analysis.json` + `feature_rankings_{survival,subtype,pcr}.csv` (main dashboard).
  - `run_feature_explorer.py` → `feature-explorer.json` (Feature Statistics page). It reads the ranking CSVs, so run it *after* `run_analysis.py`.
- **`app/`** — a Next.js App Router–shaped client-side dashboard that only renders that JSON. Two routes: `/` (`dashboard.tsx`) and `/feature-explorer`.

The dashboard must never display patient identifiers or patient-level predictions — only aggregates. This is enforced by convention (the Python scripts only ever emit aggregates), not by code.

## Commands

```bash
npm install        # local install (plain npm; scripts/install-ci.sh needs Linux flock//proc)
npm run dev        # dev server (Vite, not `next dev`)
npm run build      # vinext + Cloudflare build, then scripts/validate-artifact.sh
npm run start      # serve the built worker (vinext start, port 3000)
npm run test       # build, then node --test tests/rendered-html.test.mjs against dist/server/index.js
npm run lint       # eslint
```

`build`/`lint`/`db:generate` go through `scripts/sites-env.sh`, which sandboxes `HOME`/npm cache/wrangler state under `.sites-runtime/`. `scripts/build-verified.sh` uses GNU `timeout`, which isn't on macOS. Locally, call the tools directly through the wrapper:

```bash
bash scripts/sites-env.sh -- node_modules/.bin/tsc --noEmit
bash scripts/sites-env.sh -- node_modules/.bin/eslint app/
bash scripts/sites-env.sh -- node_modules/.bin/vinext build
node --test tests/rendered-html.test.mjs   # only test; needs a fresh build first
```

Re-run the analysis (needs numpy, pandas, scipy, scikit-learn, lifelines). The raw inputs live in `data/raw/`:

```bash
python analysis/run_analysis.py --radiomics data/raw/merged_patient_Data_filtered.csv \
  --clinical data/raw/duke_brca_clin.xlsx --output-dir app/data
python analysis/run_feature_explorer.py --radiomics data/raw/merged_patient_Data_filtered.csv \
  --clinical data/raw/duke_brca_clin.xlsx --output-dir app/data
cp app/data/feature_rankings_*.csv public/data/   # powers the "download full ranking" links
```

`run_analysis.py` takes many minutes; most of that time is the survival endpoint's nested Cox CV.

## Architecture

**This is not a plain Next.js app.** Nothing runs through the `next` CLI. `vite.config.ts` drives everything through the `vinext` plugin, which implements the App Router runtime on Vite plus `@cloudflare/vite-plugin`. `worker/index.ts` is the real Cloudflare Worker entry: it handles `/_vinext/image` and otherwise delegates to `vinext/server/app-router-entry`. Build or runtime errors can come from `vinext` itself, and Next.js APIs only work as far as `vinext` implements them. `build/sites-vite-plugin.ts` copies `.openai/hosting.json` and `drizzle/` into `dist/.openai/` for the hosting platform. D1/Drizzle is wired up (`db/`, `examples/d1/`), but `db/schema.ts` is intentionally empty and unused. `app/chatgpt-auth.ts` (Sites "Sign in with ChatGPT" headers) exists but no page calls it.

**Data flow.** `app/dataset-context.tsx` (`DatasetProvider`/`useDataset`) holds the active `analysisData` and `featureExplorerData`. It seeds them from the static imports of `app/data/*.json` (the sample dataset). After mount it swaps in a dataset the user uploaded through `DatasetSwitcher` (in `app/ui.tsx`, a native `<dialog>`), which is persisted in `localStorage` under `radiomics-explorer:*` keys. Each page wraps its `*Inner` component in its own `DatasetProvider`. Pages share state only through `localStorage`, not through a shared React tree. The context exposes a `version` counter that increments on every upload or reset. Components that seed `useState` from the data are keyed on `version`, so they remount with the new data. Keep that pattern when adding stateful children. The `AnalysisData`/`FeatureExplorerData` types are `typeof` the sample JSON. `feature-explorer.tsx` instead declares its own richer interfaces and casts through `unknown`.

**Fixed three-endpoint schema.** Both JSON files always have `endpoints.survival`, `endpoints.subtype`, and `endpoints.pcr`, in that order (survival is the default tab). Chart choice branches on the endpoint key:
- survival: Kaplan–Meier chart, risk stratification, concordance index.
- subtype: multiclass one-vs-rest ROC.
- pcr: binary ROC and calibration curve; the clinical-context section is shown only for pcr.

The survival endpoint uses lifelines Cox regression with `l1_ratio=1.0`. Classification uses L1 logistic regression. When writing generic copy, say "L1-regularized model", not "logistic regression". `lifelines.concordance_index` expects higher-is-longer-survival scores, so risk scores are negated (`survival_concordance()`).

**Keep display text data-driven.** The upload feature means any number or verdict may come from a different cohort. The dashboard's interpretation headline and tone come from whether the primary metric's 95% CI excludes 0.5. Feature counts come from `meta.radiomicFeatureCount`, and labels come from endpoint `title`/`shortTitle`/`classes`. Guardrail strings are built in Python from computed counts. Don't hardcode dataset-specific numbers, "Duke", or "pCR"/"subtype" wording in the UI.

**Feature naming convention.** Radiomics columns are named `{transform}_{family}_{measure}`, with an optional `_vessel` suffix; no suffix means the tumor region. The Duke CSV has 2 regions × 12 transforms × 86 measures = 2,064 features, and no shape family. `parse_feature()` in `run_analysis.py` is the single source of truth for the `region`/`transform`/`family`/`measure`/`display` fields used by the filters and tables. Keep it in sync if the naming changes. The clinical workbook's headers are on row 5 (`read_duke_sheet()`). Every sheet joins on `Patient ID` with `validate="one_to_one"`. The CSV columns `menopause`/`er`/`pr`/`her2` are dropped from the predictor matrix.

**Leakage rule.** Univariate screening (ANOVA F, or concordance for survival), `CorrelationPruner` (|Spearman ρ| ≥ 0.90), and tuning all run inside training folds of nested CV: 5-fold × 3 repeats outer, 3-fold inner. Full-cohort FDR results are descriptive only. Don't move any selection step outside the CV loop.

## Other

- Every chart or table has an `InfoTip` ("i" popover, in `app/ui.tsx`) that explains what it shows and how to read it. Add one for new charts.
- Global CSS sets `scroll-behavior: smooth`. Headless-browser scripts should pass `behavior: 'instant'` to `scrollIntoView` before measuring or screenshotting.
- `docs/*.pptx` are presentation decks built from scratch scripts. They aren't generated by the repo.
