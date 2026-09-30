"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { DatasetProvider, useDataset, type AnalysisData } from "./dataset-context";
import { curveColors, DatasetSwitcher, formatQ, InfoTip } from "./ui";

type EndpointKey = keyof AnalysisData["endpoints"];
type Endpoint = AnalysisData["endpoints"][EndpointKey];
type SurvivalEndpoint = AnalysisData["endpoints"]["survival"];
type ClassificationEndpoint = AnalysisData["endpoints"]["subtype"] | AnalysisData["endpoints"]["pcr"];
type Feature = Endpoint["features"][number];

const metricHelp: Record<string, string> = {
  concordance: "Concordance index (C-index): the probability that, for a random pair of patients, the one predicted higher risk actually had the shorter survival time. 0.5 = chance-level ranking, 1.0 = perfect ranking.",
  logrankP: "P-value from a log-rank test comparing survival between the model's predicted higher- and lower-risk groups (split at the median out-of-fold risk score). Lower means a more significant separation between the two groups.",
  rocAuc: "Area under the ROC curve for out-of-fold predictions. 0.5 = chance-level separation, 1.0 = perfect separation.",
  macroRocAuc: "ROC AUC averaged one-vs-rest across all classes. 0.5 = chance-level separation, 1.0 = perfect separation.",
  prAuc: "Area under the precision-recall curve. More informative than ROC AUC when the positive class is rare.",
  balancedAccuracy: "Average of each class's recall, so the largest class can't inflate the score by itself.",
  macroF1: "F1 (precision/recall balance) averaged equally across all classes.",
  logLoss: "Penalty for confident but wrong probability estimates. Lower is better; 0 is a perfect model.",
  sensitivity: "Share of true positive-class cases the model correctly flagged (recall for the positive class).",
  specificity: "Share of true negative-class cases the model correctly flagged (recall for the negative class).",
  brier: "Mean squared error between predicted probability and the actual 0/1 outcome. Lower is better.",
};

function MetricCard({ metric, primary = false }: { metric: Endpoint["metrics"][number]; primary?: boolean }) {
  return (
    <article className={`metric-card ${primary ? "metric-primary" : ""}`}>
      <div className="metric-label-row">
        <span title={metricHelp[metric.key]}>{metric.label}</span>
        <span className="metric-direction" aria-label={metric.higherBetter ? "Higher is better" : "Lower is better"}>
          {metric.higherBetter ? "↑" : "↓"}
        </span>
      </div>
      <strong>{metric.value.toFixed(3)}</strong>
      <p>95% bootstrap interval {metric.low.toFixed(3)}–{metric.high.toFixed(3)}</p>
    </article>
  );
}

function Distribution({ endpoint }: { endpoint: Endpoint }) {
  const maximum = Math.max(...endpoint.classes.map((item) => item.count));
  return (
    <div className="distribution" role="img" aria-label="Endpoint class distribution">
      {endpoint.classes.map((item, index) => (
        <div className="distribution-row" key={item.label}>
          <div className="distribution-label">
            <span className="legend-dot" style={{ background: curveColors[index % curveColors.length] }} />
            <span>{item.label}</span>
          </div>
          <div className="bar-track">
            <span
              className="bar-fill"
              style={{
                width: `${Math.max(3, (item.count / maximum) * 100)}%`,
                background: curveColors[index % curveColors.length],
              }}
            />
          </div>
          <strong>{item.count}</strong>
          <small>{item.percent}%</small>
        </div>
      ))}
    </div>
  );
}

function classColor(endpoint: Endpoint, label: string) {
  const index = endpoint.classes.findIndex((item) => item.label === label);
  return curveColors[(index === -1 ? 0 : index) % curveColors.length];
}

function RocChart({ endpoint }: { endpoint: ClassificationEndpoint }) {
  const x0 = 45;
  const y0 = 188;
  const width = 250;
  const height = 162;
  return (
    <div className="roc-layout">
      <svg className="roc-chart" viewBox="0 0 330 220" role="img" aria-label="Out-of-fold receiver operating characteristic curves">
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={x0} x2={x0 + width} y1={y0 - tick * height} y2={y0 - tick * height} className="chart-grid" />
            <line x1={x0 + tick * width} x2={x0 + tick * width} y1={y0} y2={y0 - height} className="chart-grid" />
            <text x={x0 - 9} y={y0 - tick * height + 4} textAnchor="end" className="chart-tick">{tick.toFixed(2)}</text>
            <text x={x0 + tick * width} y={y0 + 17} textAnchor="middle" className="chart-tick">{tick.toFixed(2)}</text>
          </g>
        ))}
        <line x1={x0} y1={y0} x2={x0 + width} y2={y0 - height} className="chance-line" />
        {endpoint.rocCurves.map((curve) => (
          <polyline
            key={curve.label}
            points={curve.points.map((point) => `${x0 + point.x * width},${y0 - point.y * height}`).join(" ")}
            fill="none"
            stroke={classColor(endpoint, curve.label)}
            strokeWidth="3"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}
        <text x={x0 + width / 2} y="217" textAnchor="middle" className="chart-axis">False-positive rate</text>
        <text x="13" y={y0 - height / 2} textAnchor="middle" transform={`rotate(-90 13 ${y0 - height / 2})`} className="chart-axis">True-positive rate</text>
      </svg>
      <div className="chart-legend">
        {endpoint.rocCurves.map((curve) => (
          <div key={curve.label}>
            <span className="legend-line" style={{ background: classColor(endpoint, curve.label) }} />
            <span>{curve.label}</span>
            <strong>{curve.auc.toFixed(3)}</strong>
          </div>
        ))}
        <p>Repeated out-of-fold predictions; the dashed diagonal is chance.</p>
      </div>
    </div>
  );
}

function CalibrationChart({ endpoint }: { endpoint: ClassificationEndpoint }) {
  if (!endpoint.calibration.length) return null;
  const x0 = 42;
  const y0 = 170;
  const width = 238;
  const height = 142;
  return (
    <svg className="calibration-chart" viewBox="0 0 315 205" role="img" aria-label={`${endpoint.classes[1]?.label ?? "Positive class"} calibration curve`}>
      {[0, 0.5, 1].map((tick) => (
        <g key={tick}>
          <line x1={x0} x2={x0 + width} y1={y0 - tick * height} y2={y0 - tick * height} className="chart-grid" />
          <line x1={x0 + tick * width} x2={x0 + tick * width} y1={y0} y2={y0 - height} className="chart-grid" />
          <text x={x0 - 8} y={y0 - tick * height + 4} textAnchor="end" className="chart-tick">{tick.toFixed(1)}</text>
          <text x={x0 + tick * width} y={y0 + 17} textAnchor="middle" className="chart-tick">{tick.toFixed(1)}</text>
        </g>
      ))}
      <line x1={x0} y1={y0} x2={x0 + width} y2={y0 - height} className="chance-line" />
      <polyline
        points={endpoint.calibration.map((point) => `${x0 + point.x * width},${y0 - point.y * height}`).join(" ")}
        fill="none"
        stroke="#d6a54c"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {endpoint.calibration.map((point, index) => (
        <circle key={index} cx={x0 + point.x * width} cy={y0 - point.y * height} r="4" fill="#d6a54c" />
      ))}
      <text x={x0 + width / 2} y="201" textAnchor="middle" className="chart-axis">Mean predicted probability</text>
    </svg>
  );
}

function stepPoints(points: { x: number; y: number }[]): { x: number; y: number }[] {
  const stepped: { x: number; y: number }[] = [];
  points.forEach((point, index) => {
    if (index > 0) stepped.push({ x: point.x, y: points[index - 1].y });
    stepped.push(point);
  });
  return stepped;
}

function KaplanMeierChart({ endpoint }: { endpoint: SurvivalEndpoint }) {
  const x0 = 45;
  const y0 = 188;
  const width = 250;
  const height = 162;
  const maxMonths = endpoint.survivalCurve.maxMonths;
  const toXY = (point: { x: number; y: number }) => `${x0 + (point.x / maxMonths) * width},${y0 - point.y * height}`;
  return (
    <div className="roc-layout">
      <svg className="roc-chart" viewBox="0 0 330 220" role="img" aria-label="Kaplan-Meier overall survival curve">
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={x0} x2={x0 + width} y1={y0 - tick * height} y2={y0 - tick * height} className="chart-grid" />
            <line x1={x0 + tick * width} x2={x0 + tick * width} y1={y0} y2={y0 - height} className="chart-grid" />
            <text x={x0 - 9} y={y0 - tick * height + 4} textAnchor="end" className="chart-tick">{tick.toFixed(2)}</text>
            <text x={x0 + tick * width} y={y0 + 17} textAnchor="middle" className="chart-tick">{Math.round(tick * maxMonths)}</text>
          </g>
        ))}
        <polyline
          points={stepPoints(endpoint.survivalCurve.points).map(toXY).join(" ")}
          fill="none"
          stroke={curveColors[0]}
          strokeWidth="3"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        <text x={x0 + width / 2} y="217" textAnchor="middle" className="chart-axis">Months since diagnosis</text>
        <text x="13" y={y0 - height / 2} textAnchor="middle" transform={`rotate(-90 13 ${y0 - height / 2})`} className="chart-axis">Survival probability</text>
      </svg>
      <div className="chart-legend">
        <div>
          <span className="legend-line" style={{ background: curveColors[0] }} />
          <span>Whole cohort</span>
          <strong>n={endpoint.sampleCount}</strong>
        </div>
        <p>Kaplan–Meier estimate; each step is an observed death.</p>
      </div>
    </div>
  );
}

function RiskStratification({ endpoint }: { endpoint: SurvivalEndpoint }) {
  const x0 = 45;
  const y0 = 188;
  const width = 250;
  const height = 162;
  const maxMonths = endpoint.survivalCurve.maxMonths;
  const toXY = (point: { x: number; y: number }) => `${x0 + (point.x / maxMonths) * width},${y0 - point.y * height}`;
  return (
    <div className="roc-layout">
      <svg className="roc-chart" viewBox="0 0 330 220" role="img" aria-label="Kaplan-Meier curves by predicted risk group">
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={x0} x2={x0 + width} y1={y0 - tick * height} y2={y0 - tick * height} className="chart-grid" />
            <line x1={x0 + tick * width} x2={x0 + tick * width} y1={y0} y2={y0 - height} className="chart-grid" />
            <text x={x0 - 9} y={y0 - tick * height + 4} textAnchor="end" className="chart-tick">{tick.toFixed(2)}</text>
            <text x={x0 + tick * width} y={y0 + 17} textAnchor="middle" className="chart-tick">{Math.round(tick * maxMonths)}</text>
          </g>
        ))}
        {endpoint.riskGroups.groups.map((group, index) => (
          <polyline
            key={group.label}
            points={stepPoints(group.points).map(toXY).join(" ")}
            fill="none"
            stroke={curveColors[index % curveColors.length]}
            strokeWidth="3"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}
        <text x={x0 + width / 2} y="217" textAnchor="middle" className="chart-axis">Months since diagnosis</text>
        <text x="13" y={y0 - height / 2} textAnchor="middle" transform={`rotate(-90 13 ${y0 - height / 2})`} className="chart-axis">Survival probability</text>
      </svg>
      <div className="chart-legend">
        {endpoint.riskGroups.groups.map((group, index) => (
          <div key={group.label}>
            <span className="legend-line" style={{ background: curveColors[index % curveColors.length] }} />
            <span>{group.label}</span>
            <strong>{group.events}/{group.n}</strong>
          </div>
        ))}
        <p>Split at the median out-of-fold risk score; log-rank p = {formatQ(endpoint.riskGroups.logrankP)}.</p>
      </div>
    </div>
  );
}

function FeatureFunnel({ endpoint }: { endpoint: Endpoint }) {
  const maximum = endpoint.featureFunnel[0].count;
  const maxLog = Math.log10(maximum + 1);
  return (
    <div className="funnel-list">
      {endpoint.featureFunnel.map((stage, index) => {
        const width = stage.count === 0 ? 0 : Math.max(3, (Math.log10(stage.count + 1) / maxLog) * 100);
        return (
          <div className="funnel-row" key={stage.label}>
            <div className="funnel-index">{String(index + 1).padStart(2, "0")}</div>
            <div className="funnel-content">
              <div className="funnel-heading">
                <strong>{stage.label}</strong>
                <span>{stage.count.toLocaleString()}</span>
              </div>
              <div className="funnel-track"><span style={{ width: `${width}%` }} /></div>
              <p>{stage.note}</p>
            </div>
          </div>
        );
      })}
      <p className="micro-note">Bar widths use a log scale so the last selection stages remain visible.</p>
    </div>
  );
}

function FeatureLandscape({ endpoint }: { endpoint: Endpoint }) {
  const groups = [
    { title: "Region", data: endpoint.featureGroups.regions },
    { title: "Feature family", data: endpoint.featureGroups.families },
    { title: "Transform", data: endpoint.featureGroups.transforms.slice(0, 6) },
  ];
  return (
    <div className="landscape-grid">
      {groups.map((group) => {
        const maximum = Math.max(...group.data.map((item) => item.count), 1);
        return (
          <div className="landscape-group" key={group.title}>
            <h4>{group.title}</h4>
            {group.data.map((item) => (
              <div className="mini-bar-row" key={item.label}>
                <span title={item.label}>{item.label}</span>
                <div><i style={{ width: `${(item.count / maximum) * 100}%` }} /></div>
                <strong>{item.count}</strong>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}

function FeatureTable({ endpoint, endpointKey }: { endpoint: Endpoint; endpointKey: EndpointKey }) {
  const [region, setRegion] = useState("All");
  const [family, setFamily] = useState("All");
  const [query, setQuery] = useState("");
  const [showAll, setShowAll] = useState(false);
  const [selected, setSelected] = useState<Feature | null>(endpoint.features[0] ?? null);

  const regions = ["All", ...Array.from(new Set(endpoint.features.map((feature) => feature.region)))];
  const families = ["All", ...Array.from(new Set(endpoint.features.map((feature) => feature.family)))];
  const filtered = useMemo(
    () => endpoint.features.filter((feature) => {
      const matchesRegion = region === "All" || feature.region === region;
      const matchesFamily = family === "All" || feature.family === family;
      const haystack = `${feature.display} ${feature.feature}`.toLowerCase();
      return matchesRegion && matchesFamily && haystack.includes(query.toLowerCase());
    }),
    [endpoint.features, region, family, query],
  );

  return (
    <>
      <div className="feature-controls">
        <label>
          <span>Region</span>
          <select value={region} onChange={(event) => setRegion(event.target.value)}>
            {regions.map((item) => <option key={item}>{item}</option>)}
          </select>
        </label>
        <label>
          <span>Family</span>
          <select value={family} onChange={(event) => setFamily(event.target.value)}>
            {families.map((item) => <option key={item}>{item}</option>)}
          </select>
        </label>
        <label className="search-field">
          <span>Search top 50</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="e.g. zone entropy" />
        </label>
        <a className="download-link" href={`/data/feature_rankings_${endpointKey}.csv`} download>
          Download full ranking <span aria-hidden="true">↓</span>
        </a>
      </div>

      {selected && (
        <div className="feature-focus">
          <div>
            <p>Selected feature</p>
            <h4>{selected.display}</h4>
            <code>{selected.feature}</code>
          </div>
          <dl>
            <div><dt title="Share of the 15 outer cross-validation folds in which the L1-regularized model kept this feature.">Fold stability</dt><dd>{Math.round(selected.frequency * 100)}%</dd></div>
            <div><dt title="Benjamini–Hochberg false-discovery-rate-adjusted p-value from a full-cohort univariate test; q < 0.05 is conventionally significant.">FDR q</dt><dd>{formatQ(selected.qValue)}</dd></div>
            <div><dt title="Standardized effect size from the full cohort (descriptive only, not from the validated model).">Effect</dt><dd>{selected.effect.toFixed(3)}</dd></div>
            <div><dt title="Which class this feature's value trends higher for, in the full-cohort data.">Direction</dt><dd>{selected.direction}</dd></div>
          </dl>
        </div>
      )}

      <div className="table-shell">
        <table className="feature-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Feature</th>
              <th>Region</th>
              <th title="Share of the 15 outer cross-validation folds in which the L1-regularized model kept this feature.">Stability</th>
              <th title="Standardized effect size from the full cohort (descriptive only, not from the validated model).">Effect</th>
              <th title="Benjamini–Hochberg false-discovery-rate-adjusted p-value from a full-cohort univariate test; q < 0.05 is conventionally significant.">FDR q</th>
              <th title="Stable = selected in ≥50% of outer folds. FDR = passes q < 0.05. Exploratory = neither.">Evidence</th>
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, showAll ? filtered.length : 20).map((feature) => (
              <tr key={feature.feature} className={selected?.feature === feature.feature ? "selected-row" : ""}>
                <td>{feature.rank}</td>
                <td>
                  <button className="feature-name" onClick={() => setSelected(feature)}>
                    <strong>{feature.measure}</strong>
                    <span>{feature.transform} · {feature.family}</span>
                  </button>
                </td>
                <td><span className={`region-pill ${feature.region.toLowerCase()}`}>{feature.region}</span></td>
                <td>
                  <div className="stability-cell">
                    <span>{Math.round(feature.frequency * 100)}%</span>
                    <i><b style={{ width: `${feature.frequency * 100}%` }} /></i>
                  </div>
                </td>
                <td>{feature.effect.toFixed(3)}</td>
                <td>{formatQ(feature.qValue)}</td>
                <td>
                  {feature.frequency >= 0.5 && <span className="evidence-tag stable" title="Selected in at least half of the outer cross-validation folds.">Stable</span>}
                  {feature.fdrSignificant && <span className="evidence-tag fdr" title="Passes Benjamini–Hochberg FDR q < 0.05 in the full-cohort univariate test.">FDR</span>}
                  {!feature.fdrSignificant && feature.frequency < 0.5 && <span className="evidence-tag weak" title="Neither cross-validated stability nor full-cohort FDR significance reached the threshold.">Exploratory</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!filtered.length && <p className="empty-state">No features match these filters.</p>}
      </div>
      {filtered.length > 20 && (
        <button className="table-toggle" onClick={() => setShowAll((value) => !value)}>
          {showAll ? "Show top 20" : `Show all ${filtered.length} displayed ranks`}
        </button>
      )}
    </>
  );
}

function ConfusionMatrix({ endpoint }: { endpoint: ClassificationEndpoint }) {
  return (
    <div className="matrix-wrap">
      <div className="matrix-y-label">Actual class</div>
      <div
        className="matrix-grid"
        style={{ gridTemplateColumns: `minmax(88px, 1fr) repeat(${endpoint.confusion.labels.length}, minmax(54px, 1fr))` }}
      >
        <div />
        {endpoint.confusion.labels.map((label) => <div className="matrix-label top" key={`top-${label}`}>{label}</div>)}
        {endpoint.confusion.normalized.flatMap((row, rowIndex) => [
          <div className="matrix-label" key={`label-${rowIndex}`}>{endpoint.confusion.labels[rowIndex]}</div>,
          ...row.map((value, columnIndex) => (
            <div
              key={`${rowIndex}-${columnIndex}`}
              className={`matrix-cell ${rowIndex === columnIndex ? "diagonal" : ""}`}
              style={{ "--strength": Math.max(0.08, value) } as React.CSSProperties}
            >
              <strong>{Math.round(value * 100)}%</strong>
              <span>n={endpoint.confusion.counts[rowIndex][columnIndex]}</span>
            </div>
          )),
        ])}
      </div>
      <div className="matrix-x-label">Predicted class</div>
    </div>
  );
}

function ClinicalContext() {
  const { analysisData } = useDataset();
  const grouped = analysisData.clinicalContext.reduce<Record<string, typeof analysisData.clinicalContext>>((acc, row) => {
    (acc[row.variable] ??= []).push(row);
    return acc;
  }, {});
  return (
    <div className="context-grid">
      {Object.entries(grouped).map(([variable, rows]) => (
        <article className="context-card" key={variable}>
          <h4>{variable}</h4>
          {rows.map((row) => (
            <div className="context-row" key={row.group}>
              <span>{row.group}</span>
              <div><i style={{ width: `${Math.max(3, row.rate * 100)}%` }} /></div>
              <strong>{Math.round(row.rate * 100)}%</strong>
              <small>{row.pcr}/{row.n}</small>
            </div>
          ))}
        </article>
      ))}
    </div>
  );
}

function DashboardInner() {
  const { analysisData, version } = useDataset();
  const [activeKey, setActiveKey] = useState<EndpointKey>("survival");
  const endpoint = analysisData.endpoints[activeKey] as Endpoint;
  const primary = endpoint.metrics[0];
  const isSurvival = activeKey === "survival";
  const isSubtype = activeKey === "subtype";
  const isPcr = activeKey === "pcr";
  const stableFeatures = endpoint.featureFunnel.at(-1)?.count ?? 0;
  const pointMargin = primary.higherBetter ? primary.value - 0.5 : 0.5 - primary.value;
  const ciExcludesChance = primary.higherBetter ? primary.low > 0.5 : primary.high < 0.5;
  const strengthWord = pointMargin > 0.15 ? "a real, moderate" : pointMargin > 0.05 ? "a modest but detectable" : "a narrow, borderline";
  const interpretation = ciExcludesChance
    ? {
        title: `${endpoint.title} shows ${strengthWord} signal.`,
        body: `${endpoint.featureFunnel[2]?.count ?? 0} features pass univariate FDR, and ${stableFeatures} stay stable across outer folds. ${primary.label} ${primary.value.toFixed(3)} (95% CI ${primary.low.toFixed(3)}–${primary.high.toFixed(3)}) excludes the chance-level reference of 0.5.`,
        tone: pointMargin > 0.15 ? "positive" : "caution",
      }
    : {
        title: `No reproducible ${endpoint.title.toLowerCase()} signature emerged.`,
        body: `The 95% interval for ${primary.label} includes the chance-level reference of 0.5 (${primary.value.toFixed(3)}, CI ${primary.low.toFixed(3)}–${primary.high.toFixed(3)})${endpoint.validation.finalActiveFeatures === 0 ? "; the final full-cohort model selected zero active features" : ""}. Treat the displayed top features as exploratory, not a validated signature.`,
        tone: "negative",
      };

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Radiomics explorer home">
          <span className="brand-mark">RE</span>
          <span><strong>Radiomics Explorer</strong><small>Feature selection explorer</small></span>
        </a>
        <nav aria-label="Page sections">
          <a href="#performance">Performance</a>
          <a href="#features">Features</a>
          <a href="#methods">Methods</a>
          <Link href="/feature-explorer">Feature Statistics</Link>
          <DatasetSwitcher />
        </nav>
        <span className="status-pill"><i /> Analysis complete</span>
      </header>

      <section className="page-header" id="top">
        <div className="page-header-copy">
          <p className="eyebrow">Retrospective cohort · {analysisData.meta.generatedAt}</p>
          <h1>{analysisData.meta.title}</h1>
          <p className="page-header-lede">
            {analysisData.meta.subtitle} across {analysisData.meta.radiomicFeatureCount.toLocaleString()} tumor and vessel features. Clinical data supplies labels and context—never predictors.
          </p>
          <div className="endpoint-switch" role="group" aria-label="Choose prediction endpoint">
            {(Object.keys(analysisData.endpoints) as EndpointKey[]).map((key) => (
              <button key={key} aria-pressed={activeKey === key} onClick={() => setActiveKey(key)}>
                <span>{analysisData.endpoints[key].shortTitle}</span>
                <small>{analysisData.endpoints[key].sampleCount} evaluable patients</small>
              </button>
            ))}
          </div>
        </div>

        <aside className={`result-card ${interpretation.tone}`}>
          <p className="result-card-label">Primary result</p>
          <h2>{interpretation.title}</h2>
          <div className="result-card-score">
            <strong>{primary.value.toFixed(3)}</strong>
            <span>{primary.label}<small>95% CI {primary.low.toFixed(3)}–{primary.high.toFixed(3)}</small></span>
          </div>
          <p className="result-card-body">{interpretation.body}</p>
        </aside>
      </section>

      <section className="cohort-strip" aria-label="Cohort summary">
        <div><strong>{analysisData.meta.patientCount}</strong><span>unique patients</span></div>
        <div><strong>{analysisData.meta.tumorFeatureCount.toLocaleString()}</strong><span>tumor features</span></div>
        <div><strong>{analysisData.meta.vesselFeatureCount.toLocaleString()}</strong><span>vessel features</span></div>
        <div><strong>{analysisData.meta.clinicalPredictorsUsed}</strong><span>clinical predictors</span></div>
        <div><strong>{endpoint.validation.outerFits}</strong><span>outer validation fits</span></div>
      </section>

      <section className="section-shell" id="performance">
        <div className="section-heading">
          <div><p className="eyebrow">01 · Model evidence</p><h2>{endpoint.title}</h2></div>
          <p>{endpoint.description}</p>
        </div>

        <div className="metric-grid">
          {endpoint.metrics.map((metric, index) => <MetricCard key={metric.key} metric={metric} primary={index === 0} />)}
        </div>

        <div className="analysis-grid">
          <article className="panel panel-wide">
            {isSurvival ? (
              <>
                <div className="panel-heading">
                  <div><p>Discrimination</p><h3>Overall survival</h3></div>
                  <div className="panel-heading-meta">
                    <span title="Out-of-fold: each patient's risk score comes only from a model that never saw that patient during training.">OOF risk scores</span>
                    <InfoTip label="Overall survival">
                      <p><strong>What this shows:</strong> the Kaplan–Meier estimate of overall survival for the whole evaluable cohort — the raw survival experience, not a model prediction.</p>
                      <p><strong>How to read it:</strong> the curve steps down at each observed death; a censored patient (alive at last contact) does not cause a step. It stays flat between events.</p>
                    </InfoTip>
                  </div>
                </div>
                <KaplanMeierChart endpoint={endpoint as SurvivalEndpoint} />
              </>
            ) : (
              <>
                <div className="panel-heading">
                  <div><p>Discrimination</p><h3>Receiver operating characteristic</h3></div>
                  <div className="panel-heading-meta">
                    <span title="Out-of-fold: each patient's prediction comes only from a model that never saw that patient during training.">OOF predictions</span>
                    <InfoTip label="Receiver operating characteristic">
                      <p><strong>What this shows:</strong> each curve traces the true-positive rate against the false-positive rate as the decision threshold moves, from pooled out-of-fold predictions. {isSubtype ? "One curve per subtype, each scored one-vs-rest." : "A single curve for the pCR class."}</p>
                      <p><strong>How to read it:</strong> the dashed diagonal is random guessing (AUC 0.5). The closer a curve bows toward the top-left corner, the better it separates that class from the rest.</p>
                    </InfoTip>
                  </div>
                </div>
                <RocChart endpoint={endpoint as ClassificationEndpoint} />
              </>
            )}
          </article>
          <article className="panel">
            <div className="panel-heading">
              <div><p>Cohort balance</p><h3>{isSurvival ? "Event distribution" : "Class distribution"}</h3></div>
              <div className="panel-heading-meta">
                <span>n={endpoint.sampleCount}</span>
                <InfoTip label="Class distribution">
                  <p><strong>What this shows:</strong> the number and share of the {endpoint.sampleCount} evaluable patients in each {isSurvival ? "event-status" : isSubtype ? "subtype" : "outcome"} class.</p>
                  <p><strong>How to read it:</strong> this is the raw label distribution, not a model output. It&apos;s useful for judging whether a high {isSurvival ? "concordance" : "accuracy"} score could be reached just by guessing the majority class.</p>
                </InfoTip>
              </div>
            </div>
            <Distribution endpoint={endpoint} />
            <div className="panel-callout">{endpoint.prevalenceNote}</div>
          </article>
          {isPcr && (
            <article className="panel">
              <div className="panel-heading">
                <div><p>Reliability</p><h3>Calibration</h3></div>
                <div className="panel-heading-meta">
                  <span>8 quantile bins</span>
                  <InfoTip label="Calibration">
                    <p><strong>What this shows:</strong> patients are grouped into 8 equal-sized bins by predicted pCR probability. Each point plots a bin&apos;s mean predicted probability (x) against its actual observed pCR rate (y).</p>
                    <p><strong>How to read it:</strong> a well-calibrated model&apos;s points sit on the diagonal. Points above the line mean the model under-predicted risk in that range; points below mean it over-predicted.</p>
                  </InfoTip>
                </div>
              </div>
              <CalibrationChart endpoint={endpoint as ClassificationEndpoint} />
              <div className="panel-callout">A calibrated model follows the diagonal. Here, uncertainty is substantial.</div>
            </article>
          )}
        </div>
      </section>

      <section className="section-shell tone-section" id="features">
        <div className="section-heading">
          <div><p className="eyebrow">02 · Selection evidence</p><h2>From {analysisData.meta.radiomicFeatureCount.toLocaleString()} measurements to a defensible shortlist</h2></div>
          <p>Every screening and tuning decision is learned inside training folds. Full-cohort FDR results are descriptive, not a shortcut into validation.</p>
        </div>
        <div className="selection-grid">
          <article className="panel">
            <div className="panel-heading">
              <div><p>Selection funnel</p><h3>Evidence attrition</h3></div>
              <div className="panel-heading-meta">
                <span>log scale</span>
                <InfoTip label="Evidence attrition">
                  <p><strong>What this shows:</strong> how many of the {analysisData.meta.radiomicFeatureCount.toLocaleString()} radiomic features survive each stage of the pipeline, from raw input down to features selected in at least half of the outer cross-validation folds.</p>
                  <p><strong>How to read it:</strong> bars use a log scale, so equal-looking widths do not mean equal counts — this keeps the small late-stage counts visible. Each row&apos;s caption states the filter applied at that stage.</p>
                </InfoTip>
              </div>
            </div>
            <FeatureFunnel endpoint={endpoint} />
          </article>
          <article className="panel panel-wide">
            <div className="panel-heading">
              <div><p>Top 20 stable ranks</p><h3>Feature landscape</h3></div>
              <div className="panel-heading-meta">
                <span>counts</span>
                <InfoTip label="Feature landscape">
                  <p><strong>What this shows:</strong> among the top 20 stability-ranked features for this endpoint, how they split by tumor/vessel region, radiomic feature family (e.g. GLCM texture, first order), and the image transform applied before the feature was computed (e.g. wavelet, Laplacian-of-Gaussian).</p>
                  <p><strong>How to read it:</strong> bars are counts out of 20. A family or transform that dominates suggests the signal concentrates in a particular kind of image texture rather than spreading evenly.</p>
                </InfoTip>
              </div>
            </div>
            <FeatureLandscape endpoint={endpoint} />
            <div className={`signal-note ${ciExcludesChance ? "" : "negative"}`}>
              <strong>
                {ciExcludesChance
                  ? `${stableFeatures} features remain stable across outer folds, with ${primary.label} clearly above chance.`
                  : `Do not treat the displayed ${endpoint.title.toLowerCase()} ranks as a validated signature.`}
              </strong>
              <span>
                {ciExcludesChance
                  ? "No external cohort has confirmed these associations — treat this as hypothesis-generating, and check the event/sample count before over-interpreting."
                  : endpoint.validation.finalActiveFeatures === 0
                  ? "The final full-cohort sparse model selected zero active features, and none passed FDR."
                  : "Even these require external validation and robustness testing."}
              </span>
            </div>
          </article>
        </div>

        <article className="panel feature-panel">
          <div className="panel-heading">
            <div><p>Feature registry</p><h3>Stability-ranked radiomic features</h3></div>
            <div className="panel-heading-meta">
              <span>click a feature for detail</span>
              <InfoTip label="Feature registry">
                <p><strong>What this shows:</strong> the top 50 radiomic features, ranked by how often the L1-regularized model kept them across the 15 outer cross-validation folds, then by coefficient magnitude and univariate effect size.</p>
                <p><strong>How to read it:</strong> &quot;Stable&quot; (selected in ≥50% of folds) is the strongest evidence here. &quot;FDR&quot; passes a full-cohort significance test, which is easier to satisfy and does not by itself validate a predictor. &quot;Exploratory&quot; features pass neither. Click any row for full detail.</p>
              </InfoTip>
            </div>
          </div>
          <FeatureTable key={`${activeKey}-${version}`} endpoint={endpoint} endpointKey={activeKey} />
        </article>
      </section>

      <section className="section-shell">
        <div className="section-heading">
          <div>
            <p className="eyebrow">03 · Model validation</p>
            <h2>{isSurvival ? "Does predicted risk actually separate survival?" : "Where the model succeeds—and fails"}</h2>
          </div>
          <p>
            {isSurvival
              ? "Patients are split at the median out-of-fold risk score into two equal-sized groups; the log-rank test compares their observed survival directly."
              : "Rows sum to 100%. Balanced accuracy gives each actual class equal weight despite the cohort imbalance."}
          </p>
        </div>
        <article className={isSurvival ? "panel panel-wide" : "panel matrix-panel"}>
          {isSurvival ? (
            <>
              <div className="panel-heading">
                <div><p>Model validation</p><h3>Risk-stratified survival</h3></div>
                <div className="panel-heading-meta">
                  <InfoTip label="Risk-stratified survival">
                    <p><strong>What this shows:</strong> two Kaplan–Meier curves built from the same out-of-fold risk scores used for the concordance index above, split at their median into &quot;lower&quot; and &quot;higher&quot; predicted-risk groups.</p>
                    <p><strong>How to read it:</strong> a model with real discrimination separates the two curves visibly, with the higher-risk group dropping faster. The log-rank p-value tests whether that separation is statistically significant.</p>
                  </InfoTip>
                </div>
              </div>
              <RiskStratification endpoint={endpoint as SurvivalEndpoint} />
            </>
          ) : (
            <>
              <div className="panel-heading">
                <div><p>Error analysis</p><h3>Confusion matrix</h3></div>
                <div className="panel-heading-meta">
                  <InfoTip label="Confusion matrix">
                    <p><strong>What this shows:</strong> a row-normalized breakdown of out-of-fold predictions. Each row is one actual class; each cell is the percent (and count) of that class&apos;s patients predicted into each column.</p>
                    <p><strong>How to read it:</strong> rows sum to 100%. Diagonal cells are correct predictions — the darker and more concentrated the diagonal, the better the model. Off-diagonal weight shows where the model confuses classes.</p>
                  </InfoTip>
                </div>
              </div>
              <ConfusionMatrix endpoint={endpoint as ClassificationEndpoint} />
            </>
          )}
        </article>
      </section>

      {isPcr && (
        <section className="section-shell context-section">
          <div className="section-heading">
            <div>
              <p className="eyebrow">04 · Clinical context</p>
              <div className="section-heading-title">
                <h2>Observed {endpoint.classes[1]?.label ?? "outcome"} rates differ across clinical groups</h2>
                <InfoTip label="Clinical context">
                  <p><strong>What this shows:</strong> the observed {endpoint.classes[1]?.label ?? "outcome"} rate for each subgroup of the clinical variables in this cohort, computed directly from the clinical workbook.</p>
                  <p><strong>How to read it:</strong> these are descriptive statistics only — none of these variables were used to select features or train the radiomics model. Don&apos;t read them as the model&apos;s predictions.</p>
                </InfoTip>
              </div>
            </div>
            <p>These variables are descriptive only. They were not included in feature selection or prediction.</p>
          </div>
          <ClinicalContext />
        </section>
      )}

      <section className="section-shell method-section" id="methods">
        <div className="section-heading">
          <div><p className="eyebrow">{isPcr ? "05" : "04"} · Reproducibility</p><h2>A conventional pipeline, kept inside the validation loop</h2></div>
          <p>{endpoint.targetDefinition}</p>
        </div>
        <div className="method-grid">
          {endpoint.method.map((item) => (
            <article key={item.step}><span>{item.step}</span><h3>{item.title}</h3><p>{item.detail}</p></article>
          ))}
        </div>
        <div className="validation-spec">
          <div><span title="How held-out performance was estimated: the data was split, refit, and re-tested this many times.">Outer validation</span><strong>{endpoint.validation.outerDesign}</strong></div>
          <div><span title={`Within each outer training split, this cross-validation loop picked the screening size (k) and ${isSurvival ? "regularization penalizer" : "regularization strength (C)"}.`}>Inner tuning</span><strong>{endpoint.validation.innerDesign}</strong></div>
          <div><span title={`Median number of features the ${isSurvival ? "Cox" : "L1"} model kept nonzero, across all outer folds.`}>Median active features</span><strong>{endpoint.validation.medianActiveFeatures}</strong></div>
          <div><span title={`Settings for the single model refit on the full cohort: k = features kept by the univariate screen; ${isSurvival ? "penalizer = elastic-net regularization strength (smaller means a sparser model)." : "C = inverse L1 penalty strength (smaller C means a sparser model)."}`}>Final model</span><strong>k={endpoint.validation.finalK}, {isSurvival ? "penalizer" : "C"}={endpoint.validation.finalC}</strong></div>
        </div>
      </section>

      <section className="guardrail-section">
        <div className="guardrail-heading"><span>!</span><div><p className="eyebrow">Interpretation boundary</p><h2>What this analysis cannot establish</h2></div></div>
        <div className="guardrail-grid">
          {analysisData.guardrails.map((item, index) => <p key={item}><span>{String(index + 1).padStart(2, "0")}</span>{item}</p>)}
        </div>
      </section>

      <footer>
        <div><strong>Radiomics Explorer</strong><span>Aggregate research dashboard · no patient-level data displayed</span></div>
        <div><span>Sources</span><code>{analysisData.meta.radiomicsSource}</code><code>{analysisData.meta.clinicalSource}</code></div>
      </footer>
    </main>
  );
}

function Dashboard() {
  return (
    <DatasetProvider>
      <DashboardInner />
    </DatasetProvider>
  );
}

export default Dashboard;
