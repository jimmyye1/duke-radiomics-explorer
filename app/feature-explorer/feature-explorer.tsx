"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { DatasetProvider, useDataset } from "../dataset-context";
import { curveColors, DatasetSwitcher, formatQ, InfoTip } from "../ui";

type EndpointKey = "survival" | "subtype" | "pcr";

interface DistributionStats {
  n: number;
  min: number;
  p5: number;
  p25: number;
  median: number;
  p75: number;
  p95: number;
  max: number;
  mean: number;
  sd: number;
  iqr: number;
  skewness: number;
  extremeCount: number;
  extremePct: number;
}

interface HistogramData {
  binEdges?: number[];
  counts: number[];
}

interface PercentilePoint {
  percentile: number;
  value: number;
}

interface GroupStats {
  label: string;
  n: number;
  median: number;
  iqr: number;
  histogram: HistogramData;
}

interface BinaryDiscrimination {
  type: "binary";
  auc: number;
  ciLow: number;
  ciHigh: number;
  threshold: number;
  thresholdRule: string;
  sensitivity: number;
  specificity: number;
  ppv: number;
  npv: number;
}

interface OneVsRestDiscrimination {
  type: "oneVsRest";
  byClass: { label: string; auc: number; ciLow: number; ciHigh: number }[];
  summary: string;
}

interface ConcordanceDiscrimination {
  type: "concordance";
  concordance: number;
  ciLow: number;
  ciHigh: number;
}

interface FeatureEntry {
  feature: string;
  display: string;
  region: string;
  family: string;
  transform: string;
  measure: string;
  rank: number;
  stability: number;
  univariateP: number;
  fdrQ: number;
  effect: number;
  effectCiLow: number | null;
  effectCiHigh: number | null;
  direction: string;
  finalModel: boolean;
  distribution: DistributionStats;
  histogram: { binEdges: number[]; counts: number[] };
  percentileCurve: PercentilePoint[];
  groups: GroupStats[];
  discrimination: BinaryDiscrimination | OneVsRestDiscrimination | ConcordanceDiscrimination;
}

interface QualityRow {
  feature: string;
  display: string;
  region: string;
  family: string;
  transform: string;
  missingRate: number;
  outlierRate: number;
  nearConstant: boolean;
  quality: string;
}

interface QualityRollup {
  good: number;
  caution: number;
  poor: number;
}

interface FeatureExplorerData {
  meta: {
    patientCount: number;
    radiomicFeatureCount: number;
    tumorFeatureCount: number;
    vesselFeatureCount: number;
    endpoints: Record<EndpointKey, { sampleCount: number; excludedCount: number; exclusionReasons: { reason: string; count: number }[] }>;
    qualityThresholds: {
      outlierCautionRate: number;
      outlierPoorRate: number;
      nearConstantTopValueShare: number;
      missingCautionRate: number;
      missingPoorRate: number;
    };
    discriminationThreshold: string;
    bootstrap: number;
    unavailable: { topic: string; note: string }[];
  };
  dataQuality: {
    rollup: { overall: QualityRollup; byRegion: Record<string, QualityRollup>; byFamily: Record<string, QualityRollup> };
    features: QualityRow[];
  };
  familyGroups: { family: string; interpretation: string; available: boolean; note?: string }[];
  endpoints: Record<EndpointKey, { sampleCount: number; classes: string[]; features: FeatureEntry[] }>;
}

function useFeatureData(): FeatureExplorerData {
  const { featureExplorerData } = useDataset();
  return featureExplorerData as unknown as FeatureExplorerData;
}

function interpolatePercentile(curve: PercentilePoint[], value: number): number {
  if (value <= curve[0].value) return 0;
  if (value >= curve[curve.length - 1].value) return 100;
  for (let i = 0; i < curve.length - 1; i++) {
    const a = curve[i];
    const b = curve[i + 1];
    if (value >= a.value && value <= b.value) {
      if (b.value === a.value) return a.percentile;
      const t = (value - a.value) / (b.value - a.value);
      return a.percentile + t * (b.percentile - a.percentile);
    }
  }
  return 100;
}

function qualityClass(quality: string) {
  if (quality === "good") return "stable";
  if (quality === "caution") return "fdr";
  return "poor";
}

function QualityBadge({ quality }: { quality: string }) {
  return <span className={`evidence-tag ${qualityClass(quality)}`}>{quality}</span>;
}

function StatCard({ label, value, sublabel, primary = false, title }: { label: string; value: string; sublabel?: string; primary?: boolean; title?: string }) {
  return (
    <article className={`metric-card ${primary ? "metric-primary" : ""}`}>
      <div className="metric-label-row">
        <span title={title}>{label}</span>
      </div>
      <strong>{value}</strong>
      {sublabel && <p>{sublabel}</p>}
    </article>
  );
}

function HistogramChart({ counts, edges, color = "#377f7b" }: { counts: number[]; edges: number[]; color?: string }) {
  const width = 280;
  const height = 110;
  const max = Math.max(...counts, 1);
  const barWidth = width / counts.length;
  const min = edges[0];
  const maxEdge = edges[edges.length - 1];
  return (
    <svg className="histogram-chart" viewBox={`0 0 ${width} ${height + 20}`} role="img" aria-label="Feature value distribution histogram">
      {counts.map((count, index) => {
        const barHeight = count === 0 ? 0 : Math.max(1, (count / max) * height);
        return (
          <rect
            key={index}
            x={index * barWidth + 1}
            y={height - barHeight}
            width={Math.max(barWidth - 2, 1)}
            height={barHeight}
            fill={color}
            rx="1.5"
          />
        );
      })}
      <line x1="0" x2={width} y1={height} y2={height} className="chart-grid" />
      <text x="0" y={height + 15} textAnchor="start" className="chart-tick">{min.toFixed(2)}</text>
      <text x={width} y={height + 15} textAnchor="end" className="chart-tick">{maxEdge.toFixed(2)}</text>
    </svg>
  );
}

function PercentileCalculator({ curve }: { curve: PercentilePoint[] }) {
  const [input, setInput] = useState("");
  const numeric = input === "" ? null : Number(input);
  const percentile = numeric !== null && Number.isFinite(numeric) ? interpolatePercentile(curve, numeric) : null;
  return (
    <div className="calculator">
      <label>
        <span>Enter a hypothetical value</span>
        <input
          type="number"
          inputMode="decimal"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder={curve[10].value.toFixed(3)}
        />
      </label>
      {percentile !== null ? (
        <p className="calculator-result">
          A value of <strong>{numeric}</strong> sits at approximately the <strong>{percentile.toFixed(0)}th percentile</strong> of this population.
        </p>
      ) : (
        <p className="calculator-result muted">No real patient values are used here — this only interpolates your entry against the population&apos;s percentile curve.</p>
      )}
    </div>
  );
}

type EndpointKind = "survival" | "binary" | "multiclass";

function DiscriminationPanel({ feature, kind }: { feature: FeatureEntry; kind: EndpointKind }) {
  if (kind === "survival") {
    const d = feature.discrimination as ConcordanceDiscrimination;
    return (
      <div className="discrimination-block">
        <div className="metric-grid">
          <StatCard label="Concordance index" value={d.concordance.toFixed(3)} sublabel={`95% bootstrap interval ${d.ciLow.toFixed(3)}–${d.ciHigh.toFixed(3)}`} primary />
        </div>
        <p className="micro-note">This feature&apos;s raw value is used directly as a risk score — no threshold or model fitting involved, just a ranking comparison against observed survival time.</p>
      </div>
    );
  }
  if (kind === "binary") {
    const d = feature.discrimination as BinaryDiscrimination;
    return (
      <div className="discrimination-block">
        <div className="metric-grid">
          <StatCard label="Univariate ROC AUC" value={d.auc.toFixed(3)} sublabel={`95% bootstrap interval ${d.ciLow.toFixed(3)}–${d.ciHigh.toFixed(3)}`} primary />
          <StatCard label="Sensitivity" value={d.sensitivity.toFixed(3)} sublabel="at the stated threshold" />
          <StatCard label="Specificity" value={d.specificity.toFixed(3)} sublabel="at the stated threshold" />
          <StatCard label="PPV" value={Number.isFinite(d.ppv) ? d.ppv.toFixed(3) : "—"} sublabel="positive predictive value" />
          <StatCard label="NPV" value={Number.isFinite(d.npv) ? d.npv.toFixed(3) : "—"} sublabel="negative predictive value" />
        </div>
        <p className="micro-note">Threshold rule: {d.thresholdRule} (Youden&apos;s J, chosen post hoc on this cohort — see the info icon above).</p>
      </div>
    );
  }
  const d = feature.discrimination as OneVsRestDiscrimination;
  const maxAuc = Math.max(...d.byClass.map((c) => c.auc), 0.01);
  return (
    <div className="discrimination-block">
      <div className="landscape-group">
        {d.byClass.map((cls, index) => (
          <div className="mini-bar-row" key={cls.label}>
            <span title={cls.label}>{cls.label}</span>
            <div><i style={{ width: `${(cls.auc / maxAuc) * 100}%`, background: curveColors[index % curveColors.length] }} /></div>
            <strong>{cls.auc.toFixed(3)}</strong>
          </div>
        ))}
      </div>
      <p className="micro-note">One-vs-rest AUC per subtype, with 95% bootstrap intervals: {d.byClass.map((c) => `${c.label} ${c.ciLow.toFixed(2)}–${c.ciHigh.toFixed(2)}`).join(" · ")}.</p>
      <div className={`signal-note ${d.summary.startsWith("Primarily") ? "" : "negative"}`}>
        <strong>{d.summary}</strong>
      </div>
    </div>
  );
}

function FeatureDetail({ feature, classes, kind }: { feature: FeatureEntry; classes: string[]; kind: EndpointKind }) {
  const featureData = useFeatureData();
  const dist = feature.distribution;
  const groupNoun = kind === "survival" ? "event status" : "outcome";
  return (
    <div className="feature-detail">
      <div className="feature-focus">
        <div>
          <p>Selected feature</p>
          <h4>{feature.display}</h4>
          <code>{feature.feature}</code>
        </div>
        <dl>
          <div><dt>Rank</dt><dd>{feature.rank}</dd></div>
          <div><dt>Fold stability</dt><dd>{Math.round(feature.stability * 100)}%</dd></div>
          <div><dt>FDR q</dt><dd>{formatQ(feature.fdrQ)}</dd></div>
          <div><dt>Direction</dt><dd>{feature.direction}</dd></div>
        </dl>
      </div>

      <div className="analysis-grid">
        <article className="panel">
          <div className="panel-heading">
            <div><p>Distribution</p><h3>Median, spread, and shape</h3></div>
            <div className="panel-heading-meta">
              <InfoTip label="Distribution statistics">
                <p><strong>What this shows:</strong> descriptive statistics computed over the {dist.n} evaluable patients for this endpoint. Radiomic features are often skewed, so median/IQR are shown alongside mean/SD.</p>
                <p><strong>How to read it:</strong> a large gap between mean and median, or a skewness far from 0, means the mean/SD pair is less representative than the median/IQR pair.</p>
              </InfoTip>
            </div>
          </div>
          <div className="metric-grid compact">
            <StatCard label="Median (IQR)" value={dist.median.toFixed(3)} sublabel={`IQR ${dist.iqr.toFixed(3)} (${dist.p25.toFixed(3)}–${dist.p75.toFixed(3)})`} primary />
            <StatCard label="Mean (SD)" value={dist.mean.toFixed(3)} sublabel={`SD ${dist.sd.toFixed(3)}`} />
            <StatCard label="Min–max" value={`${dist.min.toFixed(2)}–${dist.max.toFixed(2)}`} sublabel={`5th–95th pct ${dist.p5.toFixed(2)}–${dist.p95.toFixed(2)}`} />
            <StatCard label="Skewness" value={dist.skewness.toFixed(2)} sublabel={Math.abs(dist.skewness) < 0.5 ? "roughly symmetric" : "notably skewed"} />
            <StatCard label="Extreme values" value={`${dist.extremeCount}`} sublabel={`${dist.extremePct}% beyond 1.5×IQR`} />
          </div>
          <HistogramChart counts={feature.histogram.counts} edges={feature.histogram.binEdges} />
        </article>

        <article className="panel">
          <div className="panel-heading">
            <div><p>Calculator</p><h3>Where does a value fall?</h3></div>
            <div className="panel-heading-meta">
              <InfoTip label="Percentile calculator">
                <p><strong>What this shows:</strong> a client-side lookup against this feature&apos;s population percentile curve.</p>
                <p><strong>How to read it:</strong> this tool never touches real patient data — you can type any hypothetical value to see where it would land in this population&apos;s distribution.</p>
              </InfoTip>
            </div>
          </div>
          <PercentileCalculator curve={feature.percentileCurve} key={feature.feature} />
        </article>
      </div>

      <article className="panel">
        <div className="panel-heading">
          <div><p>Group comparison</p><h3>By {groupNoun} group</h3></div>
          <div className="panel-heading-meta">
            <InfoTip label="Group comparison">
              <p><strong>What this shows:</strong> median/IQR per {groupNoun} class, and the standardized effect size ({kind === "survival" ? "log hazard ratio per SD, from a univariate Cox fit" : "with 95% bootstrap CI"}), unadjusted p-value, and FDR q-value for this feature — reused from the main dashboard&apos;s Feature registry.{kind === "survival" ? " The group split is by event status only and is not adjusted for follow-up time." : ""}</p>
              <p><strong>How to read it:</strong> weigh the effect size and its confidence interval more than the p-value. A CI that includes zero (no difference) means the direction is uncertain even if the point estimate looks large.</p>
            </InfoTip>
          </div>
        </div>
        <div className="metric-grid compact">
          <StatCard
            label={kind === "survival" ? "Log hazard ratio (per SD)" : "Standardized effect size"}
            value={feature.effect.toFixed(3)}
            sublabel={feature.effectCiLow !== null && feature.effectCiHigh !== null ? `95% CI ${feature.effectCiLow.toFixed(3)} to ${feature.effectCiHigh.toFixed(3)}` : "CI unavailable"}
            primary
          />
          <StatCard label="Unadjusted p" value={formatQ(feature.univariateP)} />
          <StatCard label="FDR q" value={formatQ(feature.fdrQ)} />
        </div>
        <div className="table-shell">
          <table className="feature-table">
            <thead>
              <tr><th>Group</th><th>n</th><th>Median</th><th>IQR</th><th>Distribution</th></tr>
            </thead>
            <tbody>
              {feature.groups.map((group, index) => (
                <tr key={group.label}>
                  <td>
                    <span className="legend-dot" style={{ background: curveColors[classes.indexOf(group.label) % curveColors.length] }} /> {group.label}
                  </td>
                  <td>{group.n}</td>
                  <td>{group.median.toFixed(3)}</td>
                  <td>{group.iqr.toFixed(3)}</td>
                  <td>
                    <HistogramChart counts={group.histogram.counts} edges={feature.histogram.binEdges} color={curveColors[index % curveColors.length]} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>

      <article className="panel">
        <div className="panel-heading">
          <div><p>Univariate discrimination</p><h3>{kind === "survival" ? "Survival concordance" : kind === "binary" ? `${classes[1] ?? "Positive class"} discrimination` : "One-vs-rest across classes"}</h3></div>
          <div className="panel-heading-meta">
            <InfoTip label="Univariate discrimination">
              <p><strong>What this shows:</strong> how well this single feature, alone, {kind === "survival" ? "ranks patients by observed survival time" : `separates ${kind === "binary" ? `${classes[1] ?? "the positive class"} from ${classes[0] ?? "the rest"}` : "each class from the rest"}`} — {featureData.meta.discriminationThreshold}</p>
              <p><strong>How to read it:</strong> these are descriptive, single-feature results. A modest univariate {kind === "survival" ? "concordance index" : "AUC"} does not rule out the feature contributing inside a multivariable model, and a strong one does not guarantee it survives cross-validation (see the main dashboard for the validated result).</p>
            </InfoTip>
          </div>
        </div>
        <DiscriminationPanel feature={feature} kind={kind} />
      </article>
    </div>
  );
}

function FeaturePicker({
  features,
  selected,
  onSelect,
}: {
  features: FeatureEntry[];
  selected: FeatureEntry | null;
  onSelect: (feature: FeatureEntry) => void;
}) {
  const [region, setRegion] = useState("All");
  const [family, setFamily] = useState("All");
  const [query, setQuery] = useState("");

  const regions = ["All", ...Array.from(new Set(features.map((f) => f.region)))];
  const families = ["All", ...Array.from(new Set(features.map((f) => f.family)))];
  const filtered = useMemo(
    () =>
      features.filter((f) => {
        const matchesRegion = region === "All" || f.region === region;
        const matchesFamily = family === "All" || f.family === family;
        const haystack = `${f.display} ${f.feature}`.toLowerCase();
        return matchesRegion && matchesFamily && haystack.includes(query.toLowerCase());
      }),
    [features, region, family, query],
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
      </div>
      <div className="table-shell">
        <table className="feature-table">
          <thead>
            <tr><th>Rank</th><th>Feature</th><th>Region</th><th>Stability</th></tr>
          </thead>
          <tbody>
            {filtered.map((feature) => (
              <tr key={feature.feature} className={selected?.feature === feature.feature ? "selected-row" : ""}>
                <td>{feature.rank}</td>
                <td>
                  <button className="feature-name" onClick={() => onSelect(feature)}>
                    <strong>{feature.measure}</strong>
                    <span>{feature.transform} · {feature.family}</span>
                  </button>
                </td>
                <td><span className={`region-pill ${feature.region.toLowerCase()}`}>{feature.region}</span></td>
                <td>
                  <div className="stability-cell">
                    <span>{Math.round(feature.stability * 100)}%</span>
                    <i><b style={{ width: `${feature.stability * 100}%` }} /></i>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!filtered.length && <p className="empty-state">No features match these filters.</p>}
      </div>
    </>
  );
}

function DataQualityExplorer() {
  const featureData = useFeatureData();
  const rows = featureData.dataQuality.features;
  const [region, setRegion] = useState("All");
  const [family, setFamily] = useState("All");
  const [quality, setQuality] = useState("All");
  const [query, setQuery] = useState("");
  const [showAll, setShowAll] = useState(false);

  const regions = ["All", ...Array.from(new Set(rows.map((r) => r.region)))];
  const families = ["All", ...Array.from(new Set(rows.map((r) => r.family)))];
  const filtered = useMemo(
    () =>
      rows.filter((r) => {
        const matchesRegion = region === "All" || r.region === region;
        const matchesFamily = family === "All" || r.family === family;
        const matchesQuality = quality === "All" || r.quality === quality;
        const haystack = `${r.display} ${r.feature}`.toLowerCase();
        return matchesRegion && matchesFamily && matchesQuality && haystack.includes(query.toLowerCase());
      }),
    [rows, region, family, quality, query],
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
        <label>
          <span>Quality</span>
          <select value={quality} onChange={(event) => setQuality(event.target.value)}>
            {["All", "good", "caution", "poor"].map((item) => <option key={item}>{item}</option>)}
          </select>
        </label>
        <label className="search-field">
          <span>Search all {rows.length.toLocaleString()} features</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="e.g. wavelet glcm" />
        </label>
      </div>
      <div className="table-shell">
        <table className="feature-table">
          <thead>
            <tr><th>Feature</th><th>Region</th><th>Family</th><th>Missing</th><th>Outliers</th><th>Quality</th></tr>
          </thead>
          <tbody>
            {filtered.slice(0, showAll ? filtered.length : 20).map((row) => (
              <tr key={row.feature}>
                <td>
                  <strong title={row.feature}>{row.display}</strong>
                </td>
                <td><span className={`region-pill ${row.region.toLowerCase()}`}>{row.region}</span></td>
                <td>{row.family}</td>
                <td>{(row.missingRate * 100).toFixed(1)}%</td>
                <td>{(row.outlierRate * 100).toFixed(1)}%{row.nearConstant ? " (near-constant)" : ""}</td>
                <td><QualityBadge quality={row.quality} /></td>
              </tr>
            ))}
          </tbody>
        </table>
        {!filtered.length && <p className="empty-state">No features match these filters.</p>}
      </div>
      {filtered.length > 20 && (
        <button className="table-toggle" onClick={() => setShowAll((v) => !v)}>
          {showAll ? "Show top 20" : `Show all ${filtered.length.toLocaleString()} matching features`}
        </button>
      )}
    </>
  );
}

function QualitySummary() {
  const featureData = useFeatureData();
  const meta = featureData.meta;
  const rollup = featureData.dataQuality.rollup.overall;
  return (
    <>
      <div className="cohort-strip" aria-label="Cohort summary">
        <div><strong>{meta.patientCount}</strong><span>unique patients</span></div>
        <div><strong>{meta.endpoints.subtype.sampleCount}</strong><span>evaluable for subtype</span></div>
        <div><strong>{meta.endpoints.pcr.sampleCount}</strong><span>evaluable for pCR</span></div>
        <div><strong>{meta.radiomicFeatureCount.toLocaleString()}</strong><span>radiomic features audited</span></div>
      </div>

      <div className="metric-grid panel-spaced">
        <StatCard label="Good quality" value={rollup.good.toLocaleString()} sublabel={`${((rollup.good / meta.radiomicFeatureCount) * 100).toFixed(0)}% of all features`} primary />
        <StatCard label="Caution" value={rollup.caution.toLocaleString()} sublabel="elevated outlier rate or missingness" />
        <StatCard label="Poor" value={rollup.poor.toLocaleString()} sublabel="near-constant, high missingness, or heavy outliers" />
        <StatCard label="pCR exclusions" value={`${meta.endpoints.pcr.excludedCount}`} sublabel={meta.endpoints.pcr.exclusionReasons.map((r) => `${r.reason}: ${r.count}`).join(" · ")} />
      </div>

      <div className="guardrail-grid unavailable-grid">
        {meta.unavailable.map((item) => (
          <p key={item.topic}><span>!</span><span><strong>{item.topic}.</strong> {item.note}</span></p>
        ))}
      </div>
    </>
  );
}

function FamilyGroupings() {
  const featureData = useFeatureData();
  return (
    <div className="landscape-grid">
      {featureData.familyGroups.map((group) => (
        <div className="context-card" key={group.family}>
          <h4>{group.family}</h4>
          <p className="micro-note">{group.interpretation}</p>
          {!group.available && <p className="micro-note unavailable-note">{group.note}</p>}
        </div>
      ))}
    </div>
  );
}

function FeatureExplorerInner() {
  const { analysisData, version } = useDataset();
  const featureData = useFeatureData();
  const [activeKey, setActiveKey] = useState<EndpointKey>("survival");
  const endpoint = featureData.endpoints[activeKey];
  const kind: EndpointKind = activeKey === "survival" ? "survival" : activeKey === "pcr" ? "binary" : "multiclass";
  const [selected, setSelected] = useState<FeatureEntry | null>(endpoint.features[0] ?? null);

  function handleEndpointChange(key: EndpointKey) {
    setActiveKey(key);
    setSelected(featureData.endpoints[key].features[0] ?? null);
  }

  return (
    <main>
      <header className="topbar">
        <Link className="brand" href="/" aria-label="Radiomics explorer home">
          <span className="brand-mark">RE</span>
          <span><strong>Radiomics Explorer</strong><small>Feature statistics</small></span>
        </Link>
        <nav aria-label="Page sections">
          <a href="#quality">Data quality</a>
          <a href="#groupings">Groupings</a>
          <a href="#features">Feature detail</a>
          <DatasetSwitcher />
        </nav>
        <Link className="download-link" href="/">← Dashboard</Link>
      </header>

      <section className="page-header" id="top">
        <div className="page-header-copy">
          <p className="eyebrow">Companion page · reproducibility &amp; data quality</p>
          <h1>Feature Statistics</h1>
          <p className="page-header-lede">
            Cohort data quality across all {featureData.meta.radiomicFeatureCount.toLocaleString()} radiomic features, plus a full distribution, group-comparison, and univariate-discrimination breakdown for the top 50 stability-ranked features per endpoint — the same features shown in the main dashboard&apos;s Feature registry.
          </p>
        </div>
      </section>

      <section className="section-shell" id="quality">
        <div className="section-heading">
          <div><p className="eyebrow">01 · Cohort &amp; data quality</p><h2>Is this cohort representative and reliable?</h2></div>
          <p>Missing-value rate, near-constant detection, and outlier frequency computed across every feature — not just the ones that made the final model.</p>
        </div>
        <QualitySummary />
        <article className="panel panel-spaced">
          <div className="panel-heading">
            <div><p>Full feature audit</p><h3>Quality by feature</h3></div>
            <div className="panel-heading-meta">
              <InfoTip label="Data quality audit">
                <p><strong>What this shows:</strong> for every one of the {featureData.meta.radiomicFeatureCount.toLocaleString()} radiomic features: missing rate, outlier rate (share of values beyond 1.5×IQR), a near-constant flag, and a derived quality indicator.</p>
                <p><strong>How to read it:</strong> Good = missing rate 0% and outlier rate ≤5% and not near-constant. Caution = any missingness or outlier rate 5–15%. Poor = near-constant, missing rate &gt;5%, or outlier rate &gt;15%. Thresholds are fixed rules, not judgment calls per feature.</p>
              </InfoTip>
            </div>
          </div>
          <DataQualityExplorer />
        </article>
      </section>

      <section className="section-shell tone-section" id="groupings">
        <div className="section-heading">
          <div><p className="eyebrow">02 · Clinically meaningful groupings</p><h2>What these feature families represent</h2></div>
          <p>Framed as imaging phenotypes, not direct measurements of underlying biology.</p>
        </div>
        <FamilyGroupings />
      </section>

      <section className="section-shell" id="features">
        <div className="section-heading">
          <div><p className="eyebrow">03 · Per-feature deep dive</p><h2>Distribution, group comparison, and discrimination</h2></div>
          <p>Choose an endpoint and a feature to see its full univariate profile.</p>
        </div>

        <div className="endpoint-switch" role="group" aria-label="Choose prediction endpoint">
          {(Object.keys(featureData.endpoints) as EndpointKey[]).map((key) => (
            <button key={key} aria-pressed={activeKey === key} onClick={() => handleEndpointChange(key)}>
              <span>{analysisData.endpoints[key].shortTitle}</span>
              <small>{featureData.endpoints[key].sampleCount} evaluable patients</small>
            </button>
          ))}
        </div>

        <article className="panel panel-spaced">
          <div className="panel-heading">
            <div><p>Feature picker</p><h3>Top 50 stability-ranked features</h3></div>
          </div>
          <FeaturePicker key={`${activeKey}-${version}`} features={endpoint.features} selected={selected} onSelect={setSelected} />
        </article>

        {selected && <FeatureDetail key={`${activeKey}-${selected.feature}-${version}`} feature={selected} classes={endpoint.classes} kind={kind} />}
      </section>

      <footer>
        <div><strong>Radiomics Explorer</strong><span>Feature statistics companion page · aggregate only, no patient-level data displayed</span></div>
        <div><span>Source</span><code>feature-explorer.json</code></div>
      </footer>
    </main>
  );
}

function FeatureExplorer() {
  return (
    <DatasetProvider>
      <FeatureExplorerInner />
    </DatasetProvider>
  );
}

export default FeatureExplorer;
