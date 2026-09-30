"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import defaultAnalysis from "./data/analysis.json";
import defaultFeatureExplorer from "./data/feature-explorer.json";

export type AnalysisData = typeof defaultAnalysis;
export type FeatureExplorerData = typeof defaultFeatureExplorer;

const ANALYSIS_KEY = "radiomics-explorer:analysis";
const FEATURE_KEY = "radiomics-explorer:feature-explorer";
const LABEL_KEY = "radiomics-explorer:label";
const REQUIRED_ENDPOINTS = ["survival", "subtype", "pcr"];

export type UploadResult = { ok: true } | { ok: false; error: string };

interface DatasetState {
  analysisData: AnalysisData;
  featureExplorerData: FeatureExplorerData;
  isCustom: boolean;
  datasetLabel: string;
  version: number;
  uploadDataset: (analysis: unknown, featureExplorer: unknown | null) => UploadResult;
  resetDataset: () => void;
}

const DatasetContext = createContext<DatasetState | null>(null);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function hasRequiredEndpoints(endpoints: unknown): boolean {
  return isRecord(endpoints) && REQUIRED_ENDPOINTS.every((key) => isRecord(endpoints[key]));
}

function validateAnalysis(value: unknown): value is AnalysisData {
  return isRecord(value) && isRecord(value.meta) && Array.isArray(value.guardrails) && hasRequiredEndpoints(value.endpoints);
}

function validateFeatureExplorer(value: unknown): value is FeatureExplorerData {
  return isRecord(value) && isRecord(value.meta) && isRecord(value.dataQuality) && hasRequiredEndpoints(value.endpoints);
}

function readStored<T>(key: string): T | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

export function DatasetProvider({ children }: { children: React.ReactNode }) {
  const [analysisData, setAnalysisData] = useState<AnalysisData>(defaultAnalysis);
  const [featureExplorerData, setFeatureExplorerData] = useState<FeatureExplorerData>(defaultFeatureExplorer);
  const [isCustom, setIsCustom] = useState(false);
  const [datasetLabel, setDatasetLabel] = useState("Sample dataset");
  const [version, setVersion] = useState(0);

  // Client-only: hydrate from a previously uploaded dataset after mount, so
  // the server-rendered pass always matches the bundled sample (no hydration
  // mismatch) and localStorage is only ever touched in the browser. This is a
  // one-time mount read, not a reactive subscription, so useSyncExternalStore
  // doesn't fit here.
  useEffect(() => {
    const storedAnalysis = readStored<unknown>(ANALYSIS_KEY);
    if (!storedAnalysis || !validateAnalysis(storedAnalysis)) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setAnalysisData(storedAnalysis);
    setIsCustom(true);
    setVersion((v) => v + 1);
    const storedLabel = window.localStorage.getItem(LABEL_KEY);
    if (storedLabel) setDatasetLabel(storedLabel);
    const storedFeature = readStored<unknown>(FEATURE_KEY);
    if (storedFeature && validateFeatureExplorer(storedFeature)) {
      setFeatureExplorerData(storedFeature);
    }
  }, []);

  useEffect(() => {
    document.title = analysisData.meta.title || "Radiomics Explorer";
  }, [analysisData.meta.title]);

  const uploadDataset = useCallback((analysis: unknown, featureExplorer: unknown | null): UploadResult => {
    if (!validateAnalysis(analysis)) {
      return { ok: false, error: "This doesn't look like a valid analysis.json — it needs meta, guardrails, and endpoints.survival/subtype/pcr fields." };
    }
    if (featureExplorer !== null && !validateFeatureExplorer(featureExplorer)) {
      return { ok: false, error: "This doesn't look like a valid feature-explorer.json — it needs meta, dataQuality, and endpoints.survival/subtype/pcr fields." };
    }
    const label = analysis.meta.title || "Custom dataset";
    setAnalysisData(analysis);
    setDatasetLabel(label);
    setIsCustom(true);
    setVersion((v) => v + 1);
    window.localStorage.setItem(ANALYSIS_KEY, JSON.stringify(analysis));
    window.localStorage.setItem(LABEL_KEY, label);
    if (featureExplorer !== null) {
      setFeatureExplorerData(featureExplorer);
      window.localStorage.setItem(FEATURE_KEY, JSON.stringify(featureExplorer));
    }
    return { ok: true };
  }, []);

  const resetDataset = useCallback(() => {
    window.localStorage.removeItem(ANALYSIS_KEY);
    window.localStorage.removeItem(FEATURE_KEY);
    window.localStorage.removeItem(LABEL_KEY);
    setAnalysisData(defaultAnalysis);
    setFeatureExplorerData(defaultFeatureExplorer);
    setIsCustom(false);
    setDatasetLabel("Sample dataset");
    setVersion((v) => v + 1);
  }, []);

  const value = useMemo<DatasetState>(
    () => ({ analysisData, featureExplorerData, isCustom, datasetLabel, version, uploadDataset, resetDataset }),
    [analysisData, featureExplorerData, isCustom, datasetLabel, version, uploadDataset, resetDataset],
  );

  return <DatasetContext.Provider value={value}>{children}</DatasetContext.Provider>;
}

export function useDataset(): DatasetState {
  const ctx = useContext(DatasetContext);
  if (!ctx) throw new Error("useDataset must be used within a DatasetProvider");
  return ctx;
}
