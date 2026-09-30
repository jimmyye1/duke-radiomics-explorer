"use client";

import { useEffect, useRef, useState } from "react";
import { useDataset } from "./dataset-context";

export const curveColors = ["#d6a54c", "#4fa7a1", "#d96c5f", "#758ac7"];

export function formatQ(value: number) {
  if (value < 0.001) return value.toExponential(1);
  return value.toFixed(3);
}

export function InfoTip({ label, children }: { label: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    function handlePointerDown(event: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) setOpen(false);
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  return (
    <div className="info-tip" ref={rootRef}>
      <button
        type="button"
        className="info-tip-trigger"
        aria-expanded={open}
        aria-label={`Explain: ${label}`}
        onClick={() => setOpen((value) => !value)}
      >
        i
      </button>
      {open && (
        <div className="info-tip-panel" role="note" aria-label={`${label} explained`}>
          {children}
        </div>
      )}
    </div>
  );
}

async function readJsonFile(file: File): Promise<unknown> {
  const text = await file.text();
  return JSON.parse(text);
}

export function DatasetSwitcher() {
  const { isCustom, datasetLabel, uploadDataset, resetDataset } = useDataset();
  const dialogRef = useRef<HTMLDialogElement | null>(null);
  const analysisInputRef = useRef<HTMLInputElement | null>(null);
  const featureInputRef = useRef<HTMLInputElement | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  function openDialog() {
    setError(null);
    dialogRef.current?.showModal();
  }

  function closeDialog() {
    dialogRef.current?.close();
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const analysisFile = analysisInputRef.current?.files?.[0];
    if (!analysisFile) {
      setError("Choose an analysis.json file to upload.");
      return;
    }
    setPending(true);
    setError(null);
    try {
      const analysisJson = await readJsonFile(analysisFile);
      const featureFile = featureInputRef.current?.files?.[0];
      const featureJson = featureFile ? await readJsonFile(featureFile) : null;
      const result = uploadDataset(analysisJson, featureJson);
      if (!result.ok) {
        setError(result.error);
        return;
      }
      if (analysisInputRef.current) analysisInputRef.current.value = "";
      if (featureInputRef.current) featureInputRef.current.value = "";
      closeDialog();
    } catch {
      setError("Couldn't parse that as JSON. Make sure you're uploading the file exactly as written by the Python script.");
    } finally {
      setPending(false);
    }
  }

  function handleReset() {
    resetDataset();
    setError(null);
    closeDialog();
  }

  return (
    <>
      <button type="button" className="dataset-switch-trigger" onClick={openDialog}>
        {isCustom ? `Dataset: ${datasetLabel}` : "Load dataset"}
      </button>
      <dialog ref={dialogRef} className="dataset-dialog" onClose={() => setError(null)}>
        <form className="dataset-dialog-form" onSubmit={handleSubmit}>
          <div className="dataset-dialog-heading">
            <h3>Load a dataset</h3>
            <button type="button" className="dataset-dialog-close" aria-label="Close" onClick={closeDialog}>×</button>
          </div>
          <p className="micro-note">
            Upload the JSON produced by running <code>analysis/run_analysis.py</code> (required) and, optionally,{" "}
            <code>analysis/run_feature_explorer.py</code> locally on a different dataset in this same format. Files are read entirely in your browser — nothing is uploaded to a server.
          </p>
          <p className="micro-note">Currently showing: <strong>{isCustom ? datasetLabel : "Sample dataset"}</strong></p>
          <label className="dataset-dialog-field">
            <span>analysis.json (required)</span>
            <input ref={analysisInputRef} type="file" accept="application/json,.json" />
          </label>
          <label className="dataset-dialog-field">
            <span>feature-explorer.json (optional)</span>
            <input ref={featureInputRef} type="file" accept="application/json,.json" />
          </label>
          {error && <p className="dataset-dialog-error">{error}</p>}
          <div className="dataset-dialog-actions">
            {isCustom && (
              <button type="button" className="table-toggle" onClick={handleReset}>
                Reset to sample dataset
              </button>
            )}
            <button type="submit" className="download-link" disabled={pending}>
              {pending ? "Loading…" : "Load dataset"}
            </button>
          </div>
        </form>
      </dialog>
    </>
  );
}
