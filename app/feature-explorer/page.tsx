import type { Metadata } from "next";
import FeatureExplorer from "./feature-explorer";

export const metadata: Metadata = {
  title: "Feature Statistics · Radiomics Explorer",
  description: "Cohort data quality and per-feature distribution statistics for a radiomics cohort.",
};

export default function FeatureExplorerPage() {
  return <FeatureExplorer />;
}
