import React from "react";
import ProbabilityBars from "./ProbabilityBars";
import { COUNTERFACTUAL_FEATURES } from "../constants/counterfactualFeatures";


type CFData = {
  factual_type: string;
  counterfactual_type: string;
  factual_probabilities: Record<string, number>;
  counterfactual_probabilities: Record<string, number>;
};

type Props = {
  data: CFData | null;
  selectedFeature: string;
  onFeatureChange: (feature: string) => void;
  lambda: number;
  onLambdaChange: (value: number) => void;
};


/**
 * CounterfactualPanel
 *
 * Pure presentational component that displays factual vs counterfactual MBTI results.
 * - Renders factual -> counterfactual MBTI types
 * - Shows two ProbabilityBars side-by-side using the provided probabilities
 * - Does not fetch data or perform any computation
 */
export default function CounterfactualPanel({ data, selectedFeature, onFeatureChange, lambda, onLambdaChange, }: Props) {
  if (!data) {
    return <div style={styles.empty}>Counterfactual data unavailable.</div>;
  }

  const {
    factual_type,
    counterfactual_type,
    factual_probabilities,
    counterfactual_probabilities,
  } = data;

  const hasProbs =
    factual_probabilities &&
    typeof factual_probabilities === "object" &&
    counterfactual_probabilities &&
    typeof counterfactual_probabilities === "object";

  if (!hasProbs) {
    return <div style={styles.empty}>Counterfactual data incomplete.</div>;
  }

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Counterfactual Analysis</h3>
      <div style={styles.controls}>
        <div style={styles.controlGroup}>
          <label style={styles.label}>
            Intervention Feature
          </label>
          <select
            value={selectedFeature}
            onChange={(e) => onFeatureChange(e.target.value)}
            style={styles.select}
          >
            {COUNTERFACTUAL_FEATURES.map((f) => (
              <option key={f.key} value={f.key}>
                {f.label}
              </option>
            ))}
          </select>
        </div>

        <div style={styles.controlGroup}>
          <label style={styles.label}>
            Intervention Strength (λ): <strong>{lambda.toFixed(2)}</strong>
          </label>

          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={lambda}
            onChange={(e) => onLambdaChange(Number(e.target.value))}
            style={styles.rangeInput}
          />

          <div style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)", marginTop: 4 }}>
            0 = factual · 1 = strong intervention
          </div>
        </div>
      </div>

      <div style={styles.typeRow}>
        <div style={styles.typeBox}>
          <div style={styles.typeLabel}>Factual</div>
          <div style={styles.typeValue}>{factual_type}</div>
        </div>

        <div style={styles.arrow} aria-hidden>
          →{/* visual arrow */}
        </div>

        <div style={styles.typeBox}>
          <div style={styles.typeLabel}>Counterfactual</div>
          <div style={styles.typeValue}>{counterfactual_type}</div>
        </div>
      </div>

      <div style={styles.barsRow}>
        <div style={styles.barColumn}>
          <h4 style={styles.subTitle}>Factual Probabilities</h4>
          <ProbabilityBars probabilities={factual_probabilities} />
        </div>

        <div style={styles.barColumn}>
          <h4 style={styles.subTitle}>Counterfactual Probabilities</h4>
          <ProbabilityBars probabilities={counterfactual_probabilities} />
        </div>
      </div>
    </div>
  );
}

/* Minimal inline styles */
/* Professional Academic Styles */
const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: "24px",
    border: "1px solid var(--color-border)",
    borderRadius: "var(--radius-lg)",
    background: "var(--color-bg)", // Slightly different bg to nest inside the white section or just white?
    // Actually, Result.tsx section is white. Let's make this panel distinct or just clean.
    // Let's keep it clean, maybe just a borderless container since it is inside a card.
    // But the original had a border. Let's make it a "sub-card" or just a layout.
    // Let's remove the background and border to integrate it better, or keep it as a highlighted area.
    // Let's go with a highlighted area.
    backgroundColor: "#f8f9fa", // Light neutral
  },
  title: {
    margin: "0 0 24px 0",
    fontSize: "1.2rem",
    color: "var(--color-primary)",
    borderBottom: "1px solid var(--color-border)",
    paddingBottom: "12px",
    display: "none", // Hidden because the parent section already has a title
  },
  // Controls area
  controls: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "24px",
    marginBottom: "32px",
    padding: "16px",
    background: "var(--color-surface)",
    border: "1px solid var(--color-border)",
    borderRadius: "var(--radius-md)",
  },
  controlGroup: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  label: {
    fontSize: "0.9rem",
    fontWeight: 600,
    color: "var(--color-text-secondary)",
  },
  select: {
    padding: "8px 12px",
    borderRadius: "var(--radius-sm)",
    border: "1px solid var(--color-border)",
    fontSize: "0.95rem",
    fontFamily: "inherit",
  },
  rangeInput: {
    width: "100%",
    cursor: "pointer",
  },

  typeRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "32px",
    marginBottom: "32px",
  },
  typeBox: {
    minWidth: 160,
    padding: "20px",
    borderRadius: "var(--radius-lg)",
    background: "var(--color-surface)",
    textAlign: "center",
    boxShadow: "var(--shadow-sm)",
    border: "1px solid var(--color-border)",
    position: "relative",
  },
  typeLabel: {
    fontSize: "0.85rem",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    color: "var(--color-text-secondary)",
    marginBottom: "8px",
    fontWeight: 600
  },
  typeValue: {
    fontSize: "2.5rem",
    fontWeight: 800,
    color: "var(--color-primary)",
    lineHeight: 1,
  },
  arrow: {
    fontSize: "2rem",
    color: "var(--color-text-secondary)",
    opacity: 0.5
  },

  barsRow: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "32px",
    alignItems: "start",
  },
  barColumn: {
    flex: 1,
    background: "var(--color-surface)",
    padding: "16px",
    borderRadius: "var(--radius-md)",
    border: "1px solid var(--color-border)",
  },
  subTitle: {
    margin: "0 0 16px 0",
    fontSize: "1rem",
    fontWeight: 600,
    color: "var(--color-text-main)",
    textAlign: "center"
  },
  empty: {
    fontStyle: "italic",
    color: "var(--color-text-secondary)",
    padding: "24px",
    textAlign: "center",
    background: "var(--color-bg)",
    borderRadius: "var(--radius-md)"
  },
};