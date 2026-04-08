// ================================
// FILE STATUS: FROZEN
// Verified working with backend
// Do NOT modify unless API changes
// ================================

import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import ProbabilityBars from "../components/ProbabilityBars";
import RadarChart from "../components/RadarChart";
import SentenceAttribution from "../components/SentenceAttribution";
import ExplanationPanel from "../components/ExplanationPanel";
import CounterfactualPanel from "../components/CounterfactualPanel";
import DeltaProbabilityBars from "../components/DeltaProbabilityBars";



export interface MBTIResponse {
  mbti: string;
  confidence: number;
  probabilities: Record<string, number>;
  explanation: string;
  cognitive_features?: Record<string, number>;
  sentence_attribution?: Array<{ sentence: string; weight: number }>;
}

export interface CounterfactualResponse {
  factual_type: string;
  counterfactual_type: string;
  factual_probabilities: Record<string, number>;
  counterfactual_probabilities: Record<string, number>;
  delta_probabilities?: Record<string, number>;
  sensitivity?: Record<string, number>;
  trait_flip?: Record<string, boolean>;
  counterfactual_explanation?: string;
}

export default function Result() {
  const location = useLocation();
  const navigate = useNavigate();

  console.log("🔍 Result page - location.state:", location.state);
  console.log("🔍 Result page - pathname:", location.pathname);

  const state = location.state as
    | { response?: MBTIResponse; text?: string }
    | undefined;

  const response = state?.response;

  console.log("🔍 Result page - response exists:", !!response);
  console.log("🔍 Result page - text exists:", !!state?.text);

  const [textToUse, setTextToUse] = useState<string>("");
  const [counterfactual, setCounterfactual] = useState<CounterfactualResponse | null>(null);
  const [counterfactualLoading, setCounterfactualLoading] = useState(false);
  const [counterfactualError, setCounterfactualError] = useState<string | null>(null);
  const [selectedFeature, setSelectedFeature] = useState<string>("pronoun_ratio");
  const [lambda, setLambda] = useState<number>(0.3);



  // Resolve text for counterfactual analysis
  useEffect(() => {
    console.log("🔍 Result.tsx - Resolving text for counterfactual...");
    console.log("  - state?.text:", state?.text ? `${state.text.length} chars` : "MISSING");

    // Priority 1: Text from navigation state
    if (state?.text) {
      console.log("✅ Using text from navigation state");
      setTextToUse(state.text);
      return;
    }

    // Priority 2: Text from sessionStorage (for same-tab navigation)
    try {
      const sessionText = sessionStorage.getItem("mbti_input_text");
      if (sessionText) {
        console.log("✅ Using text from sessionStorage");
        setTextToUse(sessionText);
        return;
      }
    } catch (e) {
      console.warn("Could not read from sessionStorage:", e);
    }

    // Priority 3: Text from localStorage (for page refreshes)
    try {
      const localText = localStorage.getItem("mbti_input_text");
      if (localText) {
        console.log("✅ Using text from localStorage");
        setTextToUse(localText);
        return;
      }
    } catch (e) {
      console.warn("Could not read from localStorage:", e);
    }

    // Priority 4: Extract from saved result
    try {
      const savedResult = localStorage.getItem("last_mbti_result");
      if (savedResult) {
        const parsed = JSON.parse(savedResult);
        if (parsed.text) {
          console.log("✅ Using text from saved result");
          setTextToUse(parsed.text);
          return;
        }
      }
    } catch (e) {
      console.warn("Could not read saved result:", e);
    }

    // No text found - but DON'T redirect immediately
    // Wait to see if we have response data
    console.log("❌ No text found in state or storage");
  }, [state]);

  // Manual counterfactual fetch function
  const fetchCounterfactual = () => {
    console.log("🔘 Button clicked! textToUse:", textToUse ? `${textToUse.length} chars` : "EMPTY");

    if (!textToUse || textToUse.trim() === "") {
      console.error("❌ Cannot run counterfactual: textToUse is empty");
      setCounterfactualError("No text available for counterfactual analysis");
      return;
    }

    console.log("🚀 Triggering POST /counterfactual with text:", textToUse.substring(0, 50));

    setCounterfactualLoading(true);
    setCounterfactualError(null);

    fetch(`${import.meta.env.VITE_API_BASE_URL}/counterfactual`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: textToUse, intervention_feature: selectedFeature, intervention_lambda: lambda, }),
    })
      .then((res) => {
        if (!res.ok) {
          return res.text().then(text => {
            throw new Error(`Counterfactual analysis failed: ${text || res.statusText}`);
          });
        }
        return res.json();
      })
      .then((data) => {
        console.log("✅ Counterfactual response received:", data);
        setCounterfactual(data);
      })
      .catch((err) => {
        console.error("Counterfactual fetch failed:", err);
        setCounterfactualError(err.message);
      })
      .finally(() => {
        setCounterfactualLoading(false);
      });
  };

  // Safety check - show loading or redirect only if we definitely have no data
  if (!response) {
    // Check if we're still loading or if data is truly missing
    const savedResponse = localStorage.getItem("last_mbti_result");

    if (savedResponse) {
      // Try to use saved response from localStorage
      try {
        const parsed = JSON.parse(savedResponse);

        // Show cached data with option to refresh
        return (
          <div style={styles.container}>
            <div style={styles.warning}>
              <h3>Previous Results (Cached)</h3>
              <p>Showing results from your previous submission.</p>
              {parsed.response && parsed.response.mbti && (
                <div style={{ margin: '20px 0' }}>
                  <h4>MBTI: {parsed.response.mbti}</h4>
                  <p>Confidence: {parsed.response.confidence?.toFixed(2)}</p>
                </div>
              )}
              <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
                <button onClick={() => navigate("/")}>Submit New Text</button>
                <button onClick={() => window.location.reload()}>Try Reload</button>
              </div>
            </div>
          </div>
        );
      } catch (e) {
        // If parsing fails, redirect
        navigate("/");
        return null;
      }
    }

    // Show loading state briefly before redirecting
    return (
      <div style={styles.container}>
        <div style={styles.loading}>
          Loading results...
          <button
            onClick={() => navigate("/")}
            style={{ marginTop: 20 }}
          >
            Return to Input
          </button>
        </div>
      </div>
    );
  }

  const {
    mbti,
    confidence,
    probabilities,
    explanation,
    cognitive_features,
    sentence_attribution,
  } = response;

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div>
          <h1 style={styles.mbti}>{mbti}</h1>
          <div style={styles.confidence}>
            Confidence: {confidence.toFixed(2)}
          </div>
        </div>
        <button onClick={() => navigate("/")} style={styles.backButton}>
          Back
        </button>
      </header>

      <section style={styles.section}>
        <h3 style={styles.sectionTitle}>Dimension Probabilities</h3>
        <ProbabilityBars probabilities={probabilities} />
      </section>

      {cognitive_features && (
        <section style={styles.section}>
          <h3 style={styles.sectionTitle}>Cognitive Trait Radar</h3>
          <RadarChart traits={cognitive_features} />
        </section>
      )}

      {sentence_attribution && sentence_attribution.length > 0 && (
        <section style={styles.section}>
          <h3 style={styles.sectionTitle}>Sentence Attribution</h3>
          <SentenceAttribution attributions={sentence_attribution} />
        </section>
      )}

      <section style={styles.section}>
        <h3 style={styles.sectionTitle}>Explanation</h3>
        <ExplanationPanel explanation={explanation} />
      </section>

      <section style={styles.section}>
        <h3 style={styles.sectionTitle}>Counterfactual Analysis</h3>

        {counterfactualLoading && (
          <div style={styles.loading}>
            <div style={styles.spinner}></div>
            Analyzing counterfactual scenario...
          </div>
        )}

        {counterfactualError && (
          <div style={styles.error}>
            ⚠️ Unable to load counterfactual analysis: {counterfactualError}
            <button
              onClick={fetchCounterfactual}
              style={{ ...styles.backButton, marginTop: 12 }}
            >
              Retry
            </button>
          </div>
        )}

        {counterfactual && !counterfactualLoading && (
          <CounterfactualPanel
            data={counterfactual}
            selectedFeature={selectedFeature}
            onFeatureChange={setSelectedFeature}
            lambda={lambda}
            onLambdaChange={setLambda}
          />
        )}

        {!counterfactual && !counterfactualLoading && !counterfactualError && (
          <div style={styles.info}>
            <p>Click the button below to analyze how changing cognitive features would affect your MBTI prediction.</p>
            <button
              onClick={fetchCounterfactual}
              style={{ ...styles.backButton, background: '#2d6cdf', color: '#fff', marginTop: 12 }}
              disabled={!textToUse}
            >
              Run Counterfactual Analysis
            </button>
          </div>
        )}

        {counterfactual?.delta_probabilities && (
          <section style={{ marginTop: 24 }}>
            <h4>Δ Probability (Causal Shift)</h4>
            <DeltaProbabilityBars
              deltas={counterfactual.delta_probabilities}
            />
          </section>
        )}

        {counterfactual?.sensitivity && (
          <section style={{ marginTop: 24 }}>
            <h4>Trait Sensitivity</h4>
            <ul style={{ fontSize: 14 }}>
              {Object.entries(counterfactual.sensitivity).map(([dim, score]) => (
                <li key={dim}>
                  {dim}: {score.toFixed(3)}
                </li>
              ))}
            </ul>
          </section>
        )}
        {counterfactual?.trait_flip && (
          <section style={{ marginTop: 24 }}>
            <h4 style={{ marginBottom: 6 }}>Decision Logic (Trait Flip)</h4>

            <ul style={{ fontSize: 13 }}>
              {Object.entries(counterfactual.trait_flip).map(([dim, flipped]) => (
                <li key={dim}>
                  <strong>{dim}</strong>:{" "}
                  {flipped ? "🔁 Trait Flipped" : "✓ Stable"}
                </li>
              ))}
            </ul>

            <div style={{ fontSize: 11, color: "#666", marginTop: 6 }}>
              Flip rule: P ≥ 0.6 → ≤ 0.4 (or vice-versa)
            </div>
          </section>
        )}

        {counterfactual?.counterfactual_explanation && (
          <section style={{ marginTop: 24 }}>
            <h4>Counterfactual Explanation</h4>
            <ExplanationPanel
              explanation={counterfactual.counterfactual_explanation}
            />
          </section>
        )}
      </section>
    </div>
  );
}

/* Professional Academic Styles */
const styles: Record<string, React.CSSProperties> = {
  container: {
    maxWidth: 1100,
    margin: "40px auto",
    padding: "0 24px",
    fontFamily: "var(--font-family)",
    display: "flex", // Use flex column for spacing
    flexDirection: "column",
    gap: "32px",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "32px",
    background: "var(--color-surface)",
    borderRadius: "var(--radius-lg)",
    boxShadow: "var(--shadow-md)",
    border: "1px solid var(--color-border)",
  },
  mbti: {
    margin: 0,
    fontSize: "3.5rem",
    fontWeight: 800,
    color: "var(--color-primary)",
    letterSpacing: "-0.02em",
    lineHeight: 1,
  },
  confidence: {
    marginTop: 8,
    color: "var(--color-text-secondary)",
    fontSize: "1rem",
    fontWeight: 500,
  },
  backButton: {
    padding: "10px 20px",
    fontSize: "0.9rem",
    fontWeight: 600,
    borderRadius: "var(--radius-md)",
    border: "1px solid var(--color-border)",
    background: "var(--color-bg)",
    color: "var(--color-text-main)",
    cursor: "pointer",
    transition: "all 0.2s",
  },
  section: {
    padding: "32px",
    borderRadius: "var(--radius-lg)",
    border: "1px solid var(--color-border)",
    background: "var(--color-surface)",
    boxShadow: "var(--shadow-sm)",
  },
  sectionTitle: {
    fontSize: "1.25rem",
    fontWeight: 600,
    color: "var(--color-primary)",
    marginBottom: "24px",
    borderBottom: "2px solid var(--color-bg)",
    paddingBottom: "12px",
    display: "inline-block", // Underline only text
  },
  loading: {
    padding: 60,
    textAlign: "center",
    color: "var(--color-text-secondary)",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 16,
    background: "var(--color-surface)",
    borderRadius: "var(--radius-lg)",
    boxShadow: "var(--shadow-sm)",
  },
  spinner: {
    width: 40,
    height: 40,
    border: "4px solid var(--color-bg)",
    borderTop: "4px solid var(--color-accent)",
    borderRadius: "50%",
    animation: "spin 1s linear infinite",
  },
  error: {
    padding: "24px",
    background: "#fef2f2",
    color: "var(--color-error)",
    borderRadius: "var(--radius-md)",
    border: "1px solid #fee2e2",
    display: "flex",
    flexDirection: "column",
    alignItems: "flex-start",
    gap: 12
  },
  info: {
    padding: "24px",
    background: "var(--color-bg)",
    color: "var(--color-text-main)",
    borderRadius: "var(--radius-md)",
    border: "1px solid var(--color-border)",
    textAlign: "center",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 16
  },
  warning: {
    padding: 40,
    textAlign: "center",
    background: "#fffbeb",
    border: "1px solid #fcd34d",
    color: "#92400e",
    borderRadius: "var(--radius-lg)",
    marginTop: 50,
    boxShadow: "var(--shadow-md)",
  },
};