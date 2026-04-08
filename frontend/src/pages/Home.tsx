// ================================
// FILE STATUS: FROZEN
// Verified working with backend
// Do NOT modify unless API changes
// ================================


import React, { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import TextInput from "../components/TextInput";
import WordCounter from "../components/WordCounter";

/**
 * API response shape (matches backend api/schemas.py)
 */
export interface MBTIResponse {
  mbti: string;
  confidence: number;
  probabilities: Record<string, number>;
  explanation: string;
}

/**
 * Home - main input page for MBTI prediction
 *
 * Frontend constraints:
 * - UI-only validation (word count)
 * - No ML, no preprocessing, no thresholds applied here
 * - Uses env var for API base URL
 */
export default function Home() {
  const [text, setText] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const MIN_WORDS = 150;
  const RECOMMENDED_MIN = 200;
  const RECOMMENDED_MAX = 300;

  // Read API base from env (support common toolchains)
  const apiBase = import.meta.env.VITE_API_BASE_URL;


  // Live word count (UI-only): trim + split on whitespace
  const wordCount = useMemo(() => {
    if (!text) return 0;
    const t = text.trim();
    if (t === "") return 0;
    return t.split(/\s+/).length;
  }, [text]);

  const disabled = loading || wordCount < MIN_WORDS || !apiBase;

  async function handleSubmit(e?: React.FormEvent) {
    e?.preventDefault();
    setError(null);

    if (!apiBase) {
      setError("API base URL not configured in environment variables.");
      return;
    }
    if (wordCount < MIN_WORDS) {
      setError(`Please enter at least ${MIN_WORDS} words before submitting.`);
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${apiBase.replace(/\/$/, "")}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        const payload = await res.text();
        throw new Error(`API error ${res.status}: ${payload}`);
      }
      const data: MBTIResponse = await res.json();
      console.log("✅ Prediction response received:", data.mbti);
      console.log("📝 Saving text to storage, length:", text.length);
      // Save to storage for counterfactual analysis
      try {
        localStorage.setItem("mbti_input_text", text);
        sessionStorage.setItem("mbti_input_text", text);
        localStorage.setItem("last_mbti_result", JSON.stringify({
          response: data,
          text: text,
          timestamp: new Date().toISOString()
        }));
        console.log("✅ Text saved successfully");
      } catch (storageError) {
        console.warn("⚠️ Storage failed:", storageError);
      }
      console.log("🧭 Navigating to /result");
      // Navigate to a result page, pass the response AND text as state
      navigate("/result", { state: { response: data, text: text } });
    } catch (err: any) {
      setError(err?.message ?? "Unknown error calling API");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={containerStyle}>
      <h1 style={titleStyle}>Describe Yourself</h1>

      <p>
        Write a few sentences describing your thinking style, decision-making,
        and typical behavior. Use full sentences. Minimum {MIN_WORDS} words.
      </p>

      <form onSubmit={handleSubmit} style={formStyle}>
        <TextInput
          value={text}
          onChange={(t) => setText(t)}
          placeholder="Write 200-300 words for best results..."
          rows={10}
        />

        <div style={metaRowStyle}>
          <WordCounter text={text} minWords={MIN_WORDS} />
          <div style={guidanceStyle}>
            Recommended: {RECOMMENDED_MIN}–{RECOMMENDED_MAX} words for best
            reliability.
          </div>
        </div>

        {error && <div style={errorStyle}>{error}</div>}

        <div style={actionsStyle}>
          <button type="submit" disabled={disabled} style={buttonStyle}>
            {loading ? "Predicting..." : "Submit"}
          </button>
        </div>
      </form>
    </div>
  );
}

/* Minimal inline styles */
const containerStyle: React.CSSProperties = {
  maxWidth: 800,
  margin: "32px auto",
  fontFamily: "Arial, Helvetica, sans-serif",
  padding: 16,
};
const titleStyle: React.CSSProperties = { marginBottom: 8 };
const formStyle: React.CSSProperties = { display: "flex", flexDirection: "column", gap: 12 };
const metaRowStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
};
const guidanceStyle: React.CSSProperties = { fontSize: 13, color: "#555" };
const errorStyle: React.CSSProperties = { color: "#a00", fontSize: 13 };
const actionsStyle: React.CSSProperties = { display: "flex", justifyContent: "flex-end" };
const buttonStyle: React.CSSProperties = {
  padding: "10px 16px",
  fontSize: 15,
  borderRadius: 6,
  border: "none",
  background: "#2d6cdf",
  color: "#fff",
  cursor: "pointer",
  opacity: 1,
};
// import { useNavigate } from "react-router-dom";

// export default function Home() {
//   const navigate = useNavigate();

//   return (
//     <div style={{ padding: "40px" }}>
//       <h1>MBTI Neuro-Causal System</h1>
//       <p>Frontend is working ✅</p>

//       <button onClick={() => navigate("/input")}>
//         Go to Input
//       </button>
//     </div>
//   );
// }
