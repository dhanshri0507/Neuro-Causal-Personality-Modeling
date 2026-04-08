// ================================
// FILE STATUS: FROZEN
// Verified working with backend
// Do NOT modify unless API changes
// ================================

import React, { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import TextInput from "../components/TextInput";
import WordCounter from "../components/WordCounter";

type PredictResponse = {
    mbti: string;
    confidence: number;
    probabilities: Record<string, number>;
    explanation: string;
};

const MIN_WORDS = 150;
const RECOMMENDED_MIN = 200;
const RECOMMENDED_MAX = 300;

/**
 * InputPage
 *
 * - Collects a multi-sentence personality description
 * - Enforces UI-level word-count constraints (min 150)
 * - Calls backend POST /predict and navigates to Result page with response
 *
 * Frontend constraints:
 * - No preprocessing or inference here; backend is authoritative
 * - API base URL configured via environment variable
 */
export default function InputPage() {
    const [text, setText] = useState<string>("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const navigate = useNavigate();

    const apiBase = import.meta.env.VITE_API_BASE_URL || "";

    const wordCount = useMemo(() => {
        if (!text) return 0;
        // simple whitespace split for UI-only counting
        return text.trim().split(/\s+/).filter(Boolean).length;
    }, [text]);

    const isSubmitDisabled = loading || wordCount < MIN_WORDS || !apiBase;

    async function handleSubmit(e?: React.FormEvent) {
        e?.preventDefault();
        setError(null);

        if (wordCount < MIN_WORDS) {
            setError(`Please provide at least ${MIN_WORDS} words.`);
            return;
        }
        if (!apiBase) {
            setError("API base URL not configured.");
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
                const txt = await res.text();
                throw new Error(`API error ${res.status}: ${txt}`);
            }

            const data: PredictResponse = await res.json();

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

            // Navigate with the text in state
            navigate("/result", {
                state: {
                    response: data,
                    text: text,
                },
            });

        } catch (err: any) {
            setError(err?.message ?? "Unknown error while calling API.");
        } finally {
            setLoading(false);
        }
    }


    return (
        <div style={styles.container}>
            <h1 style={styles.title}>MBTI Personality Description</h1>

            <form onSubmit={handleSubmit} style={styles.form}>
                <label htmlFor="text" style={styles.label}>
                    Tell us about yourself (use full sentences). Minimum {MIN_WORDS} words required.
                </label>

                <TextInput
                    id="text"
                    value={text}
                    onChange={(v: string) => setText(v)}
                    placeholder="Write a paragraph describing yourself..."
                    rows={8}
                    style={styles.textarea}
                />

                <div style={styles.row}>
                    <WordCounter text={text} minWords={MIN_WORDS} />
                    <div style={styles.guidance}>
                        <div>Recommended: {RECOMMENDED_MIN}-{RECOMMENDED_MAX} words for best results.</div>
                        {wordCount < MIN_WORDS ? (
                            <div style={styles.warn}>Minimum {MIN_WORDS} words required.</div>
                        ) : null}
                    </div>
                </div>

                {error ? <div style={styles.error}>{error}</div> : null}

                <div style={styles.actions}>
                    <button
                        type="submit"
                        disabled={isSubmitDisabled}
                        style={{
                            ...styles.button,
                            opacity: isSubmitDisabled ? 0.6 : 1,
                            cursor: isSubmitDisabled ? "not-allowed" : "pointer",
                        }}
                    >
                        {loading ? "Predicting..." : "Submit for Prediction"}
                    </button>
                </div>
            </form>
        </div>
    );
}

/* Basic inline styles (kept small and replaceable with CSS modules) */
/* Professional Academic Styles using CSS Variables */
const styles: Record<string, React.CSSProperties> = {
    container: {
        maxWidth: 800,
        margin: "60px auto",
        padding: "40px",
        fontFamily: "var(--font-family)",
        backgroundColor: "var(--color-surface)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-lg)",
        border: "1px solid var(--color-border)",
    },
    title: {
        marginBottom: 32,
        fontSize: "2rem",
        color: "var(--color-primary)",
        textAlign: "center",
        fontWeight: 700,
        letterSpacing: "-0.01em",
    },
    form: {
        display: "flex",
        flexDirection: "column",
        gap: 24
    },
    label: {
        fontSize: "1rem",
        fontWeight: 600,
        color: "var(--color-text-main)",
        marginBottom: 8,
        display: "block"
    },
    textarea: {
        width: "100%",
        minHeight: 200,
        padding: 16,
        fontSize: "1rem",
        borderRadius: "var(--radius-md)",
        border: "1px solid var(--color-border)",
        resize: "vertical",
        fontFamily: "inherit",
        lineHeight: 1.6,
        backgroundColor: "var(--color-bg)",
        color: "var(--color-text-main)",
        outline: "none",
        transition: "border-color 0.2s, box-shadow 0.2s",
        boxSizing: "border-box", // Ensure padding doesn't overflow width
    },
    row: {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center", // Align to top if text wraps
        gap: 16,
        flexWrap: "wrap",
        marginTop: 4
    },
    guidance: {
        fontSize: "0.875rem",
        color: "var(--color-text-secondary)",
        textAlign: "right",
        flex: 1, // Allow it to take space
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-end",
    },
    warn: {
        color: "var(--color-error)",
        fontWeight: 600,
        marginTop: 4
    },
    error: {
        padding: "16px",
        background: "#fef2f2",
        color: "var(--color-error)",
        fontSize: "0.9rem",
        borderRadius: "var(--radius-md)",
        border: "1px solid #fee2e2",
        display: "flex",
        alignItems: "center",
        gap: "8px"
    },
    actions: {
        display: "flex",
        justifyContent: "flex-end",
        marginTop: 24
    },
    button: {
        padding: "12px 32px",
        fontSize: "1rem",
        fontWeight: 600,
        borderRadius: "var(--radius-md)",
        border: "none",
        backgroundColor: "var(--color-accent)",
        color: "white",
        cursor: "pointer",
        transition: "background-color 0.2s, transform 0.1s",
        boxShadow: "var(--shadow-sm)",
    },
};