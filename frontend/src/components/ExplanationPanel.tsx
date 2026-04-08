// ================================
// FILE STATUS: FROZEN
// Verified working with backend
// Do NOT modify unless API changes
// ================================

import React from "react";

type Props = {
  explanation?: string | null;
};

/**
 * ExplanationPanel
 *
 * - Stateless presentational component
 * - Renders backend-provided explanation text ONLY
 * - Preserves line breaks and paragraphs
 * - Does not perform logic or orchestration
 */
export default function ExplanationPanel({ explanation }: Props) {
  if (!explanation) {
    return <div style={styles.empty}>No explanation provided.</div>;
  }

  return (
    <div style={styles.container}>
      <div style={styles.explanation}>{explanation}</div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: 12,
    borderRadius: 8,
    border: "1px solid #eee",
    background: "#ffffff",
  },
  explanation: {
    whiteSpace: "pre-wrap",
    lineHeight: 1.5,
    color: "#222",
    fontSize: 14,
  },
  empty: {
    fontStyle: "italic",
    color: "#666",
  },
};
