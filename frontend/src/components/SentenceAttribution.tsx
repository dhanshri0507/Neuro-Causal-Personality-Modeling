// ================================
// FILE STATUS: FROZEN
// Verified working with backend
// Do NOT modify unless API changes
// ================================

import React from "react";

type Attribution = {
  sentence: string;
  weight: number;
};

type Props = {
  attributions?: Attribution[] | null;
};

/**
 * SentenceAttribution
 *
 * - Displays sentence-level attribution provided by backend.
 * - Sorts entries by weight (descending) only for display purposes.
 * - Does not compute or modify weights.
 */
export default function SentenceAttribution({ attributions }: Props) {
  if (!attributions || attributions.length === 0) {
    return <div style={styles.empty}>No sentence attribution data available.</div>;
  }

  // Display-only sorting: do not mutate original input
  const sorted = [...attributions].sort((a, b) => b.weight - a.weight);

  return (
    <div style={styles.container}>
      <ol style={styles.list}>
        {sorted.map((item, idx) => (
          <li key={idx} style={styles.item}>
            <div style={styles.sentence}>{item.sentence}</div>
            <div style={styles.weight}>{item.weight.toFixed(3)}</div>
          </li>
        ))}
      </ol>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    width: "100%",
  },
  empty: {
    fontStyle: "italic",
    color: "#666",
  },
  list: {
    listStyle: "decimal",
    paddingLeft: 20,
    margin: 0,
  },
  item: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: 12,
    padding: "8px 0",
    borderBottom: "1px solid #eee",
  },
  sentence: {
    flex: 1,
    marginRight: 12,
    whiteSpace: "pre-wrap",
    lineHeight: 1.4,
  },
  weight: {
    width: 90,
    textAlign: "right",
    fontFamily: "monospace",
    color: "#333",
  },
};
