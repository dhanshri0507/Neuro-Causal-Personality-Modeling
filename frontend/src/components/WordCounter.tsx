// ================================
// FILE STATUS: FROZEN
// Verified working with backend
// Do NOT modify unless API changes
// ================================

type Props = {
  text: string;
  minWords: number;
};

/**
 * WordCounter
 *
 * - Stateless component that displays the live word count for `text`.
 * - Uses a robust whitespace split: trim + split on one-or-more whitespace.
 * - Shows the current count, the minimum required, and a simple text indicator.
 *
 * Notes:
 * - No side effects, no internal state, no API calls.
 */
export default function WordCounter({ text, minWords }: Props) {
  const count = (() => {
    if (!text) return 0;
    const t = text.trim();
    if (t === "") return 0;
    return t.split(/\s+/).length;
  })();

  const meets = count >= minWords;

  return (
    <div style={{ fontSize: 13, color: "#333" }}>
      <div>Word count: {count}</div>
      <div>Minimum required: {minWords}</div>
      <div style={{ marginTop: 6, fontWeight: 600 }}>
        {meets ? "✓ Minimum met" : "✗ Below minimum"}
      </div>
    </div>
  );
}