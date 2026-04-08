
type Props = {
  deltas: Record<string, number>;
};

export default function DeltaProbabilityBars({ deltas }: Props) {
  return (
    <div>
      {Object.entries(deltas).map(([dim, delta]) => (
        <div key={dim} style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 13 }}>
            {dim}: Δ {delta.toFixed(3)}
          </div>
          <div
            style={{
              height: 8,
              background: "#eee",
              position: "relative",
            }}
          >
            <div
              style={{
                position: "absolute",
                left: "50%",
                width: `${Math.abs(delta) * 50}%`,
                height: "100%",
                background: delta >= 0 ? "#2d6cdf" : "#d9534f",
                transform: delta >= 0 ? "translateX(0)" : "translateX(-100%)",
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
