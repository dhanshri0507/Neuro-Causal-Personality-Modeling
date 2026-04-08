// ================================
// FILE STATUS: FROZEN
// Verified working with backend
// Do NOT modify unless API changes
// ================================
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LabelList,
} from "recharts";

type Props = {
  probabilities: Record<string, number>;
};

/**
 * ProbabilityBars
 *
 * Renders horizontal bars for MBTI dimension probabilities using Recharts.
 * - Expects `probabilities` to contain values in [0,1].
 * - Renders dimensions in the canonical order IE, NS, TF, JP if present in the input.
 * - Purely presentational: does not modify or compute probabilities.
 */
export default function ProbabilityBars({ probabilities }: Props) {
  const DIM_ORDER = ["IE", "NS", "TF", "JP"];

  // Build data array in canonical order but only include dimensions present
  const data = DIM_ORDER.filter((d) => d in probabilities).map((d) => ({
    name: d,
    value: Number(probabilities[d]),
  }));

  return (
    <div style={{ width: "100%", height: 240 }}>
      <ResponsiveContainer>
        <BarChart
          data={data}
          layout="vertical" // horizontal bars
          margin={{ top: 20, right: 24, left: 40, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => v.toFixed(2)}
            allowDecimals={true}
          />
          <YAxis dataKey="name" type="category" width={60} />
          <Tooltip formatter={(value) => {
            if (typeof value === "number") {
              return value.toFixed(3);
            }
            return value;
            }}
            />

          <Bar dataKey="value" fill="#4c72b0" barSize={18}>
            <LabelList dataKey="value" position="right" formatter={(value) => {
              if (typeof value === "number") {
                return value.toFixed(2);
              }
              return "";
              }}
              />

          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}