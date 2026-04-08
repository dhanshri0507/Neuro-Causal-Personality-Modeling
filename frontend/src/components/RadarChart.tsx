// ================================
// FILE STATUS: FROZEN
// Verified working with backend
// Do NOT modify unless API changes
// ================================

import {
  RadarChart as ReRadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
} from "recharts";

type Props = {
  traits: Record<string, number>;
};

/**
 * RadarChart component
 *
 * - Uses a fixed axis ordering for reproducibility.
 * - Expects `traits` values already normalized by the backend (no scaling).
 */
export default function RadarChart({ traits }: Props) {
  // FIXED ordering of axes (must match backend's convention if applicable)
  const AXIS_ORDER = [
    "pronoun_ratio",
    "modality_score",
    "negation_count",
    "emotion_intensity",
    "reasoning_prop",
    "planning_prop",
    "uncertainty_prop",
    "lexical_diversity",
  ];

  const data = AXIS_ORDER.map((key) => ({
    trait: key,
    value: traits?.[key] ?? 0,
  }));

  return (
    <div style={{ width: "100%", height: 360 }}>
      <ResponsiveContainer>
        <ReRadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid />
          <PolarAngleAxis dataKey="trait" />
          <PolarRadiusAxis angle={30} domain={[0, 1]} />
          <Radar name="document" dataKey="value" stroke="#8884d8" fill="#8884d8" fillOpacity={0.4} />
        </ReRadarChart>
      </ResponsiveContainer>
    </div>
  );
}