// Single source of truth for counterfactual-intervenable features
export const COUNTERFACTUAL_FEATURES = [
    { key: "pronoun_ratio", label: "Pronoun Usage" },
    { key: "modality_score", label: "Modality / Certainty" },
    { key: "negation_count", label: "Negation Frequency" },
    { key: "emotion_intensity", label: "Emotional Intensity" },
    { key: "lexical_diversity", label: "Lexical Diversity" },
    { key: "reasoning", label: "Reasoning Markers" },
    { key: "planning", label: "Planning Markers" },
    { key: "uncertainty", label: "Uncertainty Markers" },
] as const; 