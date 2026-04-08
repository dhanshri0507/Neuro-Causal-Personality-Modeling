# ================================
# FILE STATUS: FROZEN
# Do NOT modify unless API changes
# ================================

"""
Pydantic request and response schemas for the MBTI Neuro-Causal API.

This module defines plain data contracts (no logic) used by the API layer.
Each schema is a Pydantic BaseModel and documents the expected fields.
"""
from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Optional, Union


class MBTIResponse(BaseModel):
    mbti: str
    confidence: float
    probabilities: Dict[str, float]
    explanation: str
    cognitive_features: Optional[Dict[str, float]] = None
    sentence_attribution: Optional[List[Dict[str, Union[str, float]]]] = None
    

    model_config = ConfigDict(arbitrary_types_allowed=True)

class TextInput(BaseModel):
    """Request schema for user-provided text input.

    Fields:
    - text: Raw user-provided personality description (string).

    Notes:
    - This schema performs only basic type validation. Preprocessing is handled
      elsewhere in the Phase-1 pipeline.
    """

    text: str
    intervention_feature: Optional[str] = None
    intervention_lambda: Optional[float] = None


class CounterfactualResponse(BaseModel):
    """Response schema for counterfactual prediction outputs.

    Fields:
    - factual_type: predicted MBTI type under factual inputs
    - counterfactual_type: predicted MBTI type under the intervention
    - factual_probabilities: probabilities per dimension under factual inputs
    - counterfactual_probabilities: probabilities per dimension under intervention
    """
    
    factual_type: str
    counterfactual_type: str
    factual_probabilities: Dict[str, float]
    counterfactual_probabilities: Dict[str, float]
    delta_probabilities: Optional[Dict[str, float]] = None
    sensitivity: Optional[Dict[str, float]] = None
    trait_flip: Optional[Dict[str, bool]] = None
    counterfactual_explanation: Optional[str] = None
    
    
    
