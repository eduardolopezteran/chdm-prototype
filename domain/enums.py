"""
CHDM v0.1 — Canonical governed enums.

Every enum here corresponds directly to a governed vocabulary defined in
CHDM v0.1 (Frozen Governing Methodology Baseline). Do not add values that
are not present in the frozen baseline; do not remove values that are.

Source section references are given per enum.
"""

from enum import Enum


class Lifecycle(str, Enum):
    """CHDM v0.1 §2 — base lifecycle states."""
    L1 = "L1"  # Implementation / Migration
    L2 = "L2"  # Initial Value
    L3 = "L3"  # Adoption / Institutionalization
    L4 = "L4"  # Mature / Optimization


class Overlay(str, Enum):
    """CHDM v0.1 §2 — independent Yes/No overlays."""
    RECOVERY = "RECOVERY"
    COMMERCIAL_DECISION_ACTIVE = "COMMERCIAL_DECISION_ACTIVE"


class EvidenceState(str, Enum):
    """CHDM v0.1 §3.2 — governed evidence-quality vocabulary (state axis)."""
    CURRENT_CONFIRMED = "CURRENT_CONFIRMED"
    CURRENT_UNVERIFIED = "CURRENT_UNVERIFIED"
    STALE = "STALE"
    CONTRADICTORY = "CONTRADICTORY"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class SignalDirection(str, Enum):
    """CHDM v0.1 §3.2 — signal direction axis (orthogonal to EvidenceState)."""
    FAVORABLE = "FAVORABLE"
    CONCERNING = "CONCERNING"
    MIXED = "MIXED"
    NON_DIRECTIONAL = "NON_DIRECTIONAL"


class Provenance(str, Enum):
    """CHDM v0.1 §3.1 — Evidence Object provenance."""
    USER_PROVIDED = "USER_PROVIDED"
    AI_EXTRACTED = "AI_EXTRACTED"
    HUMAN_CONFIRMED = "HUMAN_CONFIRMED"
    DETERMINISTIC = "DETERMINISTIC"


class DimensionCode(str, Enum):
    """CHDM v0.1 §4 — canonical dimension codes. D8 is a grouping only and
    is never itself a DimensionCode with a state (D8A/D8B are the real
    governed dimensions)."""
    D1 = "D1"    # Value Realization
    D2 = "D2"    # Product Adoption
    D3 = "D3"    # Customer Engagement
    D4 = "D4"    # Experience & Sentiment
    D5 = "D5"    # Support & Service Performance
    D6 = "D6"    # Relationship Health
    D7 = "D7"    # Commercial Health
    D8A = "D8A"  # Strategic Alignment
    D8B = "D8B"  # Organizational Risk


class DimensionStateValue(str, Enum):
    """CHDM v0.1 §4 (MD-03) — common dimension-state vocabulary. Applies
    identically to D1-D7 and each D8 subcomponent."""
    SUPPORTED = "SUPPORTED"
    MIXED = "MIXED"
    CONCERNING = "CONCERNING"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RequirementClass(str, Enum):
    """CHDM v0.1 §7 — dimension applicability / requirement model."""
    UR = "UR"  # Universally Required
    LR = "LR"  # Lifecycle Required
    CR = "CR"  # Conditionally Required (promoted from S)
    S = "S"    # Supporting
    NA = "NA"  # Not Applicable


class ObjectiveOutcomeState(str, Enum):
    """CHDM v0.1 §5.1 — Objective Outcome State. NOT ordinal."""
    UNKNOWN = "UNKNOWN"
    NOT_YET_EXPECTED = "NOT_YET_EXPECTED"
    PROGRESSING = "PROGRESSING"
    ACHIEVED = "ACHIEVED"
    NOT_ACHIEVED = "NOT_ACHIEVED"
    DISPUTED = "DISPUTED"


class ValueEvidenceBasis(str, Enum):
    """CHDM v0.1 §5.2 — multi-valued; no ordinal hierarchy."""
    PROXY_SUPPORTED = "PROXY_SUPPORTED"
    MEASURED_OPERATIONAL_EVIDENCE = "MEASURED_OPERATIONAL_EVIDENCE"
    CUSTOMER_CONFIRMED = "CUSTOMER_CONFIRMED"
    INDEPENDENTLY_VERIFIED = "INDEPENDENTLY_VERIFIED"
    UNVERIFIED_CLAIM = "UNVERIFIED_CLAIM"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class RiskMechanismCode(str, Enum):
    """CHDM v0.1 §6 — CR-01 through CR-08."""
    CR_01 = "CR-01"  # Sponsor / Champion Continuity
    CR_02 = "CR-02"  # Service Failure
    CR_03 = "CR-03"  # Commercial Continuity
    CR_04 = "CR-04"  # Strategic Displacement
    CR_05 = "CR-05"  # Organizational Continuity
    CR_06 = "CR-06"  # Core Automation / Integration Failure
    CR_07 = "CR-07"  # Implementation Failure
    CR_08 = "CR-08"  # Value Failure / Rejection


class RiskSeverity(str, Enum):
    """CHDM v0.1 §6.1 — progressive severity. A risk record's
    potential_severity and activated_severity are each Optional[RiskSeverity]
    (None permitted; RESOLVED is a distinct lifecycle state, not a severity
    level held concurrently with Watch/Material/Critical)."""
    WATCH = "WATCH"
    MATERIAL = "MATERIAL"
    CRITICAL = "CRITICAL"
    RESOLVED = "RESOLVED"


class AssessmentReliabilityLevel(str, Enum):
    """CHDM v0.1 §9."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class OperationalPriority(str, Enum):
    """CHDM v0.1 §10. OP1 > OP2 > OP3 is the only ordinal relationship;
    OPU is deliberately outside that ordering (never treat OPU as a 4th
    ordinal rung — INV-19/§10 handoff note)."""
    OP1 = "OP1"  # Urgent Review
    OP2 = "OP2"  # Review Required
    OP3 = "OP3"  # Routine Monitoring
    OPU = "OPU"  # Undetermined


class EvidenceReviewStatus(str, Enum):
    """CHDM v0.1 §11."""
    ER1 = "ER1"  # Evidence Review Required
    ER0 = "ER0"  # Evidence Review Not Required


class ConfirmationAction(str, Enum):
    """CHDM v0.1 §3.3 confirmation boundary; state-transition actions a
    human may take on an extracted/unverified observation."""
    CONFIRM = "CONFIRM"
    CORRECT = "CORRECT"
    REJECT = "REJECT"
    CANNOT_CONFIRM = "CANNOT_CONFIRM"


class MethodologyObjectType(str, Enum):
    """CHDM v0.1 §14.2 — canonical object types."""
    DIM = "DIM"
    OBJ = "OBJ"
    EVID = "EVID"
    RISK = "RISK"
    RULE = "RULE"
    LIFE = "LIFE"
    STATE = "STATE"
    SAFE = "SAFE"
    VAL = "VAL"
    LIMIT = "LIMIT"


class DMEGLinkedConclusion(str, Enum):
    """CHDM v0.1 §8 DMEG-2 — the four governed-conclusion categories a
    DMEG may be linked to (registry/dmeg_rules.yaml `linked_conclusions`).
    A single DMEG may be linked to more than one category; membership is
    determined by which conclusion(s) actually differ under the DMEG-3
    differential test — never assumed from DMEG existence alone. This is
    the refinement distinguishing a DMEG in general from a
    priority-elevating DMEG specifically (a DMEG is not automatically
    OPERATIONAL_PRIORITY-linked just because it exists)."""
    OBJECTIVE_D1 = "OBJECTIVE_D1"                  # objective_outcome_or_D1
    RISK_SEVERITY = "RISK_SEVERITY"                 # activation/severity of a Material/Critical risk
    DIMENSION_STATE = "DIMENSION_STATE"              # a material dimension conclusion (D2-D8B)
    OPERATIONAL_PRIORITY = "OPERATIONAL_PRIORITY"     # Operational Priority itself


class AuthorityClassification(str, Enum):
    """CHDM v0.1 §14.3 — object property `authority_classification`."""
    DETERMINISTIC = "DETERMINISTIC"
    AI_SUPPORTED = "AI_SUPPORTED"
    HUMAN_CONFIRMED = "HUMAN_CONFIRMED"
    HUMAN_ONLY = "HUMAN_ONLY"
