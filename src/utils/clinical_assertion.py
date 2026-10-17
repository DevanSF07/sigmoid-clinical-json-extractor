"""
Clinical Assertion and Context Detection Layer
Detects assertion status for clinical entity mentions:
- AFFIRMED: Current active symptom or finding (e.g. "complains of fever")
- NEGATED: Explicitly denied or absent (e.g. "denies chest pain", "no fever")
- HISTORICAL: Prior medical condition (e.g. "history of asthma 10 years ago")
- UNCERTAIN: Suspected or differential diagnosis (e.g. "possible pneumonia", "rule out TB")
"""

import re
from enum import Enum
from typing import Tuple, Dict, Any, List


class AssertionStatus(str, Enum):
    AFFIRMED = "AFFIRMED"
    NEGATED = "NEGATED"
    HISTORICAL = "HISTORICAL"
    UNCERTAIN = "UNCERTAIN"


# Pre-negation triggers (appear before the entity)
PRE_NEGATION_TRIGGERS = [
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bdenying\b",
    r"\bno\b",
    r"\bnot\b",
    r"\bnegative for\b",
    r"\bno evidence of\b",
    r"\bno signs of\b",
    r"\bno report of\b",
    r"\bno history of\b",
    r"\bfree of\b",
    r"\bwithout\b",
    r"\bruled out\b",
    r"\bnever had\b",
    r"\bunremarkable for\b"
]

# Post-negation triggers (appear immediately after entity)
POST_NEGATION_TRIGGERS = [
    r"\bwas ruled out\b",
    r"\bnot present\b",
    r"\babsent\b",
    r"\bresolved\b"
]

# Historical triggers
HISTORICAL_TRIGGERS = [
    r"\bhistory of\b",
    r"\bpast medical history\b",
    r"\bpmh\b",
    r"\bpmhx\b",
    r"\bdiagnosed approximately\b",
    r"\bdiagnosed \d+ years ago\b",
    r"\bdiagnosed in \d{4}\b",
    r"\bchildhood history of\b",
    r"\bprior episode\b",
    r"\bprevious history\b",
    r"\blongstanding history\b",
    r"\bwell-controlled on medication\b"
]

# Uncertainty / Hypothetical triggers
UNCERTAIN_TRIGGERS = [
    r"\bpossible\b",
    r"\bpotential\b",
    r"\bsuspected\b",
    r"\brule out\b",
    r"\br/o\b",
    r"\bversus\b",
    r"\bvs\b",
    r"\bevaluate for\b",
    r"\bconcern for\b",
    r"\bquestionable\b",
    r"\bmay indicate\b",
    r"\bconsistent with possible\b"
]

# Scope termination delimiters (stop scope of negation/history)
SCOPE_TERMINATORS = [
    r"\.",
    r";",
    r"\bbut\b",
    r"\bhowever\b",
    r"\byet\b",
    r"\balso\b",
    r"\balthough\b",
    r"\bpresented with\b",
    r"\breports\b",
    r"\bcomplains of\b"
]


def detect_assertion(
    text: str,
    entity_start: int,
    entity_end: int,
    window_chars: int = 100
) -> Tuple[AssertionStatus, str]:
    """
    Determine the assertion status of an entity mention by analyzing the
    surrounding window of text.
    Returns: (AssertionStatus, matched_trigger_phrase)
    """
    # Extract left context window (before entity)
    left_start = max(0, entity_start - window_chars)
    left_text = text[left_start:entity_start].lower()

    # If scope terminators occur in left context, restrict to text after the last terminator
    last_term_idx = -1
    for term in SCOPE_TERMINATORS:
        for m in re.finditer(term, left_text):
            if m.end() > last_term_idx:
                last_term_idx = m.end()
    if last_term_idx != -1:
        left_text = left_text[last_term_idx:]

    # Extract right context window (after entity)
    right_end = min(len(text), entity_end + window_chars)
    right_text = text[entity_end:right_end].lower()
    # Restrict right context to first sentence terminator
    first_term = len(right_text)
    for term in [r"\.", r";", r"\n"]:
        m = re.search(term, right_text)
        if m and m.start() < first_term:
            first_term = m.start()
    right_text = right_text[:first_term]

    # Check Negation
    for trig in PRE_NEGATION_TRIGGERS:
        m = re.search(trig, left_text)
        if m:
            return AssertionStatus.NEGATED, m.group(0)

    for trig in POST_NEGATION_TRIGGERS:
        m = re.search(trig, right_text)
        if m:
            return AssertionStatus.NEGATED, m.group(0)

    # Check Historical mentions
    for trig in HISTORICAL_TRIGGERS:
        m = re.search(trig, left_text)
        if m:
            return AssertionStatus.HISTORICAL, m.group(0)

    # Check Uncertainty
    for trig in UNCERTAIN_TRIGGERS:
        m = re.search(trig, left_text)
        if m:
            return AssertionStatus.UNCERTAIN, m.group(0)

    return AssertionStatus.AFFIRMED, "none"
