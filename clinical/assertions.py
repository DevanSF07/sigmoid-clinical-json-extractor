"""Conservative local assertion heuristics, not a clinically validated classifier."""
import re

def assertion_status(text: str, start: int, end: int) -> str:
    # Scope resets at sentence/section boundaries and contrast conjunctions.
    prefix = re.split(r"[.!?;\n]|\b(?:but|however)\b", text[:start].lower())[-1][-150:]
    suffix = re.split(r"[.!?;\n]", text[end:].lower())[0][:60]
    if re.search(r"\b(?:no|denies|denied|without|negative for|absence of)\b", prefix):
        return "negated"
    if re.search(r"\b(?:family history|history of|previous|previously|resolved)\b", prefix):
        return "historical"
    if re.search(r"\b(?:possible|possibly|suspected|rule out|may have|concern for)\b", prefix):
        return "uncertain"
    if re.match(r"\s+(?:is |was )?(?:absent|denied|resolved)\b", suffix):
        return "negated"
    return "present"
