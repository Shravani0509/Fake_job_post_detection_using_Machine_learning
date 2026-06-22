import re
from typing import Dict, List, Optional, Tuple


def extract_salary(salary_text: str) -> Optional[str]:
    """Best-effort salary extractor.

    Keeps the original text form for display; returns None if empty.
    """
    if not salary_text:
        return None
    cleaned = salary_text.strip()
    return cleaned if cleaned else None


def extract_company_name(job_text: str) -> Optional[str]:
    """Heuristic company name extraction.

    Looks for common patterns like:
    - "Company:" / "Company -" / "Employer:"
    - first line if it looks like a company name
    """
    if not job_text:
        return None

    patterns = [
        r"company\s*[:\-]\s*(?P<name>[A-Za-z0-9&\-\s]{2,80})",
        r"employer\s*[:\-]\s*(?P<name>[A-Za-z0-9&\-\s]{2,80})",
    ]

    for pat in patterns:
        m = re.search(pat, job_text, flags=re.IGNORECASE)
        if m:
            name = m.group("name").strip()
            return name[:80] if name else None

    # Fallback: use first non-empty line
    lines = [ln.strip() for ln in job_text.splitlines() if ln.strip()]
    if not lines:
        return None
    first = lines[0]
    # If first line contains lots of words, skip it
    if len(first.split()) > 12:
        return None
    # If first line contains typical job keywords, skip it
    if re.search(r"(hiring|job|position|role|salary|apply)", first, flags=re.IGNORECASE):
        return None
    return first[:80]


def extract_location(job_text: str, fallback_location: Optional[str] = None) -> Optional[str]:
    """Extract location from job text.

    Uses simple patterns; falls back to user-provided location.
    """
    if fallback_location and fallback_location.strip():
        return fallback_location.strip()
    if not job_text:
        return None

    # Very lightweight heuristic
    m = re.search(r"location\s*[:\-]\s*(?P<loc>[A-Za-z\s,\.]{2,80})", job_text, flags=re.IGNORECASE)
    if m:
        loc = m.group("loc").strip()
        return loc[:80] if loc else None

    return None


def extract_requirements(job_text: str, max_items: int = 8) -> List[str]:
    """Extract likely requirements.

    Pulls bullet/numbered lines or sentences containing "must/have/required".
    """
    if not job_text:
        return []

    lines = [ln.strip() for ln in job_text.splitlines() if ln.strip()]

    reqs: List[str] = []

    # Bullet/numbered lines
    for ln in lines:
        if re.match(r"^([\-*•]|\d+[\.)])\s+", ln):
            reqs.append(re.sub(r"^([\-*•]|\d+[\.)])\s+", "", ln).strip())

    # Sentence-level patterns
    if len(reqs) < max_items:
        sentences = re.split(r"(?<=[\.!?])\s+", job_text)
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            if re.search(r"(required|must|should|preferred|experience|skills)", s_clean, flags=re.IGNORECASE):
                if s_clean not in reqs:
                    reqs.append(s_clean)

    # Trim
    reqs = [r for r in reqs if r]
    return reqs[:max_items]


def extract_job_description_preview(job_text: str, max_chars: int = 500) -> str:
    if not job_text:
        return ""
    s = job_text.strip().replace("\n", " ")
    return s[:max_chars]


def highlight_matches(job_text: str, phrases: List[str]) -> List[str]:
    """Return matched phrases/keywords present in the text."""
    if not job_text:
        return []
    lower = job_text.lower()
    found = []
    for ph in phrases:
        if ph.lower() in lower and ph not in found:
            found.append(ph)
    return found


def parse_free_text_fields(job_text: str) -> Dict[str, Optional[str]]:
    """Extracts optional fields from free text (best-effort)."""
    return {
        "company": extract_company_name(job_text),
        "location": extract_location(job_text),
        "salary": None,
    }

