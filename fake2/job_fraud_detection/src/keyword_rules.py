from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class KeywordRule:
    key: str
    phrases: List[str]
    message: str
    weight: float


DEFAULT_RULES: List[KeywordRule] = [
    KeywordRule(
        key="personal_id",
        phrases=[
            "aadhaar",
            "pan number",
            "passport number",
            "social security number",
            "ssn",
        ],
        message="The posting asks for personal identification numbers, which is a common fraud indicator.",
        weight=3.0,
    ),
    KeywordRule(
        key="unrealistic_salary",
        phrases=[
            "easy money",
            "high salary",
            "unlimited income",
            "work from home and earn",
            "no experience required",
            "get paid daily",
        ],
        message="The posting mentions unrealistic salary/income promises, which is suspicious.",
        weight=2.5,
    ),
    KeywordRule(
        key="payment_request",
        phrases=[
            "payment required",
            "pay to join",
            "registration fee",
            "processing fee",
            "deposit money",
            "pay registration fee",
            "transfer fee",
        ],
        message="The posting requests payment/fees, which is a common fraud tactic.",
        weight=3.5,
    ),
    KeywordRule(
        key="urgent_language",
        phrases=[
            "urgent hiring",
            "immediate join",
            "act fast",
            "limited time offer",
            "urgent opening",
        ],
        message="The posting uses urgent or pushy language, which can indicate fraud.",
        weight=1.8,
    ),
    KeywordRule(
        key="bank_account",
        phrases=[
            "bank account",
            "account number",
            "bank details",
            "account details",
            "bank info",
            "banking information",
        ],
        message="The posting asks for bank account details, which is a common indicator of fraudulent listings.",
        weight=4.0,
    ),
]


def find_suspicious_phrases(job_text: str, rules: List[KeywordRule] = DEFAULT_RULES) -> List[Dict[str, object]]:
    """Return a list of triggered rules with matched phrases."""
    if not job_text:
        return []

    lower = job_text.lower()
    triggered: List[Dict[str, object]] = []

    for rule in rules:
        matches = [p for p in rule.phrases if p.lower() in lower]
        if matches:
            triggered.append(
                {
                    "rule_key": rule.key,
                    "message": rule.message,
                    "matched_phrases": matches,
                    "weight": rule.weight,
                }
            )

    return triggered


def build_keyword_contributions(job_text: str, rules: List[KeywordRule] = DEFAULT_RULES) -> Tuple[float, List[Dict[str, object]]]:
    triggered = find_suspicious_phrases(job_text, rules)
    score = sum(float(t["weight"]) for t in triggered)
    return score, triggered

