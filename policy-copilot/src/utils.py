from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from google import genai


LIKELY_DOCUMENTS = [
    "claim form",
    "bills",
    "discharge summary",
    "doctor prescription",
    "diagnostic reports",
    "id proof",
]


@dataclass
class ClaimDetails:
    claim_type: str
    treatment_description: str
    policy_start_date: date | None
    event_date: date | None
    estimated_amount: str
    hospitalisation: str
    network_hospital: str
    available_documents: list[str]
    additional_notes: str


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def parse_optional_date(value: str) -> date | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    return datetime.strptime(cleaned, "%Y-%m-%d").date()


def normalize_flag(value: str, allow_unknown: bool = False) -> str:
    cleaned = value.strip().lower()
    if cleaned in {"y", "yes"}:
        return "yes"
    if cleaned in {"n", "no"}:
        return "no"
    if allow_unknown and cleaned in {"", "unknown", "u", "not sure"}:
        return "unknown"
    return cleaned or ("unknown" if allow_unknown else "no")


def normalize_document_list(value: str) -> list[str]:
    if not value.strip():
        return []
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def compute_policy_age_label(
    policy_start_date: date | None,
    event_date: date | None,
) -> str | None:
    if not policy_start_date or not event_date:
        return None
    delta_days = (event_date - policy_start_date).days
    if delta_days < 0:
        return "event date is before policy start date"
    if delta_days < 30:
        return "under 30 days"
    if delta_days < 365:
        return "under 1 year"
    if delta_days < 730:
        return "1 to 2 years"
    return "over 2 years"


def derive_document_gaps(available_documents: list[str]) -> list[str]:
    available = {doc.lower() for doc in available_documents}
    missing: list[str] = []
    for expected in LIKELY_DOCUMENTS:
        if not any(expected in doc or doc in expected for doc in available):
            missing.append(expected)
    return missing


def derive_consistency_hints(claim: ClaimDetails) -> list[str]:
    hints: list[str] = []
    if "surgery" in claim.claim_type.lower() and claim.hospitalisation == "no":
        hints.append(
            "Surgery is marked without hospitalisation. Confirm whether it was day-care or inpatient treatment."
        )
    if claim.policy_start_date and claim.event_date:
        delta_days = (claim.event_date - claim.policy_start_date).days
        if delta_days < 0:
            hints.append("Event date appears earlier than the policy start date.")
    if claim.network_hospital == "unknown":
        hints.append("Network hospital status is unknown and may affect claim process expectations.")
    return hints


def build_claim_query(claim: ClaimDetails, policy_age_label: str | None) -> str:
    parts = [
        claim.claim_type,
        claim.treatment_description,
        claim.hospitalisation,
        claim.network_hospital,
        "claim process",
        "documents",
        "coverage",
        "waiting period",
    ]
    if policy_age_label:
        parts.append(policy_age_label)
    if claim.additional_notes:
        parts.append(claim.additional_notes)
    return normalize_whitespace(" ".join(part for part in parts if part))


def generate_text(prompt: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing from .env")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini returned an empty response.")
    return text.strip()


def format_sources(chunks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        quote = normalize_whitespace(chunk["text"])
        if len(quote) > 280:
            quote = quote[:277].rstrip() + "..."
        lines.append(
            f"{index}. Section: {chunk['section']} | Page: {chunk['page']} | Quote: {quote}"
        )
    return "\n".join(lines)


def dedupe_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for chunk in chunks:
        fingerprint = hashlib.sha1(
            normalize_whitespace(chunk["text"]).encode("utf-8")
        ).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(chunk)
    return unique


def read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
