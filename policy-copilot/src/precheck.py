from __future__ import annotations

from prompts import build_precheck_prompt
from retriever import PolicyRetriever
from utils import (
    ClaimDetails,
    build_claim_query,
    compute_policy_age_label,
    derive_consistency_hints,
    derive_document_gaps,
    generate_text,
    normalize_document_list,
    normalize_flag,
    parse_optional_date,
)


def collect_claim_details() -> ClaimDetails | None:
    try:
        claim_type = input("\nClaim Type: ").strip()
        treatment_description = input("Treatment Description: ").strip()
        policy_start_date = parse_optional_date(input("Policy Start Date (YYYY-MM-DD, optional): "))
        event_date = parse_optional_date(input("Event Date (YYYY-MM-DD, optional): "))
        estimated_amount = input("Estimated Amount (optional): ").strip()
        hospitalisation = normalize_flag(input("Involves hospitalisation? (yes/no): "))
        network_hospital = normalize_flag(
            input("Network hospital? (yes/no/unknown): "), allow_unknown=True
        )
        available_documents = normalize_document_list(
            input("Available Documents (comma-separated, optional): ")
        )
        additional_notes = input("Additional Notes (optional): ").strip()
    except ValueError as exc:
        print(f"Invalid input: {exc}")
        return None

    return ClaimDetails(
        claim_type=claim_type,
        treatment_description=treatment_description,
        policy_start_date=policy_start_date,
        event_date=event_date,
        estimated_amount=estimated_amount,
        hospitalisation=hospitalisation,
        network_hospital=network_hospital,
        available_documents=available_documents,
        additional_notes=additional_notes,
    )


def run_claim_precheck(retriever: PolicyRetriever) -> None:
    claim = collect_claim_details()
    if not claim:
        return

    policy_age_label = compute_policy_age_label(claim.policy_start_date, claim.event_date)
    document_gaps = derive_document_gaps(claim.available_documents)
    consistency_hints = derive_consistency_hints(claim)
    query = build_claim_query(claim, policy_age_label)
    chunks = retriever.retrieve(query, k=5)

    if not chunks:
        print("No relevant policy text was retrieved.")
        return

    claim_summary = _build_claim_summary(claim)
    helper_signals = _build_helper_signals(policy_age_label, document_gaps, consistency_hints)
    prompt = build_precheck_prompt(claim_summary, helper_signals, chunks)

    try:
        response = generate_text(prompt)
    except Exception as exc:
        print(f"Failed to generate pre-check: {exc}")
        return

    print("\nClaim Pre-check")
    print("-" * 60)
    print(response)
    print(
        "\nGuardrail: This is a policy assistance and pre-check tool only. Final claim decisions depend on insurer review, documents, and policy terms."
    )


def _build_claim_summary(claim: ClaimDetails) -> str:
    parts = [
        f"Claim type: {claim.claim_type or 'not provided'}",
        f"Treatment: {claim.treatment_description or 'not provided'}",
        f"Policy start date: {claim.policy_start_date.isoformat() if claim.policy_start_date else 'not provided'}",
        f"Event date: {claim.event_date.isoformat() if claim.event_date else 'not provided'}",
        f"Estimated amount: {claim.estimated_amount or 'not provided'}",
        f"Hospitalisation: {claim.hospitalisation}",
        f"Network hospital: {claim.network_hospital}",
        f"Available documents: {', '.join(claim.available_documents) if claim.available_documents else 'not provided'}",
        f"Additional notes: {claim.additional_notes or 'not provided'}",
    ]
    return "\n".join(parts)


def _build_helper_signals(
    policy_age_label: str | None,
    document_gaps: list[str],
    consistency_hints: list[str],
) -> str:
    lines = [
        f"Policy age label: {policy_age_label or 'not available'}",
        f"Possible missing documents: {', '.join(document_gaps) if document_gaps else 'none identified'}",
    ]
    if consistency_hints:
        lines.extend(f"- {hint}" for hint in consistency_hints)
    else:
        lines.append("- No obvious consistency issues found from the provided inputs.")
    return "\n".join(lines)
