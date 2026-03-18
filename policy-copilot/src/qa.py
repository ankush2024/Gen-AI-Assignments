from __future__ import annotations

from prompts import build_qa_prompt
from retriever import PolicyRetriever
from utils import generate_text


def run_policy_qa(retriever: PolicyRetriever) -> None:
    question = input("\nEnter your policy question: ").strip()
    if not question:
        print("Question cannot be empty.")
        return

    chunks = retriever.retrieve(question, k=4)
    if not chunks:
        print("No relevant policy text was retrieved.")
        return

    prompt = build_qa_prompt(question, chunks)
    try:
        response = generate_text(prompt)
    except Exception as exc:
        print(f"Failed to generate answer: {exc}")
        return

    print("\nPolicy Q&A")
    print("-" * 60)
    print(response)
    print("\nGuardrail: This tool provides policy assistance only and does not make final claim decisions.")
