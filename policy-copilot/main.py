from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Make the `src/` package modules importable as top-level modules
# so other code can use `from precheck import ...` instead of `from src.precheck import ...`.
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from precheck import run_claim_precheck
from qa import run_policy_qa
from retriever import PolicyRetriever
from vector_store import build_or_load_retriever


def print_menu() -> None:
    print("\nPolicy & Claims Copilot")
    print("1. Ask policy question")
    print("2. Run claim pre-check")
    print("3. Exit")


def main() -> None:
    load_dotenv()

    policy_path = Path("data") / "policy.pdf"
    if not policy_path.exists():
        print("Missing file: data/policy.pdf")
        sys.exit(1)

    try:
        retriever: PolicyRetriever = build_or_load_retriever(policy_path)
    except Exception as exc:
        print(f"Failed to prepare vector store: {exc}")
        sys.exit(1)

    while True:
        print_menu()
        choice = input("\nSelect an option: ").strip()

        if choice == "1":
            run_policy_qa(retriever)
        elif choice == "2":
            run_claim_precheck(retriever)
        elif choice == "3":
            print("Exiting.")
            break
        else:
            print("Enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
