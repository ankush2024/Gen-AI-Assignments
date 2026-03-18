from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils import dedupe_chunks


@dataclass
class PolicyRetriever:
    collection: Any
    embedding_model: Any

    def retrieve(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        items: list[dict[str, Any]] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for document, metadata in zip(documents, metadatas):
            items.append(
                {
                    "text": document,
                    "page": metadata.get("page", "Unknown"),
                    "section": metadata.get("section", "General"),
                    "source": metadata.get("source", "policy.pdf"),
                }
            )

        return dedupe_chunks(items)
