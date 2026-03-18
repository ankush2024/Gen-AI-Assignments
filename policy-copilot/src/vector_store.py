from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import chunk_pages
from pdf_loader import load_pdf_pages
from retriever import PolicyRetriever
from utils import read_manifest, write_manifest


COLLECTION_NAME = "policy_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class LocalEmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()


def build_or_load_retriever(policy_path: Path, db_path: Path | None = None) -> PolicyRetriever:
    db_dir = db_path or Path("db")
    db_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_dir))
    embedding_model = LocalEmbeddingModel(EMBEDDING_MODEL_NAME)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    manifest_path = db_dir / "index_manifest.json"
    current_manifest = {
        "source": policy_path.name,
        "size": policy_path.stat().st_size,
        "modified": int(policy_path.stat().st_mtime),
        "embedding_model": EMBEDDING_MODEL_NAME,
    }
    saved_manifest = read_manifest(manifest_path)

    should_rebuild = saved_manifest != current_manifest or collection.count() == 0

    if should_rebuild:
        pages = load_pdf_pages(policy_path)
        chunks = chunk_pages(pages, source=policy_path.name)
        if collection.count() > 0:
            existing_ids = collection.get(include=[])["ids"]
            if existing_ids:
                collection.delete(ids=existing_ids)

        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_model.encode(texts)
        metadatas = [
            {
                "page": chunk["page"],
                "section": chunk["section"],
                "source": chunk["source"],
                "text": chunk["text"],
            }
            for chunk in chunks
        ]
        ids = [f"chunk-{index}" for index in range(len(chunks))]
        collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

        current_manifest["chunk_count"] = len(chunks)
        write_manifest(manifest_path, current_manifest)

    return PolicyRetriever(collection=collection, embedding_model=embedding_model)
