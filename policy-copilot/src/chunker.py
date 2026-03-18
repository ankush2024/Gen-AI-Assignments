from __future__ import annotations

import re
from typing import Any


HEADING_PATTERN = re.compile(r"^[A-Z][A-Za-z0-9 /,&()\-]{2,80}$")


def looks_like_heading(line: str) -> bool:
    stripped = line.strip().strip(":")
    if len(stripped) < 3 or len(stripped) > 80:
        return False
    if stripped.endswith("."):
        return False
    words = stripped.split()
    if len(words) > 10:
        return False
    if stripped.isupper():
        return True
    title_words = sum(1 for word in words if word[:1].isupper())
    if title_words >= max(1, len(words) - 1) and HEADING_PATTERN.match(stripped):
        return True
    return False


def split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            split_point = text.rfind(" ", start, end)
            if split_point > start + (chunk_size // 2):
                end = split_point
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_pages(
    pages: list[dict[str, Any]],
    source: str,
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current_section = "General"

    for page_data in pages:
        page_number = page_data["page"]
        page_text = page_data["text"]
        if not page_text.strip():
            continue

        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        buffer: list[str] = []
        section_for_buffer = current_section

        for line in lines:
            if looks_like_heading(line):
                if buffer:
                    chunks.extend(
                        _make_chunks(
                            "\n".join(buffer),
                            page_number,
                            section_for_buffer,
                            source,
                            chunk_size,
                            overlap,
                        )
                    )
                    buffer = []
                current_section = line.strip().strip(":")
                section_for_buffer = current_section
                continue
            buffer.append(line)

        if buffer:
            chunks.extend(
                _make_chunks(
                    "\n".join(buffer),
                    page_number,
                    section_for_buffer,
                    source,
                    chunk_size,
                    overlap,
                )
            )

    return chunks


def _make_chunks(
    text: str,
    page: int,
    section: str,
    source: str,
    chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for piece in split_long_text(text, chunk_size, overlap):
        output.append(
            {
                "text": piece,
                "page": page,
                "section": section or "General",
                "source": source,
            }
        )
    return output
