from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz


def load_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []

    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            text = page.get_text("text")
            cleaned_text = text.replace("\x00", " ").strip()
            pages.append({"page": index, "text": cleaned_text})

    return pages
