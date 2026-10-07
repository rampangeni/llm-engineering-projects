"""Knowledge-base loading."""

from __future__ import annotations

from pathlib import Path

from .models import Document


def load_documents(knowledge_base: Path) -> list[Document]:
    """Load every ``*.md`` under ``<kb>/<doc_type>/**``; the folder name is the doc type."""
    kb = Path(knowledge_base)
    if not kb.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb}")
    docs: list[Document] = []
    for folder in sorted(p for p in kb.iterdir() if p.is_dir()):
        for file in sorted(folder.rglob("*.md")):
            docs.append(
                Document(
                    source=file.relative_to(kb).as_posix(),
                    type=folder.name,
                    text=file.read_text(encoding="utf-8"),
                )
            )
    return docs
