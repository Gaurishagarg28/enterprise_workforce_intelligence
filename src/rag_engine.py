from __future__ import annotations

from pathlib import Path

KNOWLEDGE_DIR = Path("data/knowledge_base")

DEFAULT_DOC = KNOWLEDGE_DIR / "workforce_policy.md"

DEFAULT_TEXT = """# Workforce Decision Support Policy

- Attrition predictions are decision-support signals, not automatic employment decisions.
- High-risk employees should receive human review before any intervention.
- Reskilling is preferred when a material skill gap exists and readiness is sufficient.
- Recommendations should use job, performance, engagement and training evidence available in the system.
- Sensitive demographic attributes must not be used to justify a recommendation.
"""


def retrieve(query: str, top_k: int = 3) -> list[dict[str, str]]:
    """Lightweight lexical RAG for the MVP; replace with Qdrant retrieval when the corpus grows."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_DOC.exists():
        DEFAULT_DOC.write_text(DEFAULT_TEXT, encoding="utf-8")

    terms = {t.lower() for t in query.split() if len(t) > 2}
    text = DEFAULT_DOC.read_text(encoding="utf-8")
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    scored = []
    for chunk in chunks:
        score = sum(term in chunk.lower() for term in terms)
        scored.append((score, chunk))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [{"source": DEFAULT_DOC.name, "text": chunk} for score, chunk in scored[:top_k] if score > 0]
