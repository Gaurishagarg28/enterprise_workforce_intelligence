from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _model():
    """Load a sentence-transformer lazily so the basic API does not pay startup cost unless used."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def semantic_skill_match(query_skill: str, candidate_skills: list[str], threshold: float = 0.55) -> list[dict]:
    """Use transformer embeddings to match semantically related skills."""
    if not candidate_skills:
        return []
    model = _model()
    embeddings = model.encode([query_skill] + candidate_skills, normalize_embeddings=True)
    query = embeddings[0]
    scores = embeddings[1:] @ query
    results = [
        {"skill": skill, "similarity": round(float(score), 4)}
        for skill, score in zip(candidate_skills, scores)
        if float(score) >= threshold
    ]
    return sorted(results, key=lambda x: x["similarity"], reverse=True)
