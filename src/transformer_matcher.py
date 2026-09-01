from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _model():
    """Load the sentence-transformer lazily."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def semantic_skill_match(
    query_skill: str,
    candidate_skills: list[str],
    threshold: float = 0.55,
) -> list[dict]:
    """Return transformer-embedding matches with defensive input normalization."""
    query_skill = str(query_skill).strip()
    candidates = [str(skill).strip() for skill in candidate_skills if str(skill).strip()]
    if not query_skill or not candidates:
        return []

    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        threshold_value = 0.55

    model = _model()
    # Explicitly pass an integer batch size and request a NumPy array. This avoids
    # version-specific argument inference issues in sentence-transformers.
    embeddings = model.encode(
        [query_skill, *candidates],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    query_vector = embeddings[0]
    candidate_vectors = embeddings[1:]
    scores = candidate_vectors @ query_vector

    results = [
        {"skill": skill, "similarity": round(float(score), 4)}
        for skill, score in zip(candidates, scores)
        if float(score) >= threshold_value
    ]
    return sorted(results, key=lambda item: item["similarity"], reverse=True)
