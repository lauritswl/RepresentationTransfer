from typing import Optional
import numpy as np

"""
RepresentationTransfer.src package initializer.

Provides a small utility class to build a concept vector from two sets
of embeddings (positive / negative examples). The concept vector is
by default the difference between the mean positive and mean negative
embeddings, optionally L2-normalized. You can project new embeddings
onto the concept vector or compute similarity scores.
"""

class ConceptVector:
    """
    Build and use a concept vector from two embedding sets.

    Usage:
        cv = ConceptVector()
        cv.fit(pos_embeddings, neg_embeddings)          # compute vector
        projection = cv.project(new_embeddings)         # scalar projection(s)
        sims = cv.cosine_similarity(new_embeddings)     # cosine similarity(s)

    Parameters:
        normalize (bool): If True, the concept vector is L2-normalized.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.vector: Optional[np.ndarray] = None
        self.pos_mean: Optional[np.ndarray] = None
        self.neg_mean: Optional[np.ndarray] = None
        self.dim: Optional[int] = None

    def _to_2d(self, x):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("Embeddings must be 1D or 2D array-like")
        return arr

    def fit(self, pos_embeddings, neg_embeddings):
        """
        Compute the concept vector from positive and negative embeddings.

        pos_embeddings: array-like, shape (n_pos, d) or (d,)
        neg_embeddings: array-like, shape (n_neg, d) or (d,)
        """
        pos = self._to_2d(pos_embeddings)
        neg = self._to_2d(neg_embeddings)

        if pos.shape[1] != neg.shape[1]:
            raise ValueError("Positive and negative embeddings must have same dimensionality")

        self.dim = pos.shape[1]
        self.pos_mean = pos.mean(axis=0)
        self.neg_mean = neg.mean(axis=0)

        vec = self.neg_mean - self.pos_mean 
        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        self.vector = vec
        return self

    def project(self, embeddings):
        """
        Project embeddings onto the concept vector.

        Returns scalar projection(s). If embeddings is shape (n, d) returns (n,),
        if embeddings is shape (d,) returns a scalar.
        """
        if self.vector is None:
            raise RuntimeError("Concept vector not set. Call fit() first.")
        emb = self._to_2d(embeddings)
        if emb.shape[1] != self.dim:
            raise ValueError("Embedding dimensionality mismatch")
        # If vector is normalized, dot gives signed magnitude along the concept.
        scores = emb.dot(self.vector)
        return scores if scores.shape[0] > 1 else float(scores[0])

    def as_array(self):
        """Return the concept vector as a 1D numpy array (or None)."""
        return None if self.vector is None else np.asarray(self.vector)