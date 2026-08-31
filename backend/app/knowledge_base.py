"""FAISS-backed retrieval over past incident write-ups.

Backs the `query_similar_incidents` agent tool. Fifteen write-ups is small
enough that a brute-force scan would work, but the index is the piece that has
to survive the corpus growing, so it's built properly and persisted.

Embeddings are TF-IDF vectors, L2-normalised so FAISS inner product *is*
cosine similarity. That choice is deliberate: it needs no model download, no
network, and no API key, which keeps `make seed`-to-`make investigate`
runnable offline and deterministic. It matches on shared vocabulary rather
than meaning, so a query phrased entirely in synonyms will miss --- swap
`_embed_corpus`/`_embed_query` for a sentence-embedding model when that starts
to bite; nothing else in this module or its callers needs to change.
"""
import json
import re
from dataclasses import dataclass

import numpy as np

from app.config import FAISS_INDEX_PATH, FAISS_VECTORIZER_PATH, PAST_INCIDENTS_PATH

try:
    import faiss
    HAS_FAISS = True
except ImportError:  # pragma: no cover - exercised only in stripped installs
    HAS_FAISS = False

MIN_SCORE = 0.02   # below this a "match" is shared stopwords, not shared meaning


@dataclass(frozen=True)
class Retrieval:
    incident: dict
    score: float


def load_corpus(path=PAST_INCIDENTS_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"past incident corpus not found at {path}; it ships with the repo, "
            f"so this usually means the working directory is wrong")
    corpus = json.loads(path.read_text())
    if not corpus:
        raise ValueError(f"{path} is empty")
    return corpus


def document_text(incident):
    """Flatten one write-up into the text that gets indexed.

    Title and lesson carry the most retrieval signal per token, so they're
    weighted by repetition rather than by a separate field-boosting scheme.
    """
    parts = [
        incident.get('title', ''), incident.get('title', ''),
        incident.get('service', ''),
        ' '.join(incident.get('metrics', []) or []),
        incident.get('tier', ''),
        incident.get('summary', ''),
        incident.get('root_cause', ''),
        incident.get('evidence', ''),
        incident.get('lesson', ''), incident.get('lesson', ''),
    ]
    return ' '.join(p for p in parts if p)


def _tokenize(text):
    return re.findall(r"[a-z_][a-z0-9_\-]+", text.lower())


class TfidfEmbedder:
    """Minimal TF-IDF vectoriser.

    Hand-rolled rather than sklearn's so the persisted index and the vocabulary
    that produced it stay a single small JSON blob — an sklearn pickle would
    silently break across library upgrades, and a stale vocabulary paired with
    a live index returns confident nonsense.
    """

    def __init__(self, vocabulary=None, idf=None):
        self.vocabulary = vocabulary or {}
        self.idf = np.asarray(idf, dtype='float32') if idf is not None else None

    def fit(self, documents):
        tokenized = [_tokenize(d) for d in documents]
        vocab = sorted({t for doc in tokenized for t in doc})
        self.vocabulary = {term: i for i, term in enumerate(vocab)}

        n_docs = len(documents)
        doc_freq = np.zeros(len(vocab), dtype='float32')
        for doc in tokenized:
            for term in set(doc):
                doc_freq[self.vocabulary[term]] += 1
        # Smoothed idf, as in sklearn: never divides by zero, never yields 0
        # weight for a term that appears in every document.
        self.idf = np.log((1.0 + n_docs) / (1.0 + doc_freq)).astype('float32') + 1.0
        return self

    def transform(self, documents):
        if not self.vocabulary:
            raise RuntimeError("TfidfEmbedder.transform called before fit")
        matrix = np.zeros((len(documents), len(self.vocabulary)), dtype='float32')
        for row, text in enumerate(documents):
            for term in _tokenize(text):
                index = self.vocabulary.get(term)
                if index is not None:      # unseen terms carry no weight
                    matrix[row, index] += 1.0
        matrix *= self.idf
        # L2-normalise so inner product == cosine similarity. An all-zero row
        # (a query sharing no vocabulary with the corpus) would divide by zero.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-9)

    def fit_transform(self, documents):
        return self.fit(documents).transform(documents)

    def state_dict(self):
        return {'vocabulary': self.vocabulary, 'idf': self.idf.tolist()}

    @classmethod
    def from_state_dict(cls, state):
        return cls(vocabulary=state['vocabulary'], idf=state['idf'])


class IncidentIndex:
    """Searchable index over the past-incident corpus."""

    def __init__(self, corpus, embedder, index):
        self.corpus = corpus
        self.embedder = embedder
        self.index = index

    @classmethod
    def build(cls, corpus=None):
        corpus = corpus if corpus is not None else load_corpus()
        embedder = TfidfEmbedder()
        vectors = embedder.fit_transform([document_text(i) for i in corpus])
        return cls(corpus, embedder, _new_index(vectors))

    def search(self, query, k=3):
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        k = max(1, min(int(k), len(self.corpus)))
        vector = self.embedder.transform([query])
        scores, indices = _search_index(self.index, vector, k)
        return [
            Retrieval(incident=self.corpus[int(i)], score=float(s))
            for s, i in zip(scores[0], indices[0])
            # FAISS pads with -1 when fewer than k neighbours exist.
            if i >= 0 and s >= MIN_SCORE
        ]

    def save(self, index_path=FAISS_INDEX_PATH, vectorizer_path=FAISS_VECTORIZER_PATH):
        index_path.parent.mkdir(parents=True, exist_ok=True)
        vectorizer_path.write_text(json.dumps(self.embedder.state_dict()))
        if HAS_FAISS:
            faiss.write_index(self.index, str(index_path))
        else:
            np.save(index_path.with_suffix('.npy'), self.index)
        return index_path


def _new_index(vectors):
    if HAS_FAISS:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index
    # Without faiss the corpus is small enough to scan directly; keeping this
    # path means an install missing a compiled wheel degrades instead of dying.
    return vectors


def _search_index(index, query_vectors, k):
    if HAS_FAISS:
        return index.search(query_vectors, k)
    similarities = query_vectors @ index.T
    top = np.argsort(-similarities, axis=1)[:, :k]
    return np.take_along_axis(similarities, top, axis=1), top


_CACHED_INDEX = None


def get_index():
    """Process-wide singleton. Building is cheap but not free, and every agent
    tool call would otherwise re-read and re-vectorise the whole corpus."""
    global _CACHED_INDEX
    if _CACHED_INDEX is None:
        _CACHED_INDEX = IncidentIndex.build()
    return _CACHED_INDEX


def main():
    index = IncidentIndex.build()
    path = index.save()
    backend = "faiss" if HAS_FAISS else "numpy fallback"
    print(f"indexed {len(index.corpus)} past incident write-up(s) using {backend}")
    print(f"  vocabulary: {len(index.embedder.vocabulary)} terms")
    print(f"  index -> {path}")

    for probe in ["cpu spike with no traffic increase",
                  "latency slowly climbing over hours",
                  "two services rose together at the same time"]:
        hits = index.search(probe, k=2)
        print(f"\n  probe: {probe!r}")
        for hit in hits:
            print(f"    {hit.score:.3f}  {hit.incident['id']}  {hit.incident['title']}")


if __name__ == "__main__":
    main()
