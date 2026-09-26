"""Shared training/inference MiniLM encoding and normalized-vector cosine.

Inference caches each distinct normalized string on disk, loading only the current
batch into RAM. Empty strings have no vector and contribute cosine 0, as in training.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Revision already present in the cache used by the original training run.
EMBEDDING_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
ENCODE_BATCH_SIZE = 512


def load_embedding_model(device=None):
    import torch
    from sentence_transformers import SentenceTransformer
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    return SentenceTransformer(EMBEDDING_MODEL_NAME, revision=EMBEDDING_REVISION,
                               device=device, local_files_only=True)


def encode_unique(model, strings):
    texts = sorted({s for s in strings if s})
    if not texts:
        return {}
    vectors = model.encode(texts, batch_size=ENCODE_BATCH_SIZE,
                           normalize_embeddings=True, show_progress_bar=True)
    return dict(zip(texts, vectors))


def embedding_cosine(first, second):
    return float(np.dot(first, second)) if first is not None and second is not None else 0.0


class EmbeddingCache:
    def __init__(self, path, model=None, device=None):
        self.model = model if model is not None else load_embedding_model(device)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS embeddings (text TEXT PRIMARY KEY, vector BLOB)")
        contract = json.dumps({"model": EMBEDDING_MODEL_NAME, "revision": EMBEDDING_REVISION,
                               "normalize_embeddings": True, "dtype": "float32",
                               "max_seq_length": self.model.max_seq_length}, sort_keys=True)
        old = self.connection.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
        if old and old[0] != contract:
            raise ValueError("Embedding cache belongs to a different encoding contract")
        self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('contract', ?)", (contract,))
        self.connection.commit()

    def get_many(self, strings):
        texts = sorted({s for s in strings if s})
        found = {}
        for start in range(0, len(texts), 500):
            batch = texts[start:start + 500]
            marks = ','.join('?' for _ in batch)
            for text, blob in self.connection.execute(
                    f"SELECT text, vector FROM embeddings WHERE text IN ({marks})", batch):
                found[text] = np.frombuffer(blob, dtype=np.float32)
        missing = [s for s in texts if s not in found]
        # Bound each encode call so a large country never retains all its vectors.
        for start in range(0, len(missing), 8192):
            new = encode_unique(self.model, missing[start:start + 8192])
            self.connection.executemany("INSERT OR IGNORE INTO embeddings VALUES (?, ?)",
                                        [(s, np.asarray(v, dtype=np.float32).tobytes()) for s, v in new.items()])
            self.connection.commit()
            found.update(new)
        return found

    def close(self):
        self.connection.close()
