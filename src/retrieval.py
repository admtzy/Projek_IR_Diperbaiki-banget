"""Retrieval BM25, semantic retrieval multilingual-E5, dan hybrid RRF."""
from __future__ import annotations
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.preprocessing import preprocess_for_bm25

# Model E5 dibuat untuk retrieval multilingual dan mendukung prefix query:/passage:.
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_MAX_SEQ_LENGTH = 512


class IRSystem:
    def __init__(self, documents, bm25_index, embeddings, model_name=DEFAULT_MODEL,
                 max_seq_length=DEFAULT_MAX_SEQ_LENGTH, model_type="e5"):
        required = {"doc_id", "title", "document_text", "bm25_text"}
        missing = required - set(documents.columns)
        if missing:
            raise ValueError(f"Kolom dokumen tidak lengkap: {sorted(missing)}")
        if documents.empty:
            raise ValueError("Dataset dokumen kosong.")
        self.documents = documents.reset_index(drop=True).copy()
        for column in ["doc_id", "title", "document_text", "bm25_text"]:
            self.documents[column] = self.documents[column].fillna("").astype(str)
        for column in ["abstract", "keywords", "metadata", "url"]:
            if column not in self.documents:
                self.documents[column] = ""
            self.documents[column] = self.documents[column].fillna("").astype(str)
        if self.documents["doc_id"].str.strip().eq("").any():
            raise ValueError("Terdapat doc_id kosong pada indeks.")
        if self.documents["doc_id"].duplicated().any():
            raise ValueError("doc_id pada indeks harus unik.")
        self.bm25 = bm25_index
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.model_name = model_name
        self.max_seq_length = int(max_seq_length)
        self.model_type = str(model_type).lower()
        if self.embeddings.ndim != 2 or len(self.documents) != len(self.embeddings):
            raise ValueError("Jumlah/bentuk embedding tidak cocok dengan dokumen. Bangun ulang indeks.")
        if getattr(self.bm25, "corpus_size", len(self.documents)) != len(self.documents):
            raise ValueError("Jumlah dokumen pada indeks BM25 tidak cocok. Bangun ulang indeks.")
        print(f"Memuat model semantic retrieval: {model_name}")
        self.model = SentenceTransformer(model_name)
        # E5 mendukung input sampai 512 token. Atur eksplisit untuk menghindari truncation 128-token.
        self.model.max_seq_length = self.max_seq_length
        if self.embeddings.shape[1] != self.model.get_sentence_embedding_dimension():
            raise ValueError(
                "Dimensi embedding tidak cocok dengan model. Hapus artefak lama lalu jalankan scripts/build_index.py."
            )

    @classmethod
    def load(cls, artifacts_dir):
        artifacts_dir = Path(artifacts_dir)
        paths = {
            "documents": artifacts_dir / "documents.csv",
            "bm25": artifacts_dir / "bm25_index.pkl",
            "embeddings": artifacts_dir / "document_embeddings.npy",
            "config": artifacts_dir / "config.json",
        }
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError("Artefak indeks belum lengkap:\n" + "\n".join(missing) +
                                    "\nJalankan: python scripts/build_index.py")
        documents = pd.read_csv(paths["documents"], keep_default_na=False, dtype={"doc_id": str})
        with open(paths["bm25"], "rb") as f:
            bm25_index = pickle.load(f)
        embeddings = np.load(paths["embeddings"], allow_pickle=False)
        with open(paths["config"], encoding="utf-8") as f:
            config = json.load(f)
        return cls(documents, bm25_index, embeddings,
                   model_name=config.get("model_name", DEFAULT_MODEL),
                   max_seq_length=config.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH),
                   model_type=config.get("model_type", "e5"))

    def _encode_query(self, query: str):
        # E5 model card mensyaratkan prefix query: untuk query dan passage: untuk dokumen.
        text = f"query: {query}" if self.model_type == "e5" else query
        return self.model.encode([text], convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)

    def search(self, query, mode="bm25", top_k=10):
        query = str(query or "").strip()
        top_k = max(1, int(top_k))
        if not query:
            result = self.documents.iloc[0:0].copy()
            result["score"] = pd.Series(dtype=float)
            result["rank"] = pd.Series(dtype=int)
            return result
        method = str(mode).lower().replace("_", "-")
        n_docs = len(self.documents)
        if method == "bm25":
            tokens = preprocess_for_bm25(query)
            scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)
        elif method in {"sbert", "semantic", "sentence-bert", "e5"}:
            q_emb = self._encode_query(query)
            # Embedding dokumen dan query telah dinormalisasi; dot product = cosine similarity.
            scores = np.asarray(self.embeddings @ q_emb[0], dtype=np.float32)
        elif method in {"hybrid", "rrf"}:
            # Reciprocal Rank Fusion menggabungkan ranking tanpa membandingkan skala skor mentah.
            bm25_scores = np.asarray(self.bm25.get_scores(preprocess_for_bm25(query)), dtype=np.float32)
            q_emb = self._encode_query(query)
            semantic_scores = np.asarray(self.embeddings @ q_emb[0], dtype=np.float32)
            bm25_rank = np.empty(n_docs, dtype=np.int64)
            bm25_rank[np.argsort(-bm25_scores, kind="stable")] = np.arange(1, n_docs + 1)
            sem_rank = np.empty(n_docs, dtype=np.int64)
            sem_rank[np.argsort(-semantic_scores, kind="stable")] = np.arange(1, n_docs + 1)
            rrf_k = 60
            scores = 1.0 / (rrf_k + bm25_rank) + 1.0 / (rrf_k + sem_rank)
        else:
            raise ValueError("mode harus 'bm25', 'sbert', atau 'hybrid'.")
        order = np.argsort(-scores, kind="stable")[:min(top_k, n_docs)]
        results = self.documents.iloc[order].copy()
        results["score"] = scores[order]
        results["rank"] = np.arange(1, len(results) + 1)
        return results.reset_index(drop=True)
