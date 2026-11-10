"""Bangun indeks BM25 dan embedding multilingual-E5. Jalankan dari root proyek."""
import json
import pickle
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.data_loader import load_dataset
from src.preprocessing import preprocess_for_bm25
from src.retrieval import DEFAULT_MODEL, DEFAULT_MAX_SEQ_LENGTH

DATA_PATH = ROOT_DIR / "data" / "processed_documents.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_NAME = DEFAULT_MODEL
BATCH_SIZE = 24


def main():
    if not DATA_PATH.exists():
        raise SystemExit("processed_documents.csv belum ada. Jalankan: python scripts/preprocess.py")
    docs = load_dataset(DATA_PATH)
    required = {"doc_id", "title", "document_text"}
    missing = required - set(docs.columns)
    if missing:
        raise SystemExit(f"Kolom wajib tidak ditemukan: {sorted(missing)}")
    if docs.empty:
        raise SystemExit("Dataset kosong.")
    for col in ["doc_id", "title", "document_text"]:
        docs[col] = docs[col].fillna("").astype(str).str.strip()
    docs = docs[docs["document_text"].ne("")].copy()
    if docs["doc_id"].eq("").any():
        raise SystemExit("Ada doc_id kosong. Perbaiki preprocessing terlebih dahulu.")
    if docs["doc_id"].duplicated().any():
        raise SystemExit("Ada doc_id duplikat. Perbaiki preprocessing terlebih dahulu.")
    for col in ["abstract", "keywords", "metadata", "url"]:
        if col not in docs:
            docs[col] = ""
        docs[col] = docs[col].fillna("").astype(str)
    if "bm25_text" not in docs:
        docs["bm25_text"] = ""
    docs["bm25_text"] = docs["bm25_text"].fillna("").astype(str).str.strip()
    empty_bm25 = docs["bm25_text"].eq("")
    if empty_bm25.any():
        print(f"Membuat teks BM25 cadangan untuk {int(empty_bm25.sum())} dokumen...")
        docs.loc[empty_bm25, "bm25_text"] = docs.loc[empty_bm25, "document_text"].map(lambda t: " ".join(preprocess_for_bm25(t)))
    docs = docs.reset_index(drop=True)
    if docs["bm25_text"].str.strip().eq("").all():
        raise SystemExit("Semua teks BM25 kosong; tidak bisa membangun indeks.")
    print("=" * 65)
    print("MEMBANGUN INDEKS INFORMATION RETRIEVAL")
    print(f"Dokumen: {len(docs):,}")
    print(f"Model semantik: {MODEL_NAME}")
    print("=" * 65)
    print("[1/3] Membangun BM25...")
    tokenized = [text.split() for text in docs["bm25_text"]]
    bm25 = BM25Okapi(tokenized)
    print("[2/3] Memuat model E5 dan membuat embedding...")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = DEFAULT_MAX_SEQ_LENGTH
    # Prefix passage: adalah bagian dari cara penggunaan model multilingual-E5.
    passages = [f"passage: {text}" for text in docs["document_text"].tolist()]
    embeddings = model.encode(passages, batch_size=BATCH_SIZE, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) != len(docs):
        raise RuntimeError("Bentuk/jumlah embedding tidak sesuai dengan jumlah dokumen.")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Embedding mengandung nilai NaN/inf.")
    print("[3/3] Menyimpan artefak...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    docs.to_csv(ARTIFACTS_DIR / "documents.csv", index=False, encoding="utf-8-sig")
    with open(ARTIFACTS_DIR / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(ARTIFACTS_DIR / "document_embeddings.npy", embeddings, allow_pickle=False)
    config = {
        "model_name": MODEL_NAME,
        "model_type": "e5",
        "max_seq_length": DEFAULT_MAX_SEQ_LENGTH,
        "document_count": int(len(docs)),
        "embedding_dimension": int(embeddings.shape[1]),
        "batch_size": BATCH_SIZE,
        "text_strategy": "title + abstract (fallback available) + keywords; passage prefix",
    }
    with open(ARTIFACTS_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print("\nIndeks berhasil dibangun.")
    print(f"Dokumen       : {len(docs):,}")
    print(f"Dimensi vektor: {embeddings.shape[1]}")
    print(f"Maks. token   : {DEFAULT_MAX_SEQ_LENGTH}")
    print(f"Lokasi        : {ARTIFACTS_DIR}")
    print("Lanjutkan dengan: streamlit run app.py")

if __name__ == "__main__":
    main()
