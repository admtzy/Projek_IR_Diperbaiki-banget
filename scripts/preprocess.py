"""Preprocessing dataset artikel. Jalankan dari root: python scripts/preprocess.py"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import pandas as pd
from tqdm import tqdm
from src.data_loader import load_dataset, build_document_text, find_column
from src.preprocessing import preprocess_for_bm25

DATA = ROOT_DIR / "data" / "full_articles.csv"
OUT = ROOT_DIR / "data" / "processed_documents.csv"


def main():
    if not DATA.exists():
        raise SystemExit(f"Dataset tidak ditemukan: {DATA}")
    df = load_dataset(DATA)
    print(f"Dataset mentah: {len(df):,} baris")
    print("Kolom:", list(df.columns))
    title_col = find_column(df, ["Title", "Judul", "Article Title", "Document Title"])
    # Dataset proyek memiliki Abstrak (Indonesia) dan Abstract (Inggris).
    # Pilih Abstrak sebagai utama, lalu fallback per baris ke Abstract.
    abstract_col = find_column(df, ["Abstrak", "Abstract", "Ringkasan", "Description", "Content", "Text"])
    id_col = find_column(df, ["ID", "Document ID", "Article ID", "DOI", "URL"])
    if not title_col or not abstract_col or not id_col:
        raise SystemExit(f"Kolom wajib gagal dideteksi: title={title_col}, text={abstract_col}, id={id_col}. Periksa nama kolom dataset.")
    fallback_cols = []
    for candidate in ["Abstract", "Abstrak", "Introduction", "Result & Discussion", "Conclusion"]:
        col = find_column(df, [candidate])
        if col and col != abstract_col and col not in fallback_cols:
            fallback_cols.append(col)
    extra_cols = [c for c in [find_column(df,["Journal"]), find_column(df,["Authors"]), find_column(df,["Year"])] if c]
    print(f"Pemetaan: ID={id_col!r}; judul={title_col!r}; abstrak utama={abstract_col!r}; fallback={fallback_cols}")
    docs = build_document_text(df, title_col, abstract_col, id_col, extra_cols=extra_cols, fallback_text_cols=fallback_cols)
    docs["doc_id"] = docs["doc_id"].astype(str).str.strip()
    docs["document_text"] = docs["document_text"].fillna("").astype(str).str.strip()
    docs = docs[docs["document_text"].str.len() >= 20].copy()
    # Jika ID duplikat, hentikan dengan pesan; jangan diam-diam menghapus dokumen.
    duplicated = docs["doc_id"].duplicated(keep=False)
    if duplicated.any():
        examples = docs.loc[duplicated, "doc_id"].head(10).tolist()
        raise SystemExit(f"ID dokumen duplikat ditemukan (contoh {examples}). Perbaiki ID sebelum indeks dibangun.")
    tqdm.pandas(desc="Preprocessing BM25")
    docs["bm25_text"] = docs["document_text"].progress_map(lambda t: " ".join(preprocess_for_bm25(t)))
    if docs["bm25_text"].str.strip().eq("").any():
        count = int(docs["bm25_text"].str.strip().eq("").sum())
        print(f"Peringatan: {count} dokumen menghasilkan teks BM25 kosong.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    docs.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\nSelesai: {OUT}\nDokumen tersimpan: {len(docs):,}")
    print(f"Abstrak/teks kosong: {int(docs['abstract'].str.strip().eq('').sum()):,}")
    print("Kolom output:", list(docs.columns))

if __name__ == "__main__":
    main()
