"""Evaluasi BM25, E5, dan hybrid memakai penilaian relevansi manual.
Format: query_id,query,doc_id,relevance (0, 1, atau 2).
Jalankan: python scripts/evaluate.py [path_csv_judgments]
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import pandas as pd
from src.data_loader import load_dataset
from src.retrieval import IRSystem
from src.evaluation import evaluate_query, precision_at_k

DEFAULT_JUDGMENTS = ROOT_DIR / "relevance_judgments.csv"
OUT_DETAIL = ROOT_DIR / "data" / "evaluation_per_query.csv"
OUT_SUMMARY = ROOT_DIR / "data" / "evaluation_results.csv"


def main():
    judgment_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_JUDGMENTS
    if not judgment_path.exists():
        raise SystemExit(f"File penilaian tidak ditemukan: {judgment_path}\nSalin template menjadi relevance_judgments.csv dan isi penilaian relevansi yang benar.")
    judgments = load_dataset(judgment_path)
    required = {"query_id", "query", "doc_id", "relevance"}
    if not required.issubset(judgments.columns):
        raise SystemExit(f"Kolom wajib: {sorted(required)}")
    judgments["query_id"] = judgments["query_id"].astype(str).str.strip()
    judgments["query"] = judgments["query"].astype(str).str.strip()
    judgments["doc_id"] = judgments["doc_id"].astype(str).str.strip()
    judgments["relevance"] = pd.to_numeric(judgments["relevance"], errors="coerce")
    if judgments[["query_id", "query", "doc_id", "relevance"]].isna().any().any():
        raise SystemExit("Terdapat nilai kosong/tidak valid pada file judgments.")
    if not judgments["relevance"].isin([0, 1, 2]).all():
        raise SystemExit("Nilai relevance harus 0, 1, atau 2.")
    if judgments.duplicated(["query_id", "doc_id"]).any():
        judgments = judgments.groupby(["query_id", "query", "doc_id"], as_index=False)["relevance"].max()
    ir = IRSystem.load(ROOT_DIR / "artifacts")
    ids = set(ir.documents["doc_id"].astype(str))
    unknown = set(judgments["doc_id"]) - ids
    if unknown:
        raise SystemExit(f"Ada {len(unknown)} doc_id pada judgments yang tidak ada di indeks. Contoh: {list(unknown)[:5]}")
    rows = []
    for (qid, query), group in judgments.groupby(["query_id", "query"], sort=False):
        rel_map = dict(zip(group["doc_id"], group["relevance"].astype(int)))
        relevant = {doc for doc, grade in rel_map.items() if grade > 0}
        if not relevant:
            print(f"Lewati {qid}: belum ada dokumen relevan berlabel 1 atau 2.")
            continue
        for mode in ["bm25", "sbert", "hybrid"]:
            ranked = ir.search(query, mode=mode, top_k=10)["doc_id"].astype(str).tolist()
            m5 = evaluate_query(ranked, rel_map, relevant, k=5)
            m10 = evaluate_query(ranked, rel_map, relevant, k=10)
            rows.append({"query_id": qid, "query": query, "metode": mode.upper(),
                         "Precision@5": m5["Precision@K"], "Precision@10": m10["Precision@K"],
                         "Recall@10": m10["Recall@K"], "F1-Score": m10["F1-Score"],
                         "AP": m10["AP"], "nDCG@10": m10["nDCG@K"]})
    if not rows:
        raise SystemExit("Tidak ada query yang bisa dievaluasi. Periksa relevance judgment.")
    detail = pd.DataFrame(rows)
    summary = detail.groupby("metode", as_index=False)[["Precision@5", "Precision@10", "Recall@10", "F1-Score", "AP", "nDCG@10"]].mean().rename(columns={"AP": "MAP"})
    OUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    print("\nDETAIL PER QUERY\n", detail.to_string(index=False))
    print("\nRINGKASAN\n", summary.to_string(index=False))
    print(f"\nDisimpan: {OUT_DETAIL}\nDisimpan: {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
