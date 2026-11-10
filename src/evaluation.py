"""Metrik evaluasi retrieval berdasarkan relevance judgment."""
import numpy as np
import pandas as pd


def precision_at_k(ranked_ids, relevant_ids, k):
    if k <= 0:
        return 0.0
    ranked = list(ranked_ids[:k])
    return sum(doc_id in relevant_ids for doc_id in ranked) / k


def recall_at_k(ranked_ids, relevant_ids, k):
    if k <= 0 or not relevant_ids:
        return 0.0
    return sum(doc_id in relevant_ids for doc_id in ranked_ids[:k]) / len(relevant_ids)


def average_precision(ranked_ids, relevant_ids):
    if not relevant_ids:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            total += hits / rank
    return total / len(relevant_ids)


def ndcg_at_k(ranked_ids, relevance_map, k):
    if k <= 0:
        return 0.0
    gains = [max(0, int(relevance_map.get(doc, 0))) for doc in ranked_ids[:k]]
    dcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(gains))
    ideal = sorted((max(0, int(v)) for v in relevance_map.values()), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def evaluate_query(ranked_ids, relevance_map, relevant_ids=None, k=10):
    if relevant_ids is None:
        relevant_ids = {doc_id for doc_id, grade in relevance_map.items() if int(grade) > 0}
    p = precision_at_k(ranked_ids, relevant_ids, k)
    r = recall_at_k(ranked_ids, relevant_ids, k)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"Precision@K": p, "Recall@K": r, "F1-Score": f1,
            "AP": average_precision(ranked_ids, relevant_ids),
            "nDCG@K": ndcg_at_k(ranked_ids, relevance_map, k)}


def aggregate_evaluation(results):
    """Rata-rata metrik per metode dari DataFrame/list hasil per query."""
    df = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    if df.empty:
        return pd.DataFrame()
    method_col = next((c for c in ["metode", "method", "Metode", "Method"] if c in df.columns), None)
    if not method_col:
        raise ValueError("Kolom metode tidak ditemukan.")
    aliases = {"Precision@5": ["Precision@5"], "Precision@10": ["Precision@10"],
               "Recall@10": ["Recall@10"], "F1-Score": ["F1-Score"],
               "MAP": ["MAP", "AP"], "nDCG@10": ["nDCG@10"]}
    rename = {}
    for output, names in aliases.items():
        source = next((name for name in names if name in df.columns), None)
        if source:
            rename[source] = output
    if not rename:
        raise ValueError("Tidak ada kolom metrik yang dikenali.")
    out = df.groupby(method_col, as_index=False)[list(rename)].mean().rename(columns=rename)
    return out.rename(columns={method_col: "metode"})
