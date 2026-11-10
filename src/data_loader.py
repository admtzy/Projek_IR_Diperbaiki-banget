"""Fungsi membaca CSV dan menyusun representasi dokumen artikel."""
from __future__ import annotations
import html
import re
from pathlib import Path
from typing import Iterable
import pandas as pd


def load_dataset(source) -> pd.DataFrame:
    """Membaca CSV dengan fallback encoding dan membersihkan nama kolom."""
    if hasattr(source, "read"):
        try:
            source.seek(0)
        except Exception:
            pass
        try:
            df = pd.read_csv(source, low_memory=False, keep_default_na=False)
        except UnicodeDecodeError:
            source.seek(0)
            df = pd.read_csv(source, encoding="latin-1", low_memory=False, keep_default_na=False)
    else:
        path = Path(source)
        try:
            df = pd.read_csv(path, low_memory=False, keep_default_na=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1", low_memory=False, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_display_text(value) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_column(df: pd.DataFrame, candidates: Iterable[str], exclude=None):
    """Cari kolom dengan pencocokan nama yang tidak sensitif kapital/spasi."""
    exclude = set(exclude or [])
    normalized = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in df.columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
        if key in normalized and normalized[key] not in exclude:
            return normalized[key]
    return None


def build_document_text(
    df: pd.DataFrame,
    title_col: str,
    abstract_col: str,
    id_col: str,
    extra_cols=None,
    fallback_text_cols=None,
) -> pd.DataFrame:
    """Bangun satu baris per dokumen; gunakan abstrak cadangan jika abstrak utama kosong."""
    extra_cols = list(extra_cols or [])
    fallback_text_cols = list(fallback_text_cols or [])
    rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        title = clean_display_text(row.get(title_col, ""))
        abstract = clean_display_text(row.get(abstract_col, ""))
        if not abstract:
            for col in fallback_text_cols:
                candidate = clean_display_text(row.get(col, ""))
                if candidate:
                    abstract = candidate
                    break
        keywords = ""
        for col in df.columns:
            if str(col).strip().lower() in {"keywords", "keyword", "kata kunci", "kata_kunci"}:
                keywords = clean_display_text(row.get(col, ""))
                break
        if not title and not abstract and not keywords:
            continue
        raw_id = clean_display_text(row.get(id_col, ""))
        doc_id = raw_id or f"DOC{i:06d}"
        # Teks indeks berisi judul, abstrak, dan kata kunci. Tidak di-stem untuk SBERT.
        parts = []
        if title:
            parts.append(f"Judul: {title}")
        if abstract:
            parts.append(f"Abstrak: {abstract}")
        if keywords:
            parts.append(f"Kata kunci: {keywords}")
        metadata_items = []
        for col in extra_cols:
            val = clean_display_text(row.get(col, ""))
            if val:
                metadata_items.append(f"{col}: {val}")
        url_col = find_column(df, ["url", "link", "article_url", "doi_url", "tautan"])
        url = clean_display_text(row.get(url_col, "")) if url_col else ""
        rows.append({
            "doc_id": str(doc_id),
            "title": title or "(Tanpa judul)",
            "abstract": abstract,
            "keywords": keywords,
            "document_text": ". ".join(parts),
            "metadata": " · ".join(metadata_items),
            "url": url,
        })
    return pd.DataFrame(rows)


def normalize_columns(df: pd.DataFrame):
    return {str(c).strip().lower(): c for c in df.columns}
