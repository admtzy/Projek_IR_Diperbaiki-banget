"""Streamlit UI untuk membandingkan BM25, multilingual-E5, dan hybrid RRF."""
from pathlib import Path
import pandas as pd
import streamlit as st
from src.retrieval import IRSystem
from src.evaluation import evaluate_query

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
st.set_page_config(page_title="IR Lab | BM25 vs E5", page_icon="🔎", layout="wide")
st.title("🔎 Information Retrieval: BM25 vs Semantic Search")
st.caption("Perbandingan pencarian kata, semantic retrieval multilingual-E5, dan hybrid pada artikel ilmiah.")

@st.cache_resource(show_spinner="Memuat indeks dan model semantic retrieval...")
def load_system():
    return IRSystem.load(ARTIFACTS_DIR)

try:
    system = load_system()
except Exception as exc:
    st.error("Sistem belum siap. Jalankan preprocessing dan pembangunan indeks terlebih dahulu.")
    st.code("python scripts/preprocess.py\npython scripts/build_index.py\nstreamlit run app.py", language="powershell")
    with st.expander("Detail error"):
        st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("Pengaturan")
    top_k = st.slider("Jumlah hasil", 5, 20, 10)
    st.divider()
    st.write(f"Dokumen terindeks: **{len(system.documents):,}**")
    st.write(f"Model semantik: `{system.model_name}`")
    st.write(f"Batas input model: **{system.max_seq_length} token**")
    st.caption("Skor BM25 dan cosine similarity berbeda skala; bandingkan ranking atau metrik evaluasi.")


def show_results(results, show_text=True):
    if results.empty:
        st.info("Tidak ada hasil untuk query ini.")
        return
    for _, row in results.iterrows():
        with st.container(border=True):
            st.markdown(f"**{int(row['rank'])}. {row.get('title', '(Tanpa judul)')}**")
            st.caption(f"ID: {row.get('doc_id','')} · Skor: {float(row.get('score',0)):.4f}")
            metadata = str(row.get("metadata", "")).strip()
            if metadata:
                st.caption(metadata)
            if show_text:
                abstract = str(row.get("abstract", "")).strip()
                text = abstract or str(row.get("document_text", "")).strip()
                if text:
                    st.write(text[:1000] + ("…" if len(text) > 1000 else ""))
                keywords = str(row.get("keywords", "")).strip()
                if keywords:
                    st.caption(f"Kata kunci: {keywords}")
            url = str(row.get("url", "")).strip()
            if url.startswith(("https://", "http://")):
                st.markdown(f"[Buka sumber artikel]({url})")

search_tab, compare_tab, eval_tab, about_tab = st.tabs(["🔍 Pencarian", "⚖️ Perbandingan", "📊 Evaluasi", "ℹ️ Tentang"])

with search_tab:
    query = st.text_input("Query pencarian", placeholder="Contoh: pengaruh literasi digital terhadap minat baca", key="single_query")
    method_label = st.selectbox("Metode", ["BM25 (kata kunci)", "E5 (kemiripan semantik)", "Hybrid (BM25 + E5)"])
    if st.button("Cari artikel", type="primary", key="single_search"):
        if not query.strip():
            st.warning("Masukkan query terlebih dahulu.")
        else:
            mode = {"BM25 (kata kunci)":"bm25", "E5 (kemiripan semantik)":"sbert", "Hybrid (BM25 + E5)":"hybrid"}[method_label]
            with st.spinner("Mencari dokumen..."):
                results = system.search(query, mode=mode, top_k=top_k)
            st.write(f"Hasil teratas: {len(results)} dokumen · metode {method_label}")
            show_results(results)

with compare_tab:
    query2 = st.text_input("Query yang sama untuk semua metode", placeholder="Masukkan satu query yang jelas", key="compare_query")
    if st.button("Bandingkan metode", type="primary", key="compare_button"):
        if not query2.strip():
            st.warning("Masukkan query terlebih dahulu.")
        else:
            cols = st.columns(3)
            for col, mode, title in zip(cols, ["bm25", "sbert", "hybrid"], ["BM25", "Multilingual-E5", "Hybrid RRF"]):
                with col:
                    st.subheader(title)
                    show_results(system.search(query2, mode=mode, top_k=top_k), show_text=False)
            st.info("Hybrid menggunakan Reciprocal Rank Fusion (RRF), sehingga menggabungkan posisi ranking BM25 dan E5 tanpa menyamakan skala skor mentah.")

with eval_tab:
    st.subheader("Evaluasi berdasarkan relevance judgment")
    st.markdown("Nilai relevansi: **0** tidak relevan, **1** relevan sebagian, **2** sangat relevan. Nilai harus diberikan oleh penilai berdasarkan isi dokumen, bukan dihasilkan otomatis oleh model yang sedang dibandingkan.")
    template = "query_id,query,doc_id,relevance\nQ001,contoh query,ID_DOKUMEN,2\n"
    st.download_button("Unduh template CSV", template, file_name="relevance_judgments_template.csv", mime="text/csv")
    uploaded = st.file_uploader("Unggah CSV relevance judgment", type=["csv"], key="eval_upload")
    if uploaded is not None:
        try:
            judgments = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
            required = {"query_id", "query", "doc_id", "relevance"}
            if not required.issubset(judgments.columns):
                st.error("Kolom wajib: query_id, query, doc_id, relevance.")
            else:
                judgments["relevance"] = pd.to_numeric(judgments["relevance"], errors="coerce")
                invalid = judgments["query_id"].str.strip().eq("") | judgments["query"].str.strip().eq("") | judgments["doc_id"].str.strip().eq("") | judgments["relevance"].isna() | ~judgments["relevance"].isin([0,1,2])
                if invalid.any():
                    st.error("Ada baris kosong/tidak valid. Isi query_id, query, doc_id, dan relevance 0/1/2.")
                elif judgments.duplicated(["query_id", "doc_id"]).any():
                    st.error("Setiap pasangan query_id dan doc_id harus unik.")
                else:
                    ids = set(system.documents["doc_id"].astype(str))
                    unknown = set(judgments["doc_id"].astype(str)) - ids
                    if unknown:
                        st.error(f"{len(unknown)} doc_id tidak ditemukan pada indeks. Contoh: {list(unknown)[:5]}")
                    elif st.button("Jalankan evaluasi", type="primary", key="run_eval"):
                        detail_rows = []
                        for (qid, query_text), group in judgments.groupby(["query_id", "query"], sort=False):
                            rel_map = dict(zip(group["doc_id"].astype(str), group["relevance"].astype(int)))
                            relevant = {doc for doc, grade in rel_map.items() if grade > 0}
                            if not relevant:
                                continue
                            for mode in ["bm25", "sbert", "hybrid"]:
                                ranked = system.search(query_text, mode=mode, top_k=10)["doc_id"].astype(str).tolist()
                                m5 = evaluate_query(ranked, rel_map, relevant, k=5)
                                m10 = evaluate_query(ranked, rel_map, relevant, k=10)
                                detail_rows.append({"query_id":qid,"query":query_text,"metode":mode.upper(),"Precision@5":m5["Precision@K"],"Precision@10":m10["Precision@K"],"Recall@10":m10["Recall@K"],"F1-Score":m10["F1-Score"],"AP":m10["AP"],"nDCG@10":m10["nDCG@K"]})
                        if not detail_rows:
                            st.error("Tidak ada query dengan dokumen relevan berlabel 1 atau 2.")
                        else:
                            detail = pd.DataFrame(detail_rows)
                            summary = detail.groupby("metode", as_index=False)[["Precision@5","Precision@10","Recall@10","F1-Score","AP","nDCG@10"]].mean().rename(columns={"AP":"MAP"})
                            st.markdown("### Rata-rata semua query")
                            st.dataframe(summary, use_container_width=True, hide_index=True)
                            st.markdown("### Detail per query")
                            st.dataframe(detail, use_container_width=True, hide_index=True)
                            st.download_button("Unduh hasil ringkasan", summary.to_csv(index=False).encode("utf-8-sig"), file_name="evaluation_results.csv", mime="text/csv")
                            st.download_button("Unduh detail evaluasi", detail.to_csv(index=False).encode("utf-8-sig"), file_name="evaluation_per_query.csv", mime="text/csv")
                            st.caption("Evaluasi valid memerlukan cakupan dokumen relevan yang memadai untuk setiap query. Jika hanya sebagian dokumen dinilai, interpretasikan Recall/MAP dengan hati-hati.")
        except Exception as exc:
            st.error(f"Gagal membaca file penilaian: {exc}")

with about_tab:
    st.subheader("Tentang sistem")
    st.markdown("""
    - **BM25** menghitung kecocokan kata dengan mempertimbangkan frekuensi istilah dan panjang dokumen.
    - **Multilingual-E5** membuat embedding teks multilingual untuk retrieval semantik. Query diberi awalan `query:` dan dokumen `passage:` sesuai penggunaan model.
    - **Hybrid RRF** menggabungkan ranking BM25 dan E5 menggunakan Reciprocal Rank Fusion.
    - Model pretrained tidak dilatih ulang pada dataset ini; proses build hanya membuat embedding dan indeks.
    - Relevansi hasil harus diuji dengan query dan relevance judgment yang dinilai manusia.
    """)
    st.write(f"Model: `{system.model_name}`")
    st.write(f"Dimensi embedding: {system.embeddings.shape[1]}")
