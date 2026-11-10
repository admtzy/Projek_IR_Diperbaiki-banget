# IR Lab — BM25 vs Multilingual-E5 vs Hybrid

Aplikasi Streamlit untuk pencarian artikel ilmiah dan evaluasi retrieval.

## Apa yang diperbaiki
- Semantic retrieval menggunakan `intfloat/multilingual-e5-small`, model multilingual yang ditujukan untuk retrieval.
- Prefix E5 diterapkan secara konsisten: `passage:` pada dokumen dan `query:` pada query.
- Batas input diatur ke 512 token agar tidak memakai batas pendek default secara tidak sengaja.
- Teks dokumen dibangun dari judul + abstrak (fallback ke abstrak bahasa lain bila kosong) + kata kunci.
- Menambahkan opsi Hybrid RRF untuk menggabungkan ranking BM25 dan E5.
- Evaluasi memvalidasi ID dokumen dan relevance judgment serta menghindari hasil contoh dianggap sebagai hasil penelitian.

## Instalasi Windows
Disarankan Python 3.10 atau 3.11.

```powershell
cd "Projek IR"
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Urutan menjalankan
Letakkan dataset asli di `data/full_articles.csv`, lalu jalankan dari root proyek:

```powershell
python scripts/preprocess.py
python scripts/build_index.py
streamlit run app.py
```

Build pertama mengunduh model E5 jika belum ada di cache dan dapat membutuhkan waktu. Jangan tutup terminal selama proses berjalan.

## File utama
- `scripts/preprocess.py`: membaca data mentah dan membentuk `data/processed_documents.csv`.
- `scripts/build_index.py`: membangun BM25, membuat embedding E5, dan menyimpan file ke `artifacts/`.
- `src/retrieval.py`: memuat artefak dan menjalankan BM25, E5, atau hybrid RRF.
- `src/evaluation.py`: Precision, Recall, F1, AP/MAP, nDCG.
- `app.py`: UI pencarian, perbandingan, dan evaluasi.
- `scripts/evaluate.py`: evaluasi lewat terminal dengan file relevance judgment.

## Mengapa SBERT sebelumnya dapat tidak relevan?
1. `paraphrase-multilingual-MiniLM-L12-v2` pada konfigurasi umum sering memakai batas input relatif pendek; abstrak panjang dapat terpotong.
2. Kode lama tidak menggunakan model retrieval khusus dan tidak memakai instruksi/prefix E5.
3. Kolom `Abstrak` dan `Abstract` sama-sama ada. Memilih satu kolom secara otomatis dapat membuat banyak dokumen memakai abstrak kosong meskipun kolom lainnya berisi teks.
4. Metadata/kata kunci tidak disusun ke dalam teks pencarian.
5. Tanpa relevance judgment, belum bisa dibuktikan bahwa satu model lebih baik; tampilan yang terlihat tidak relevan perlu diuji pada query nyata.

Perubahan ini dirancang memperbaiki penyebab umum tersebut, tetapi tidak menjamin semua hasil relevan. Evaluasi dengan query dan label relevansi manusia tetap diperlukan.

## Tes cepat hasil pencarian

Setelah indeks selesai dibangun, jalankan:

```powershell
python scripts/diagnose_search.py
```

Skrip ini mencetak top-5 dari BM25, E5, dan hybrid untuk beberapa query contoh. Ini hanya inspeksi kualitatif, bukan evaluasi ilmiah; hasilnya tetap harus dinilai manusia.

## Relevance judgment
Salin template menjadi `relevance_judgments.csv` atau unggah CSV lewat Streamlit. Format:

`query_id,query,doc_id,relevance`

Nilai: `0` tidak relevan, `1` relevan sebagian, `2` sangat relevan. Gunakan ID persis dari `artifacts/documents.csv`. Nilai relevansi ditentukan manusia berdasarkan isi dokumen. Sebaiknya tiap query memiliki beberapa dokumen relevan dan cukup dokumen tidak relevan yang dinilai; jika hanya sebagian korpus dinilai, Recall dan MAP perlu ditafsirkan hati-hati.

Evaluasi lewat terminal:

```powershell
python scripts/evaluate.py
```

atau berikan path CSV:

```powershell
python scripts/evaluate.py data/relevance_judgments.csv
```

## Catatan teknis
- Jika mengganti model, preprocessing, atau isi teks dokumen, jalankan ulang `python scripts/build_index.py`.
- Jangan mengedit urutan `artifacts/documents.csv` secara manual; urutannya harus sama dengan embedding dan BM25.
- Skor mentah BM25 dan cosine similarity tidak sebanding langsung.
- `Hybrid RRF` menggabungkan peringkat, bukan mencampur skor mentah.
- Untuk riset yang dapat direplikasi, simpan query set, relevance judgment, versi model, dan konfigurasi yang digunakan.
