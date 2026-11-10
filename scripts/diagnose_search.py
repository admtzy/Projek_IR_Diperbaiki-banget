"""Tes manual hasil BM25/E5/Hybrid pada query yang ditentukan.
Jalankan: python scripts/diagnose_search.py
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from src.retrieval import IRSystem

QUERIES = [
    "pengaruh literasi digital terhadap minat baca mahasiswa",
    "promosi perpustakaan untuk meningkatkan minat kunjung siswa",
    "pemanfaatan teknologi informasi dalam layanan perpustakaan",
]


def main():
    system = IRSystem.load(ROOT_DIR / "artifacts")
    for query in QUERIES:
        print("\n" + "=" * 90)
        print("QUERY:", query)
        for mode in ["bm25", "sbert", "hybrid"]:
            print(f"\n--- {mode.upper()} ---")
            results = system.search(query, mode=mode, top_k=5)
            for _, row in results.iterrows():
                print(f"{int(row['rank']):>2}. [{row['score']:.4f}] {row['doc_id']} | {row['title']}")

if __name__ == "__main__":
    main()
