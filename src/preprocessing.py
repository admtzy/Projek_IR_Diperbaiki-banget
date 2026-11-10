"""Preprocessing khusus BM25. Teks SBERT sengaja tidak di-stem."""
import html
import re
from functools import lru_cache


def clean_text(text) -> str:
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Pertahankan angka, kata, tanda hubung, dan istilah seperti e-journal.
    text = re.sub(r"[^\w\s\-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


@lru_cache(maxsize=1)
def get_stopwords():
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        return set(StopWordRemoverFactory().get_stop_words())
    except Exception:
        return set("yang dan di ke dari ini itu untuk pada dengan adalah atau sebagai dalam juga akan dapat oleh terhadap suatu para secara karena maka merupakan yaitu".split())


@lru_cache(maxsize=1)
def get_stemmer():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        return StemmerFactory().create_stemmer()
    except Exception:
        return None


def preprocess_for_bm25(text, remove_stopwords=True, stemming=True):
    """Menghasilkan list token untuk BM25; Sastrawi dipakai jika terpasang."""
    tokens = clean_text(text).split()
    if remove_stopwords:
        stopwords = get_stopwords()
        tokens = [token for token in tokens if token not in stopwords]
    stemmer = get_stemmer() if stemming else None
    if stemmer:
        tokens = [stemmer.stem(token) for token in tokens]
    return tokens
