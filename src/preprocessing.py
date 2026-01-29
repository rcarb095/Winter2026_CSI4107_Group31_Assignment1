import os
import re
import json
from typing import Iterable, List, Optional, Set, Iterator, Tuple

from nltk.stem import PorterStemmer  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore


DEFAULT_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "if", "in", "into", "is", "it", "no", "not", "of", "on", "or",
    "such", "that", "the", "their", "then", "there", "these", "they",
    "this", "to", "was", "will", "with",
}


# Cache stopwords so we only have to read the file once
_STOPWORDS_CACHE: Optional[Set[str]] = None
_DEFAULT_STOPWORDS_PATH = os.path.join("resources", "List_of_Stopwords.txt")
_TAG_RE = re.compile(r"<[^>]+>")


def _load_stopwords_from_txt(path: str) -> Set[str]:
    """
    Load a stopword list from the text file provides.
    The text file should contain 1 word per line.
    """
    with open(path, "r", encoding="utf-8") as f:
        words = [w.strip().lower() for w in f.read().splitlines()]
    return {w for w in words if w}


def _get_stopwords(
    stopwords: Optional[Iterable[str]],
    stopwords_path: Optional[str],
) -> Set[str]:
    """
    Resolve stopwords from provided iterable type, optional text file, or default stopword list.
    """
    global _STOPWORDS_CACHE
    # Use the iterable provided if there is any
    if stopwords is not None:
        return set(stopwords)
    # If no explicit path is given, fall back to the default filename
    path = stopwords_path or _DEFAULT_STOPWORDS_PATH
    if _STOPWORDS_CACHE is not None:
        # Reuse cached stopwords if there is a saved cache
        return _STOPWORDS_CACHE
    if os.path.exists(path):
        # Load and cache from disk
        _STOPWORDS_CACHE = _load_stopwords_from_txt(path)
        return _STOPWORDS_CACHE
    return DEFAULT_STOPWORDS


_PORTER_STEMMER = PorterStemmer()


def tokenize(
    text: str,
    stopwords: Optional[Iterable[str]] = None,
    stopwords_path: Optional[str] = None,
) -> List[str]:
    """
    Tokenize and normalize text for indexing

    Steps:
    - Remove markup
    - Lowercase
    - Remove punctuation/symbols and numbers
    - Split on whitespace
    - Remove stopwords
    - Optionally stem tokens
    """
    # Strip simple tag-based markup like <b>...</b>.
    text = _TAG_RE.sub(" ", text)
    text = text.lower()

    # NLTK's tokenizer splits punctuation and contractions into separate tokens
    tokens = word_tokenize(text)
    # Keep tokens that contain at least one letter; drops pure numbers/punctuation.
    tokens = [t for t in tokens if re.search(r"[a-z]", t)]

    stopword_set = _get_stopwords(stopwords, stopwords_path)
    # Removes stopwords so the remaining tokens are index terms
    tokens = [t for t in tokens if t not in stopword_set]

    return tokens


def stem_tokens(tokens: Iterable[str]) -> List[str]:
    """
    Stem a token sequence using NLTK's PorterStemmer.
    """
    return [_PORTER_STEMMER.stem(t) for t in tokens]


def preprocess_documents(
    documents: Iterable[str],
    stopwords: Optional[Iterable[str]] = None,
    stopwords_path: Optional[str] = None,
) -> Iterable[List[str]]:
    """
    Process documents one by one and yield token lists.
    """
    # Stream docs one by one.
    for doc in documents:
        tokens = tokenize(
            doc,
            stopwords=stopwords,
            stopwords_path=stopwords_path,
        )
        yield stem_tokens(tokens)


def iter_scifact_corpus(
    path: str = os.path.join("data", "scifact", "corpus.jsonl"),
    fields: Tuple[str, ...] = ("title", "text"),
) -> Iterator[Tuple[str, str]]:
    """
    Yield (doc_id, combined_text) from the SciFact corpus.
    """
    # Stream JSONL to not load everything into memory
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = str(obj.get("_id", ""))
            parts = [obj.get(k, "") for k in fields]
            combined = " ".join(p for p in parts if p)
            yield doc_id, combined


def iter_scifact_tokens(
    path: str = os.path.join("data", "scifact", "corpus.jsonl"),
    fields: Tuple[str, ...] = ("title", "text"),
    stopwords: Optional[Iterable[str]] = None,
    stopwords_path: Optional[str] = None,
) -> Iterator[Tuple[str, List[str]]]:
    """
    Yield (doc_id, tokens) for each SciFact document in the corpus
    """
    # Yield token lists alongside document IDs for easier indexing
    for doc_id, text in iter_scifact_corpus(path=path, fields=fields):
        tokens = tokenize(
            text,
            stopwords=stopwords,
            stopwords_path=stopwords_path,
        )
        yield doc_id, stem_tokens(tokens)

#main function for testing purposes
if __name__ == "__main__":
    for i, (doc_id, tokens) in enumerate(iter_scifact_tokens()):
        print(f"{doc_id}\t{tokens}")
        if i >= 2:
            break
