import math
from collections import defaultdict
from typing import Dict, List, Tuple

from preprocessing import tokenize, stem_tokens, iter_scifact_tokens
from indexing import build_inverted_index
def compute_idf(inverted_index: Dict[str, Dict[str, int]], N: int) -> Dict[str, float]:
    #idf(t) = log(N / df(t))
    idf: Dict[str, float] = {}
    for term, postings in inverted_index.items():
        df = len(postings)
        if df > 0:
            idf[term] = math.log(N / df)
    return idf

def compute_doc_norms(inverted_index: Dict[str, Dict[str, int]], idf: Dict[str, float],) -> Dict[str, float]:
    doc_norms: Dict[str, float] = defaultdict(float)

    for term, postings in inverted_index.items():
        term_idf = idf.get(term, 0.0)
        for doc_id, tf in postings.items():
            weight = tf * term_idf
            doc_norms[doc_id] += weight * weight

    # Take square root to get the Euclidean norm
    for doc_id in doc_norms:
        doc_norms[doc_id] = math.sqrt(doc_norms[doc_id])

    return doc_norms

def rank_documents(
    query_text: str,
    inverted_index: Dict[str, Dict[str, int]],
    idf: Dict[str, float],
    doc_norms: Dict[str, float],
    top_k: int = 100,
) -> List[Tuple[str, float]]:

    # Query preprocessing
    query_tokens = stem_tokens(tokenize(query_text))

    # Query TF
    q_tf = defaultdict(int)
    for t in query_tokens:
        q_tf[t] += 1

    # Query weights + norm
    q_weights = {}
    q_norm = 0.0
    for term, tf in q_tf.items():
        if term in idf:
            w = tf * idf[term]
            q_weights[term] = w
            q_norm += w * w
    q_norm = math.sqrt(q_norm)

    scores = defaultdict(float)

    # Dot product
    for term, w_q in q_weights.items():
        if term in inverted_index:
            for doc_id, tf_d in inverted_index[term].items():
                w_d = tf_d * idf[term]
                scores[doc_id] += w_q * w_d

    # Cosine normalization
    for doc_id in scores:
        denom = q_norm * doc_norms.get(doc_id, 0.0)
        if denom > 0:
            scores[doc_id] /= denom

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

if __name__ == "__main__":
    # Minimal sanity test for Step 3

    # Build index from a few documents only (fast)
    token_stream = []
    for i, item in enumerate(iter_scifact_tokens()):
        token_stream.append(item)
        if i >= 200:   # limit for quick testing
            break

    inverted_index = build_inverted_index(token_stream)
    N = len(token_stream)

    idf = compute_idf(inverted_index, N)
    doc_norms = compute_doc_norms(inverted_index, idf)

    print("STEP 3: IDF sample")
    for term in list(idf.keys())[:3]:
        print(term, idf[term])

    query = "COVID-19 vaccines"
    results = rank_documents(query, inverted_index, idf, doc_norms)

    print("\nSTEP 3: Top 3 results")
    for doc_id, score in results[:3]:
        print(doc_id, f"{score:.4f}")



