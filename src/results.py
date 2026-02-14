import os
import json
from typing import Iterator, Tuple, List

from preprocessing import iter_scifact_tokens, tokenize, stem_tokens
from indexing import build_inverted_index
from retrieval import compute_idf, compute_doc_norms, rank_documents


def iter_scifact_queries(
    path: str = os.path.join("data", "scifact", "queries.jsonl"),
) -> Iterator[Tuple[int, str]]:
    # Reads the SciFact queries file and yields (query_id, query_text) one by one
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            qid_raw = obj.get("_id", None)
            qtext = obj.get("text", "") or ""
            if qid_raw is None:
                continue

            try:
                qid = int(str(qid_raw))
            except ValueError:
                continue

            yield qid, qtext


def shorten_query(qtext: str, max_terms: int = 5) -> str:
    # Title-only query
    tokens = stem_tokens(tokenize(qtext))
    return " ".join(tokens[:max_terms])


def write_results_file(
    queries: List[Tuple[int, str]],
    inverted_index,
    idf,
    doc_norms,
    results_path: str,
    run_name: str,
    top_k: int = 100,
):

    # Writes: query_id Q0 doc_id rank score tag
    with open(results_path, "w", encoding="utf-8") as out:
        for qid, qtext in queries:
            ranked = rank_documents(
                query_text=qtext,
                inverted_index=inverted_index,
                idf=idf,
                doc_norms=doc_norms,
                top_k=top_k,
            )
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")


def main(top_k: int = 100):
    # Build index on full corpus
    token_stream = list(iter_scifact_tokens())  # list[(doc_id, tokens)]
    inverted_index = build_inverted_index(token_stream)
    N = len(token_stream)

    idf = compute_idf(inverted_index, N)
    doc_norms = compute_doc_norms(inverted_index, idf)

    # Load only odd-numbered test queries
    queries_full = [(qid, qtext) for (qid, qtext) in iter_scifact_queries() if qid % 2 == 1]
    queries_full.sort(key=lambda x: x[0])

    # Run B: full query text (title + full text)
    write_results_file(
        queries=queries_full,
        inverted_index=inverted_index,
        idf=idf,
        doc_norms=doc_norms,
        results_path="Results",
        run_name="tfidf_text",
        top_k=top_k,
    )

    # Run A: title-only run
    queries_title = [(qid, shorten_query(qtext)) for (qid, qtext) in queries_full]

    write_results_file(
        queries=queries_title,
        inverted_index=inverted_index,
        idf=idf,
        doc_norms=doc_norms,
        results_path="Results_title",
        run_name="tfidf_title",
        top_k=top_k,
    )

    print(f"Wrote Results and Results_title with {len(queries_full)} queries each, top_k={top_k}")


if __name__ == "__main__":
    main(top_k=100)
