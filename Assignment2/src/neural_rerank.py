import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer, util
import torch


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "scifact"
RESULTS_DIR = BASE_DIR / "Results"


def _select_device() -> str:
    """
    Use CUDA when available, otherwise fall back to CPU.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_text_by_id(path: Path) -> Dict[str, str]:
    """
    Load a JSONL file and return {id: text} with IDs normalized to strings.
    """
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            item_id = str(obj.get("_id", ""))
            text = obj.get("text", "") or ""
            if item_id:
                values[item_id] = text
    return values


def _load_top_docs_by_query(path: Path) -> Dict[str, List[str]]:
    """
    Read TREC-style baseline run file and return {query_id: [doc_id,...]}.
    """
    top_docs: Dict[str, List[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, docid, _, _, _ = parts[:6]
            top_docs[qid].append(docid)
    return top_docs


def rerank_and_save(
    model_name: str,
    run_tag: str,
    output_filename: str,
    baseline_filename: str = "baseline_results.txt",
    device: Optional[str] = None,
) -> Path:
    """
    Re-rank baseline top docs with a neural embedding model and write TREC output.
    """
    active_device = device or _select_device()
    model = SentenceTransformer(model_name, device=active_device)

    corpus_path = DATA_DIR / "corpus.jsonl"
    queries_path = DATA_DIR / "queries.jsonl"
    baseline_path = RESULTS_DIR / baseline_filename

    corpus = _load_text_by_id(corpus_path)
    queries = _load_text_by_id(queries_path)
    top_docs_per_query = _load_top_docs_by_query(baseline_path)

    # Build query/doc worklists first so each unique document is embedded once.
    docs_by_query: Dict[str, List[str]] = {}
    query_ids: List[str] = []
    query_texts: List[str] = []
    unique_doc_ids: List[str] = []
    seen_docs = set()

    for qid, doc_ids in top_docs_per_query.items():
        query_text = queries.get(qid, "")
        if not query_text:
            continue

        filtered_doc_ids = [doc_id for doc_id in doc_ids if doc_id in corpus]
        if not filtered_doc_ids:
            continue

        docs_by_query[qid] = filtered_doc_ids
        query_ids.append(qid)
        query_texts.append(query_text)

        for doc_id in filtered_doc_ids:
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            unique_doc_ids.append(doc_id)

    doc_texts = [corpus[doc_id] for doc_id in unique_doc_ids]
    doc_embs = model.encode(doc_texts, convert_to_tensor=True, batch_size=32)
    query_embs = model.encode(query_texts, convert_to_tensor=True, batch_size=32)
    doc_index = {doc_id: idx for idx, doc_id in enumerate(unique_doc_ids)}

    results: List[str] = []
    for query_idx, qid in enumerate(query_ids):
        filtered_doc_ids = docs_by_query[qid]
        doc_indices = [doc_index[doc_id] for doc_id in filtered_doc_ids]
        query_doc_embs = doc_embs[doc_indices]

        sim_scores = util.cos_sim(query_embs[query_idx], query_doc_embs).flatten()
        ranked_indices = sim_scores.argsort(descending=True).tolist()

        for rank, doc_idx in enumerate(ranked_indices, start=1):
            score = float(sim_scores[doc_idx])
            doc_id = filtered_doc_ids[doc_idx]
            results.append(f"{qid} Q0 {doc_id} {rank} {score:.4f} {run_tag}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / output_filename
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(results))

    return output_path


if __name__ == "__main__":
    device = _select_device()
    output = rerank_and_save(
        model_name="all-MiniLM-L6-v2",
        run_tag="BERT-rerank",
        output_filename="Results_rerank_bert.txt",
        device=device,
    )
    print(f"BERT reranked results saved to {output} (device={device})")
