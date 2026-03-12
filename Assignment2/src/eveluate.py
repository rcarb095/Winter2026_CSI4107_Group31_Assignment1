import os
from collections import defaultdict
import shutil

DATA_DIR = "../data/scifact"
RESULTS_DIR = "./Results"


def load_qrels():
    qrels = defaultdict(dict)

    with open(os.path.join(DATA_DIR, "qrels", "test.tsv"), "r") as f:
        for line in f:
            qid, docid, rel = line.strip().split()
            qrels[qid][docid] = int(rel)

    return qrels


def load_run(path):
    run = defaultdict(list)

    with open(path, "r") as f:
        for line in f:
            qid, _, docid, rank, score, tag = line.split()
            run[qid].append(docid)

    return run


def precision_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    hits = sum(1 for d in retrieved_k if d in relevant)
    return hits / k


def average_precision(relevant, retrieved):
    hits = 0
    sum_prec = 0

    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            sum_prec += hits / i

    if len(relevant) == 0:
        return 0

    return sum_prec / len(relevant)


def evaluate(run_file):
    qrels = load_qrels()
    run = load_run(run_file)

    AP_scores = []
    P10_scores = []

    for qid in qrels:

        relevant = set(qrels[qid].keys())
        retrieved = run.get(qid, [])

        ap = average_precision(relevant, retrieved)
        p10 = precision_at_k(relevant, retrieved, 10)

        AP_scores.append(ap)
        P10_scores.append(p10)

    MAP = sum(AP_scores) / len(AP_scores)
    P10 = sum(P10_scores) / len(P10_scores)

    return MAP, P10


runs = {
    "baseline": os.path.join(RESULTS_DIR, "baseline_results.txt"),
    "MiniLM": os.path.join(RESULTS_DIR, "Results_rerank_bert.txt"),
    "MPNet": os.path.join(RESULTS_DIR, "Results_rerank_mpnet.txt"),
}

best_map = -1
best_file = None
best_name = None

for name, file in runs.items():
    MAP, P10 = evaluate(file)

    print(f"{name}")
    print(f"MAP  = {MAP:.4f}")
    print(f"P@10 = {P10:.4f}")
    print()

    if MAP > best_map:
        best_map = MAP
        best_file = file
        best_name = name


shutil.copy(best_file, "Results")

print("Best system:", best_name)
print("Results file created: Results")