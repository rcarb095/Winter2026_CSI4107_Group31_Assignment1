import json
from sentence_transformers import SentenceTransformer, util
import os

# Load pre-trained BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load corpus and queries
with open("../data/scifact/corpus.jsonl") as f:
    corpus = {json.loads(line)["_id"]: json.loads(line)["text"] for line in f}

with open("../data/scifact/queries.jsonl") as f:
    queries = {json.loads(line)["_id"]: json.loads(line)["text"] for line in f}

# Load baseline results
baseline_file = "Results/baseline_results.txt"
top_docs_per_query = {}
with open(baseline_file) as f:
    for line in f:
        qid, _, docid, rank, score, method = line.strip().split()
        top_docs_per_query.setdefault(qid, []).append(docid)

# Rerank using Sentence-BERT
Results = []
for qid, docids in top_docs_per_query.items():
    q_text = queries[qid]
    doc_texts = [corpus[d] for d in docids]

    # Compute embeddings
    q_emb = model.encode(q_text, convert_to_tensor=True)
    doc_embs = model.encode(doc_texts, convert_to_tensor=True)

    # Cosine similarity
    sim_scores = util.cos_sim(q_emb, doc_embs).flatten()
    ranked_idx = sim_scores.argsort(descending=True)

    for rank, idx in enumerate(ranked_idx, 1):
        Results.append(f"{qid} Q0 {docids[idx]} {rank} {sim_scores[idx]:.4f} BERT-rerank")

# Save reranked results
os.makedirs("Results", exist_ok=True)
with open("Results/Results_rerank_bert.txt", "w") as f:
    f.write("\n".join(Results))

print("BERT reranked results saved to Results/Results_rerank_bert.txt")