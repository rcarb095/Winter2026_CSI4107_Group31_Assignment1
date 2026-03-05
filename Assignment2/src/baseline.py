import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load corpus and queries
with open("../data/scifact/corpus.jsonl") as f:
    corpus = [json.loads(line) for line in f]
docs = [doc["text"] for doc in corpus]
doc_ids = [doc["_id"] for doc in corpus]

with open("../data/scifact/queries.jsonl") as f:
    queries = [json.loads(line) for line in f]

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
doc_vectors = vectorizer.fit_transform(docs)

# Compute top-100 per query
Results = []
for q in queries:
    q_text = q["text"]
    q_id = q["_id"]
    q_vec = vectorizer.transform([q_text])
    sim_scores = cosine_similarity(q_vec, doc_vectors).flatten()
    top_idx = sim_scores.argsort()[::-1][:100]  # top-100 docs
    for rank, idx in enumerate(top_idx, 1):
        Results.append(f"{q_id} Q0 {doc_ids[idx]} {rank} {sim_scores[idx]:.4f} baseline")

# Save results
os.makedirs("Results", exist_ok=True)
with open("Results/baseline_results.txt", "w") as f:
    f.write("\n".join(Results))

print("Baseline results saved to Results/baseline_results.txt")