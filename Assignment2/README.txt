Rebecca Giles (300288250)
Rafael Carballo (300283390)
Qingcheng Meng (300221769)
Implemented baseline retrieval plus two neural re-ranking methods.
Model 1 (Sentence-BERT MiniLM / all-MiniLM-L6-v2) was implemented by Rebecca.
Model 2 (Sentence-Transformer MPNet / all-mpnet-base-v2) was implemented by Rafael.
Produce a file called Results with the results for all the test queries was implemented by Qingcheng.

HOW TO RUN
Run all commands from the `Assignment2` directory.

1. install python packages
        pip install sentence-transformers scikit-learn nltk tqdm

2. download tokenizer
        python -m nltk.downloader punkt

3. run baseline retrieval system (top-100 documents per query)
        python src/baseline.py
   output:
        src/Results/baseline_results.txt

4. run neural model 1 (Sentence-BERT MiniLM)
        python src/neural_rerank.py
   output:
        src/Results/Results_rerank_bert.txt

5. run neural model 2 (Sentence-Transformer MPNet)
        python src/neural_rerank_mpnet.py
   output:
        src/Results/Results_rerank_mpnet.txt

Note: GPU is optional. The code automatically uses CUDA when available and falls back to CPU otherwise.

FUNCTIONALITY
- baseline retrieval (tf-idf): loads corpus and queries, computes tf-idf vectors for all documents, computes cosine similarity between each query and all documents, selects top 100 documents per query, saves results in src/Results/baseline_results.txt
- neural re-ranking model 1 (all-MiniLM-L6-v2): loads top-100 docs per query from baseline, encodes queries/documents using SentenceTransformer, computes cosine similarity, re-ranks and saves to src/Results/Results_rerank_bert.txt
- neural re-ranking model 2 (all-mpnet-base-v2): same pipeline with a different neural embedding model, saves to src/Results/Results_rerank_mpnet.txt

ALGORITHM, DATA STRUCTURE, OPTIMIZATION
- TF-IDF baseline: TfidfVectorizer from scikit-learn with sparse matrix cosine similarity for efficient first-stage retrieval.
- Neural re-ranking: dense sentence embeddings + cosine similarity for semantic matching.
- Optimization: only top-100 baseline documents are re-ranked, avoiding full-corpus neural scoring.
