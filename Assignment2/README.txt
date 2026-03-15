Rebecca Giles (300288250)
Rafael Carballo (300283390)
Qingcheng Meng (300221769)
Mathias Bertrand (300113314)
Implemented baseline retrieval plus two neural re-ranking methods.
Model 1 (Sentence-BERT MiniLM / all-MiniLM-L6-v2) was implemented by Rebecca.
Model 2 (Sentence-Transformer MPNet / all-mpnet-base-v2) was implemented by Rafael.
Produce a file called Results with the results for all the test queries was implemented by Qingcheng.
MAP and P@10 evaluation scores for the baseline and neural re-ranking models and discussion by Mathias.

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

MAP AND P@10 SCORES
- TF-IDF Baseline 
        MAP: 0.5190 P@10: 0.0810
- Model 1 (Sentence-BERT MiniLM / all-MiniLM-L6-v2)
        MAP: 0.5929 P@10: 0.0877
- Model 2 (Sentence-Transformer MPNet / all-mpnet-base-v2)
        MAP: 0.6082 P@10: 0.0893

DISCUSSION
- The neural re-ranking models improved the retrieval performance compared to the TF-IDF baseline system.
- The TF-IDF baseline achieved MAP = 0.5190 and P@10 = 0.0810.
- Model 1 (Sentence-BERT MiniLM / all-MiniLM-L6-v2) improved the results, achieving MAP = 0.5929 and P@10 = 0.0877.
- Model 2 (Sentence-Transformer MPNet / all-mpnet-base-v2) achieved the best overall performance with MAP = 0.6082 and P@10 = 0.0893.
- Therefore, Model 2 (Sentence-Transformer MPNet / all-mpnet-base-v2) was selected as the final system.

QUERIES SNIPPET
0 Q0 10608397 1 0.0867 baseline
0 Q0 10607877 2 0.0820 baseline
0 Q0 28138927 3 0.0635 baseline
0 Q0 10931595 4 0.0631 baseline
0 Q0 13231899 5 0.0627 baseline
0 Q0 825728 6 0.0608 baseline
0 Q0 16939583 7 0.0596 baseline
0 Q0 40212412 8 0.0585 baseline
0 Q0 803312 9 0.0571 baseline
0 Q0 27049238 10 0.0510 baseline

0 Q0 17388232 1 0.3100 BERT-rerank
0 Q0 803312 2 0.2389 BERT-rerank
0 Q0 40212412 3 0.2315 BERT-rerank
0 Q0 8891333 4 0.2234 BERT-rerank
0 Q0 25404036 5 0.2064 BERT-rerank
0 Q0 4435369 6 0.1981 BERT-rerank
0 Q0 21874414 7 0.1957 BERT-rerank
0 Q0 825728 8 0.1934 BERT-rerank
0 Q0 2682251 9 0.1889 BERT-rerank
0 Q0 6863070 10 0.1877 BERT-rerank

0 Q0 10607877 1 0.4112 MPNet-rerank
0 Q0 40212412 2 0.4004 MPNet-rerank
0 Q0 17388232 3 0.3405 MPNet-rerank
0 Q0 25404036 4 0.2907 MPNet-rerank
0 Q0 1469751 5 0.2843 MPNet-rerank
0 Q0 17123657 6 0.2783 MPNet-rerank
0 Q0 6863070 7 0.2768 MPNet-rerank
0 Q0 20758340 8 0.2643 MPNet-rerank
0 Q0 3770726 9 0.2582 MPNet-rerank
0 Q0 16736872 10 0.2541 MPNet-rerank
