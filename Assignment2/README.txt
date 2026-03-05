Rebecca Giles (300288250)
Implemented the neural re-ranking using Sentence-BERT, integrated baseline results, outputed reranked results.

HOW TO RUN
1. install python packages
        pip3 install sentence-transformers scikit-learn nltk tqdm

2. download tokenizer   
       python3 -m nltk.downloader punkt

3. run baseline retrival system (produce top 100 document queries)
in files 'baseline_result.txtx'
        python3 baseline.py

4. run neural model 1 (output file in 'results_rerank_bert.txt')
        python3 neural_rerank.py


FUNCTIONALITY
- baseline retrival (tf-idf): loads corpus and queries, computes tf-idf vectors for all documents, computes cosine similarity between each query and all documents, select the top 100 documents per query, save reults in Results/baseline_results.txt
- neural re-ranking (BERT): load top-100 documents per query from baseline, encode queries and documents using pre-trained Sentence-BERT, computes cosine similarity between each query and document embeddings, re-rank top-100 documents based on neural similarity scores, save in Results/Results_rerank_bert.txt


ALGORITHM, DATA STRUCTURE, OPTIMIZATION
- TF-IDF baseline: used TDidfVectorizer form scikit-learn, a sparse matrix representation for efficiency, and cosine similarity to rank documents.
- Neural Reranking: sentence embeddings using SentenceTransformer, cosine similarity between query and document embeddings, and only top-100 baseline results are re-ranked to reduce computation
- Optimization: avoided usingfull corpus embedding with BERT by limiting to baseline top-100.