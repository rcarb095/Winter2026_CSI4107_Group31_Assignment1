Group 31 - Assignment 1

Students and task division:
Rafael Carballo Maduro (Step 1: Preprocessing)
Rebecca Giles (Step 2: Indexing)
Qingcheng Meng (Step 3: Retrieval)
Mathias Bertrand (Testing and README)

Program functionality:

The preprocessing.py file is responsible for preparing the text before indexing and retrieval. 
It cleans the raw text by removing simple markup, converting everything to lowercase, tokenizing the text with NLTK, and filtering out tokens that do not contain letters (which removes punctuation and numbers).
Stopwords are removed using a stopword list, and the remaining tokens are stemmed using the Porter stemmer.
The SciFact documents are read from corpus.jsonl using generators, so they can be processed one by one instead of loading the entire dataset into memory at once.

The indexing.py file indexing.py constructs the inverted index by mapping each term to the list of documents that contain it, along with term frequencies. 
This data structure allows fast access to candidate documents during retrieval.

The retrieval.py file ranks documents using TF-IDF and cosine similarity. Queries are preprocessed the same way as documents, 
and similarity scores are computed using the inverted index to efficiently score only documents that share terms with the query.
The top 100 ranked documents are returned for each query.

Run instructions:

Have the required dependacies (NLTK) and the SciFact files.
Run results.py to run the system and generate the Result files.

Vocabulary size: 37,887
Sample of vocabulary (100):
['microstructur', 'develop', 'human', 'newborn', 'cerebr', 'white', 'matter', 'assess', 'vivo', 'diffus',
 'tensor', 'magnet', 'reson', 'imag', 'alter', 'architectur', 'brain', 'affect', 'cortic', 'result', 'function', 'disabl',
  'line', 'scan', 'diffusion-weight', 'mri', 'sequenc', 'analysi', 'appli', 'measur', 'appar', 'coeffici', 'calcul', 'rel',
   'anisotropi', 'delin', 'three-dimension', 'fiber', 'preterm', 'full-term', 'infant', 'effect', 'prematur', 'earli',
    'gestat', 'studi', 'term', 'central', 'mean', 'wk', 'microm2/m', 'decreas', 'posterior', 'limb', 'intern', 'capsul',
     'versu', 'closer', 'birth', 'absolut', 'valu', 'area', 'compar', 'nonmyelin', 'corpu', 'callosum', 'visibl', 'mark',
      'differ', 'organ', 'data', 'indic', 'quantit', 'water', 'insight', 'live', 'induct', 'myelodysplasia', 'myeloid-deriv',
       'suppressor', 'cell', 'myelodysplast', 'syndrom', 'md', 'age-depend', 'stem', 'malign', 'share', 'biolog', 'featur',
        'activ', 'adapt', 'immun', 'respons', 'ineffect', 'hematopoiesi', 'report', 'mdsc', 'classic', 'link']

First 10 results for Query 1:
1 Q0 13231899 1 0.097666 tfidf_text
1 Q0 42421723 2 0.081973 tfidf_text
1 Q0 35008773 3 0.074447 tfidf_text
1 Q0 994800 4 0.069006 tfidf_text
1 Q0 12156187 5 0.062602 tfidf_text
1 Q0 7581911 6 0.058235 tfidf_text
1 Q0 18953920 7 0.056701 tfidf_text
1 Q0 10786948 8 0.054142 tfidf_text
1 Q0 21257564 9 0.054065 tfidf_text
1 Q0 8185080 10 0.052331 tfidf_text

First 10 results for Query 3:
3 Q0 23389795 1 0.441085 tfidf_text
3 Q0 2739854 2 0.352551 tfidf_text
3 Q0 4632921 3 0.268610 tfidf_text
3 Q0 14717500 4 0.262711 tfidf_text
3 Q0 4378885 5 0.233146 tfidf_text
3 Q0 4414547 6 0.184474 tfidf_text
3 Q0 32181055 7 0.175558 tfidf_text
3 Q0 19058822 8 0.175550 tfidf_text
3 Q0 8411251 9 0.164619 tfidf_text
3 Q0 12271486 10 0.151876 tfidf_text

Mean Average Precision (MAP) for best run (titles and full text): 0.5329

Discussion:

Our system achieved a MAP of 0.5329 when using the full query text, which shows that the TF-IDF vector space model works reasonably well on the SciFact dataset. For the run where only the titles were used, the performance dropped to 0.4415, which suggests that having more context in the query helps the system retrieve better results. Overall, TF-IDF with cosine similarity gives decent rankings, but it mainly relies on exact word matching and does not handle semantic similarities between different words very well.
