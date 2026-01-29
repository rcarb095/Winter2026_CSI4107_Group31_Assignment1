from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from preprocessing import iter_scifact_tokens


def build_inverted_index(
    token_stream: Iterable[Tuple[str, List[str]]]
) -> Dict[str, Dict[str, int]]:
    inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)

    for doc_id, tokens in token_stream:
        term_freq = defaultdict(int)

        for token in tokens:
            term_freq[token] += 1

        for term, freq in term_freq.items():
            inverted_index[term][doc_id] = freq

    return inverted_index

if __name__ == "__main__":
    print("STEP 2: Inverted index sample")

    index = build_inverted_index(iter_scifact_tokens())

    for term in list(index.keys())[:5]:
        print(term, index[term])
