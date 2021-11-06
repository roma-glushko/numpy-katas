from collections import Counter
from typing import List

import numpy as np

# TF-IDF encoding
# https://en.wikipedia.org/wiki/Tf%E2%80%93idf
# https://monkeylearn.com/blog/what-is-tf-idf/


class TFIDFEncoder:
    def get_document_terms(self, documents) -> List[List[str]]:
        processed_documents: List[List[str]] = []

        for document in documents:
            # collecting document- and corpus-wise statistics
            document = document.lower()
            terms: List[str] = document.split(" ")

            processed_documents.append(terms)

        return processed_documents

    def transform(self, documents: List[str]) -> np.ndarray:
        document_terms: List[Counter] = []
        document_frequency: Counter = Counter()

        documents: List[List[str]] = self.get_document_terms(documents)

        for terms in documents:
            document_frequency.update(set(terms))
            document_terms.append(Counter(terms))

        # encoding
        encoded_docs: List[List[float]] = []
        # num_docs: int = len(documents)

        # for document_id, terms in enumerate(documents):
        #     num_terms_in_doc: int = len(terms)
        #     document_term_frequency: Counter =
        #     encoded_doc: List[float] = []
        #
        #     for term in terms:
        #
        #         term_frequency: float =
        #
        #     encoded_docs.append(encoded_doc)

        return np.array(encoded_docs)


if __name__ == "__main__":
    documents: List[str] = [
        "This is a sample of a real world sentence",
        "This is an example of TFIDF implementation",
        "Example of sentence sample",
    ]

    tfidf_encoder: TFIDFEncoder = TFIDFEncoder()

    encoded_docs = tfidf_encoder.transform(documents)
