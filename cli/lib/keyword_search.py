import json
import os
import pickle
import string
from math import log

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from search_utils import BM25_B, BM25_K1


def process_text(input_string):
    query = input_string.lower()
    query = remove_punctuation_translate(query)
    tokenized_query = query.split()
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in tokenized_query if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(words) for words in filtered_words]

    return stemmed_words


def remove_punctuation_translate(input_string):
    translator = str.maketrans("", "", string.punctuation)
    new_string = input_string.translate(translator)
    return new_string


class InvertedIndex:
    def __init__(self, index, docmap, term_frequencies) -> None:
        self.index = index
        self.docmap = docmap
        self.term_frequencies = term_frequencies
        self.doc_lengths = {}
        self.index_path = "cache/index.pkl"
        self.docmap_path = "cache/docmap.pkl"
        self.tf_path = "cache/term_frequencies.pkl"
        self.doc_lengths_path = "cache/doc_lengths.pkl"
        self.avg_docs_length = 0.0

    def __add_document(self, doc_id, text):
        tokens = process_text(text)
        counter = {}
        no_of_tokens = 0

        for token in tokens:
            no_of_tokens += 1
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

            counter[token] = counter.get(token, 0) + 1

        self.term_frequencies[doc_id] = counter
        self.doc_lengths[doc_id] = no_of_tokens

    def get_document(self, term):
        id_list = list(self.index[term.lower()])
        id_list.sort()
        return id_list

    def build(self):
        with open("data/movies.json", "r") as file:
            data = json.load(file)
            movies = data["movies"]

        for m in movies:
            text = f"{m['title']} {m['description']}"
            self.__add_document(m["id"], text)
            self.docmap[m["id"]] = m

    def save(self):
        os.makedirs("cache/", exist_ok=True)

        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)
        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)
        with open(self.tf_path, "wb") as file:
            pickle.dump(self.term_frequencies, file)
        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self):
        try:
            with open(self.index_path, "rb") as file:
                self.index = pickle.load(file)
            with open(self.docmap_path, "rb") as file:
                self.docmap = pickle.load(file)
            with open(self.tf_path, "rb") as file:
                self.term_frequencies = pickle.load(file)
            with open(self.doc_lengths_path, "rb") as file:
                self.doc_lengths = pickle.load(file)
            self.avg_docs_length = self.__get_avg_doc_length()
        except FileNotFoundError as e:
            print(f"File Not Found: {e}")

        return (self.index, self.docmap, self.term_frequencies, self.doc_lengths)

    def get_tf(self, doc_id, token):
        return self.term_frequencies.get(doc_id, {}).get(token, 0)

    def get_bm25_idf(self, token):
        df = len(self.index.get(token, []))
        N = len(self.docmap)
        return log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.avg_docs_length
        doc_length = self.doc_lengths[doc_id]

        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        bm25_satu = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return bm25_satu

    def __get_avg_doc_length(self) -> float:
        avg_doc_len = 0.0
        all_len = sum(self.doc_lengths.values())
        count = len(self.doc_lengths)

        if count == 0:
            return avg_doc_len

        avg_doc_len = float(all_len / count)

        return avg_doc_len

    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)

        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        query_tokens = process_text(query)
        scores = {}

        candidate_docs = set()
        for token in query_tokens:
            candidate_docs |= self.index.get(token, set())

        for doc_id in candidate_docs:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        return sorted_scores[:limit]
