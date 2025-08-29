# -*- coding: utf-8 -*-
"""Sentence Extraction using the TextRank algorithm."""

import re
import numpy as np
from nltk import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance


MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)


def normalize_whitespace(text: str) -> str:
    """Translate multiple whitespace into a single space character."""

    return MULTIPLE_WHITESPACE_PATTERN.sub(_replace_whitespace, text)


def _replace_whitespace(match: re.Match) -> str:
    text = match.group()

    if "\n" in text or "\r" in text:
        return "\n"
    else:
        return " "


def is_blank(string: str) -> bool:
    """Return ``True`` if string contains only white-space characters."""

    return not string or string.isspace()


def get_symmetric_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric version of ``matrix``."""

    return matrix + matrix.T - np.diag(matrix.diagonal())


def core_cosine_similarity(vector1: list, vector2: list) -> float:
    """Measure cosine similarity between two vectors."""

    return 1 - cosine_distance(vector1, vector2)


"""Note: This is not a summarization algorithm. This algorithm picks top
sentences irrespective of the order they appeared."""


class TextRank4Sentences:
    def __init__(self) -> None:
        self.damping = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 100  # iteration steps
        self.text_str = None
        self.sentences = None
        self.pr_vector = None

    def _sentence_similarity(self, sent1, sent2, stopwords=None) -> float:
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return core_cosine_similarity(vector1, vector2)

    def _build_similarity_matrix(self, sentences, stopwords=None) -> np.ndarray:
        # create an empty similarity matrix
        sm = np.zeros([len(sentences), len(sentences)])

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    continue

                sm[idx1][idx2] = self._sentence_similarity(
                    sentences[idx1], sentences[idx2], stopwords=stopwords
                )

        # Get Symmetric matrix
        sm = get_symmetric_matrix(sm)

        # Normalize matrix by column
        norm = np.sum(sm, axis=0)
        sm_norm = np.divide(sm, norm, where=norm != 0)  # ignore zero element

        return sm_norm

    def _run_page_rank(self, similarity_matrix: np.ndarray) -> np.ndarray:
        pr_vector = np.array([1] * len(similarity_matrix))

        # Iteration
        previous_pr = 0
        for _ in range(self.steps):
            pr_vector = (
                (1 - self.damping)
                + self.damping * np.matmul(similarity_matrix, pr_vector)
            )
            if abs(previous_pr - sum(pr_vector)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr_vector)

        return pr_vector

    def _get_sentence(self, index: int) -> str:
        try:
            return self.sentences[index]
        except IndexError:
            return ""

    def get_top_sentences(self, number: int = 5) -> list:
        top_sentences = []

        if self.pr_vector is not None:
            sorted_pr = np.argsort(self.pr_vector)
            sorted_pr = list(sorted_pr)
            sorted_pr.reverse()

            index = 0
            for _ in range(number):
                sent = self.sentences[sorted_pr[index]]
                sent = normalize_whitespace(sent)
                top_sentences.append(sent)
                index += 1

        return top_sentences

    def analyze(self, text: str, stop_words=None) -> None:
        self.text_str = text
        self.sentences = sent_tokenize(self.text_str)

        tokenized_sentences = [word_tokenize(sent) for sent in self.sentences]

        similarity_matrix = self._build_similarity_matrix(
            tokenized_sentences, stop_words
        )

        self.pr_vector = self._run_page_rank(similarity_matrix)


def main() -> None:
    import nltk
    import spacy
    from textrank4zh import TextRank4Sentence

    nltk.download("punkt")
    nlp = spacy.load("en_core_web_sm")

    text_str = (
        """
    Mr. President, I offer you our congratulations on your election as the President of the current session of the General Assembly.
    You represent Norway, a country which can take pride in its reputation as peaceful, just and progressive.
    Your personal qualifications and your family's dedication to international effort are well known.
    I should also like to express our appreciation of the services of your distinguished predecessor, Mrs. Angie Brooks Randolph.
    I would also repeat our admiration for U Thant, whose skill and dedication have won him our respect
    """
    )

    doc = nlp(text_str)
    for token in doc:
        print(token.text, "->", token.pos_)

    tr4s = TextRank4Sentence()
    tr4s.analyze(text_str, lower=True, source="all_filters")
    print(tr4s.get_key_sentences(num=5))


if __name__ == "__main__":
    main()

