
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


class WordCharFeatures(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        word_ngram=(1, 2),
        char_ngram=(3, 5),
        min_df=2
    ):
        self.word_ngram = word_ngram
        self.char_ngram = char_ngram
        self.min_df = min_df

    def fit(self, X, y=None):
        self.word = TfidfVectorizer(
            ngram_range=self.word_ngram,
            min_df=self.min_df
        )

        self.char = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.char_ngram,
            min_df=self.min_df
        )

        self.word.fit(X)
        self.char.fit(X)

        return self

    def transform(self, X):
        word_features = self.word.transform(X)
        char_features = self.char.transform(X)

        return hstack([
            word_features,
            char_features
        ])

