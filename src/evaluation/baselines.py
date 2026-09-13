import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def majority_baseline(y_train, y_test):
    majority = y_train.value_counts().idxmax()
    return [majority] * len(y_test)


def build_tfidf_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000
        )),
        ("clf", LogisticRegression(max_iter=1000))
    ])


# This module is intentionally generic. Connect it to the manually labelled
# golden set after the brand-specific intent taxonomy has been created.