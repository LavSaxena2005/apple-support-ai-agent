import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report


TRAIN_FILE = "data/processed/apple_support_training_labeled.csv"
GOLDEN_FILE = "evaluation/golden_set.csv"


print("Loading training data...")
train = pd.read_csv(TRAIN_FILE)

print("Loading golden evaluation set...")
test = pd.read_csv(GOLDEN_FILE)

print("Training examples:", len(train))
print("Golden examples:", len(test))


X_train = train["text"].fillna("")
y_train = train["intent"]

X_test = test["text"].fillna("")
y_test = test["intent"]


# --------------------------------------------------
# Majority Class Baseline
# --------------------------------------------------

majority_class = y_train.value_counts().idxmax()

majority_predictions = [majority_class] * len(y_test)

majority_accuracy = accuracy_score(
    y_test,
    majority_predictions
)

majority_f1 = f1_score(
    y_test,
    majority_predictions,
    average="macro",
    zero_division=0
)

print("\n==============================")
print("MAJORITY CLASS BASELINE")
print("==============================")

print("Majority class:", majority_class)
print("Accuracy:", round(majority_accuracy, 4))
print("Macro-F1:", round(majority_f1, 4))


# --------------------------------------------------
# TF-IDF + Logistic Regression
# --------------------------------------------------

print("\nTraining TF-IDF + Logistic Regression...")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


model.fit(X_train, y_train)

predictions = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)


print("\n==============================")
print("TF-IDF + LOGISTIC REGRESSION")
print("==============================")

print("Accuracy:", round(accuracy, 4))
print("Macro-F1:", round(macro_f1, 4))


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)