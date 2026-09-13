import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


DATA = "evaluation/golden_set_200.csv"

df = pd.read_csv(DATA)

df = df[
    df["intent"].notna()
    & (df["intent"].astype(str).str.strip() != "")
].copy()

print("Total labeled examples:", len(df))

X = df["text"]
y = df["intent"]


# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training examples:", len(X_train))
print("Test examples:", len(X_test))


# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_features=10000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Logistic Regression
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)


accuracy = accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro"
)


print("\n" + "=" * 60)
print("TF-IDF + LOGISTIC REGRESSION")
print("=" * 60)

print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)