import os
import re
import joblib
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# -----------------------------
# Download NLTK resources
# -----------------------------
nltk.download("stopwords")
nltk.download("wordnet")

# -----------------------------
# Project folder path
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")

MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")

VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

CONFUSION_MATRIX_PATH = os.path.join(
    BASE_DIR,
    "confusion_matrix.png",
)

# -----------------------------
# Check dataset exists
# -----------------------------
if not os.path.exists(DATASET_PATH):

    print("\nERROR: dataset.csv not found")

    print("\nExpected location:")

    print(DATASET_PATH)

    exit()

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATASET_PATH)

# -----------------------------
# Check required columns
# -----------------------------
required_columns = ["text", "sentiment"]

for col in required_columns:

    if col not in df.columns:

        print(
            f"\nERROR: Missing column '{col}'"
        )

        print(
            "\nYour dataset.csv must contain:"
        )

        print("text,sentiment")

        exit()

# -----------------------------
# Initialize NLP tools
# -----------------------------
lemmatizer = WordNetLemmatizer()

stop_words = set(
    stopwords.words("english")
)

# -----------------------------
# Text preprocessing
# -----------------------------
def preprocess_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+",
        "",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -----------------------------
# Clean text
# -----------------------------
df["clean_text"] = (
    df["text"]
    .astype(str)
    .apply(preprocess_text)
)

# -----------------------------
# Features
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words="english"
)
X = vectorizer.fit_transform(
    df["clean_text"]
)

y = df["sentiment"]

# -----------------------------
# Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train model
# -----------------------------
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(
    X_test
)

# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")

print(
    round(accuracy * 100, 2),
    "%"
)

print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)

# -----------------------------
# Save model files
# -----------------------------
joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

print(
    "\nModel saved successfully."
)

# -----------------------------
# Create confusion matrix
# -----------------------------
cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

fig, ax = plt.subplots(
    figsize=(7, 7)
)

disp.plot(ax=ax)

plt.title(
    "Sentiment Analysis Confusion Matrix"
)

plt.savefig(
    CONFUSION_MATRIX_PATH
)

plt.close()

print(
    "\nConfusion Matrix saved successfully."
)

print(
    "\nProject Files:"
)

print(MODEL_PATH)

print(VECTORIZER_PATH)

print(CONFUSION_MATRIX_PATH)
