import pandas as pd
import numpy as np
import re
import html
import string

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# 1. CLEAN TEXT FUNCTION
# =========================
def clean_text(text):
    if pd.isna(text):
        return ""

    text = html.unescape(str(text))
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# =========================
# 2. LOAD DATASETS
# =========================
files = {
    "CEAS_08.csv": 1,
    "Ling.csv": 1,
    "Nazario.csv": 1,
    "Nigerian_Fraud.csv": 1,
    "SpamAssasin.csv": 0,
    "Enron.csv": 0,
    "phishing_email.csv": 1
}

dataframes = []


# =========================
# 3. PROCESS EACH FILE
# =========================
for file, label in files.items():
    try:
        df = pd.read_csv(file, encoding="latin-1")

        # detect text column
        possible_cols = ["text", "body", "content", "email", "message"]
        text_col = None

        for col in possible_cols:
            if col in df.columns:
                text_col = col
                break

        if text_col is None:
            text_col = df.columns[0]

        df = df[[text_col]].copy()
        df.columns = ["text"]

        df["text"] = df["text"].apply(clean_text)
        df["label"] = label

        dataframes.append(df)

        print(f"Loaded {file}")

    except Exception as e:
        print(f"Error loading {file}: {e}")


# =========================
# 4. MERGE DATASETS
# =========================
data = pd.concat(dataframes, ignore_index=True)

data = data.dropna()
data = data[data["text"].str.len() > 10]

print("\nFinal dataset size:", data.shape)


# =========================
# 5. SPLIT DATA
# =========================

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 6. TEXT → NUMBERS
# =========================
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# =========================
# 7. RANDOM FOREST MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_vec, y_train)


# =========================
# 8. EVALUATION
# =========================
y_pred = model.predict(X_test_vec)

print("\n================ RESULTS ================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# =========================
# 9. TEST YOUR OWN EMAIL
# =========================
def predict_email(email_text):
    cleaned = clean_text(email_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "🚨 Phishing Email"
    else:
        return "✅ Safe Email"


# TEST EXAMPLE
sample_email = """
Dear user,
Your bank account is suspended.
Click the link immediately to verify.
"""

print("\nTEST EMAIL RESULT:")
print(predict_email(sample_email))
