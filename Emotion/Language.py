
# ============================================================
# PROJECT: Multilingual Emotion Detection with Sparse Coding
# Languages: English and Yoruba
# Models: Dense XLM-R vs XLM-R + Sparse Coding
# ============================================================

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


# 1. SETTINGS
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET = "vgaraujov/semeval-2025-task11-track-a"
MODEL_NAME = "FacebookAI/xlm-roberta-base"

EMOTIONS = [
    "anger", "disgust", "fear",
    "joy", "sadness", "surprise"
]

print("Device:", DEVICE)


# 2. LOAD THE DATA
def load_language(language):
    train = load_dataset(
        DATASET, language, split="train"
    ).to_pandas()

    dev = load_dataset(
        DATASET, language, split="dev"
    ).to_pandas()

    train["language"] = language
    dev["language"] = language

    return train, dev


en_train, en_dev = load_language("eng")
yo_train, yo_dev = load_language("yor")

train_df = pd.concat(
    [en_train, yo_train], ignore_index=True
)

dev_df = pd.concat(
    [en_dev, yo_dev], ignore_index=True
)

# Check required fields
for name, df in [("train", train_df), ("dev", dev_df)]:
    required = ["text", "emotions"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

# Remove records without text or emotion annotations
train_df = train_df.dropna(
    subset=["text", "emotions"]
).copy()

dev_df = dev_df.dropna(
    subset=["text", "emotions"]
).copy()


# 3. CONVERT EMOTIONS INTO SIX BINARY LABELS
def make_labels(df):
    labels = []

    for value in df["emotions"]:
        if isinstance(value, np.ndarray):
            emotions = value.tolist()
        elif isinstance(value, (list, tuple)):
            emotions = list(value)
        else:
            raise ValueError(
                f"Unexpected emotion annotation: {value!r}"
            )

        unknown = set(emotions) - set(EMOTIONS)

        if unknown:
            raise ValueError(f"Unknown labels: {unknown}")

        labels.append([
            int(emotion in emotions)
            for emotion in EMOTIONS
        ])

    return np.asarray(labels, dtype=np.int32)


Y_train = make_labels(train_df)
Y_dev = make_labels(dev_df)

train_texts = train_df["text"].astype(str).tolist()
dev_texts = dev_df["text"].astype(str).tolist()

print("Training samples:", len(train_texts))
print("Development samples:", len(dev_texts))
print("Number of emotion labels:", len(EMOTIONS))


# 4. LOAD XLM-R AND EXTRACT TEXT EMBEDDINGS
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
encoder.eval()


def get_embeddings(texts, batch_size=4):
    all_vectors = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.inference_mode():
            output = encoder(**inputs).last_hidden_state

            mask = inputs["attention_mask"].unsqueeze(-1)

            vectors = (
                (output * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1)
            )

        all_vectors.append(vectors.cpu().numpy())

    return np.concatenate(all_vectors, axis=0)


X_train = get_embeddings(train_texts)
X_dev = get_embeddings(dev_texts)

print("Training embedding shape:", X_train.shape)
print("Development embedding shape:", X_dev.shape)


# 5. DEFINE THE CLASSIFIER AND EVALUATION
def create_classifier():
    return OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            random_state=SEED
        )
    )


def evaluate(y_true, predictions):
    return f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )


def train_and_predict(Xtr, Xdv):
    model = create_classifier()
    model.fit(Xtr, Y_train)
    return model.predict(Xdv)


# 6. BASELINE: DENSE XLM-R EMBEDDINGS
print("\nTraining baseline model...")

dense_predictions = train_and_predict(X_train, X_dev)

dense_overall = evaluate(Y_dev, dense_predictions)

print("Dense XLM-R Macro-F1:", round(dense_overall, 4))


# 7. PROPOSED METHOD: XLM-R + SPARSE CODING
print("\nLearning sparse representations...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_dev_scaled = scaler.transform(X_dev)


dictionary = MiniBatchDictionaryLearning(
    n_components=128,
    alpha=2.0,
    max_iter=100,
    batch_size=128,
    transform_algorithm="lasso_cd",
    transform_max_iter=2000,
    transform_alpha=2.0,
    random_state=SEED
)


dictionary.fit(X_train_scaled)

Z_train = dictionary.transform(X_train_scaled)
Z_dev = dictionary.transform(X_dev_scaled)

sparse_predictions = train_and_predict(Z_train, Z_dev)

sparse_overall = evaluate(Y_dev, sparse_predictions)

print("Sparse XLM-R Macro-F1:", round(sparse_overall, 4))


# 8. EVALUATE EACH LANGUAGE SEPARATELY
language_results = []

dev_languages = dev_df["language"].to_numpy()

for language in ["eng", "yor"]:
    mask = dev_languages == language

    dense_score = evaluate(
        Y_dev[mask], dense_predictions[mask]
    )

    sparse_score = evaluate(
        Y_dev[mask], sparse_predictions[mask]
    )

    language_results.append({
        "Language": language,
        "Dense XLM-R Macro-F1": round(dense_score, 4),
        "Sparse XLM-R Macro-F1": round(sparse_score, 4),
        "Difference": round(sparse_score - dense_score, 4)
    })


# 9. DISPLAY AND SAVE RESULTS
results = pd.DataFrame(language_results)

print("\nResults by language:")
print(results.to_string(index=False))


print("\nCombined development-set results:")
print("Dense XLM-R:", round(dense_overall, 4))
print("Sparse XLM-R:", round(sparse_overall, 4))
print(
    "Difference:",
    round(sparse_overall - dense_overall, 4)
)

results.to_csv("emotion_results.csv", index=False)
print("\nResults saved to emotion_results.csv")
