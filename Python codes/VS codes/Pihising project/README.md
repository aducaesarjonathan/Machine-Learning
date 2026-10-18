# Phishing Email Detection Using Machine Learning

## Overview

This project develops a supervised machine learning model to classify email messages as phishing or legitimate using natural language processing (NLP) and text classification techniques.

The pipeline combines email datasets from multiple sources, cleans and preprocesses email text, converts messages into numerical features using Term Frequency–Inverse Document Frequency (TF-IDF), and trains a Random Forest classifier to distinguish potentially malicious emails from legitimate messages.

The project also evaluates classification performance using accuracy, precision, recall, and F1-score, and includes a function for testing custom email messages.

## Objectives

- Develop a machine learning model for phishing email detection.
- Combine email datasets from multiple sources into a unified dataset.
- Apply text cleaning and preprocessing techniques.
- Extract numerical text features using TF-IDF.
- Train a Random Forest classifier for binary classification.
- Evaluate model performance using standard classification metrics.
- Test the trained model on custom email messages.

## Dataset

The project combines seven CSV datasets representing phishing and legitimate emails.

| Dataset | Assigned label |
|---|---|
| `CEAS_08.csv` | Phishing (1) |
| `Ling.csv` | Phishing (1) |
| `Nazario.csv` | Phishing (1) |
| `Nigerian_Fraud.csv` | Phishing (1) |
| `phishing_email.csv` | Phishing (1) |
| `SpamAssasin.csv` | Legitimate (0) |
| `Enron.csv` | Legitimate (0) |

**Important:** These labels reflect the assignments in the current script. Before interpreting results, verify that each dataset contains the expected email category. Spam, legitimate email, and phishing are not interchangeable classes.

The script searches for a text column named `text`, `body`, `content`, `email`, or `message`. If none is found, it uses the first column.

## Methodology

### 1. Text preprocessing

The `clean_text()` function prepares email messages for classification by:

- Handling missing values.
- Decoding HTML entities.
- Converting text to lowercase.
- Removing URLs.
- Removing numeric characters.
- Removing punctuation.
- Normalizing whitespace.

Messages shorter than or equal to ten characters are excluded.

### 2. Dataset integration

The selected CSV files are loaded and their text columns standardized. Each dataset receives a binary label, and the resulting dataframes are concatenated into a unified dataset.

### 3. Train-test split

The data is split into training and test sets using an 80:20 ratio. Stratified sampling preserves the approximate class proportions, and `random_state=42` makes the split reproducible for the same input data.

### 4. TF-IDF feature extraction

The project uses `TfidfVectorizer` to convert email text into numerical features.

Configuration:

- Maximum vocabulary size: 5,000 features.
- English stop-word removal enabled.
- Vocabulary learned from the training data only.

TF-IDF assigns weights to words according to their importance within individual messages relative to the training corpus.

### 5. Random Forest classification

A Random Forest classifier is trained on the TF-IDF feature matrix.

| Parameter | Configuration |
|---|---|
| Algorithm | Random Forest |
| Number of trees | 200 |
| Random seed | 42 |
| Parallel processing | All available CPU cores |

The model predicts one of two classes: phishing (1) or legitimate (0).

### 6. Model evaluation

The model is evaluated on the held-out test set using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted phishing messages that are actually labelled phishing.
- **Recall:** Proportion of labelled phishing messages correctly detected.
- **F1-score:** Harmonic mean of precision and recall.

The classification report provides class-specific results.

### 7. Custom email prediction

The `predict_email()` function accepts a new email message, applies the same text-cleaning and TF-IDF transformations, and returns the predicted class as either `Phishing Email` or `Safe Email`.

A sample message about a suspended bank account is included to demonstrate the prediction workflow.

The output is a model prediction, not a guarantee that an email is safe or malicious.

## Technologies Used

- **Python** — Core programming language
- **Pandas** — Data loading and manipulation
- **NumPy** — Numerical operations
- **Scikit-learn** — Feature extraction, classification, data splitting, and evaluation
- **Regular expressions** — Text cleaning
- **HTML utilities** — HTML entity decoding

## Project Structure

```text
phishing-email-detection/
├── README.md
├── phishing_detection.py
├── CEAS_08.csv
├── Ling.csv
├── Nazario.csv
├── Nigerian_Fraud.csv
├── SpamAssasin.csv
├── Enron.csv
└── phishing_email.csv
```

The CSV files are expected by the current script. If you publish the project, follow each dataset's licensing and redistribution requirements; alternatively, document the download sources and keep the data files out of the repository.

## Installation

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn
```

## Usage

1. Place the required CSV datasets in the project directory.
2. Ensure each dataset contains the expected email text column.
3. Run the Python script:

```bash
python phishing_detection.py
```

The script loads and combines the datasets, preprocesses the text, trains the classifier, prints the evaluation metrics, and tests the sample email.

## Example

Sample input:

```text
Dear user,
Your bank account is suspended.
Click the link immediately to verify.
```

The model returns either:

- `Phishing Email`
- `Safe Email`

The actual prediction depends on the trained model and the contents of the supplied datasets.

## Limitations

- **Dataset labels:** Incorrect or overly broad labels can compromise model quality.
- **Dataset leakage:** Similar or duplicated messages across datasets may appear in both training and test sets, inflating performance estimates.
- **Source bias:** The classifier may learn dataset-specific patterns instead of general phishing characteristics.
- **Feature limitations:** Removing URLs, digits, and punctuation can discard useful phishing indicators.
- **Class definition:** Spam and phishing are different categories; treating all non-phishing emails as legitimate can introduce label noise.
- **Generalization:** A model trained on historical datasets may not detect new phishing techniques.
- **Security:** A legitimate prediction does not establish that an email is trustworthy.

## Future Improvements

- Deduplicate messages and investigate dataset overlap.
- Verify labels and distinguish phishing, spam, and legitimate email.
- Evaluate generalization using source-based or time-based test splits.
- Compare Random Forest with Logistic Regression, Naive Bayes, and linear SVM.
- Experiment with word and character n-grams.
- Compare TF-IDF with transformer-based text representations.
- Analyze false positives and false negatives.
- Develop a simple interface for email classification.
- Evaluate precision-recall trade-offs for security-sensitive use cases.

## Conclusion

This project demonstrates a traditional machine learning approach to phishing email detection by combining text preprocessing, TF-IDF feature extraction, and Random Forest classification.

It provides a foundation for exploring NLP-based security applications and evaluating the strengths and limitations of supervised text classification.

## Author

**Jonathan Adu-Caesar**

BSc Mathematics, Kwame Nkrumah University of Science and Technology (KNUST).

My research interests include machine learning, deep learning, representation learning, optimization, and natural language processing (NLP).
