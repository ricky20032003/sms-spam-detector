"""
SMS Spam Detection using TF-IDF and Naive Bayes
Dataset: UCI SMS Spam Collection (5,574 messages)
Accuracy: ~96%
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


# ─── Preprocessing ───────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Clean and normalize SMS text."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)          # Remove special chars
    text = re.sub(r'\d+', 'NUM', text)                # Replace numbers
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)


# ─── Load & Prepare Data ──────────────────────────────────────────────────────

def load_data(filepath: str = 'data/spam.csv') -> pd.DataFrame:
    """Load the UCI SMS Spam Collection dataset."""
    df = pd.read_csv(filepath, encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df['clean_message'] = df['message'].apply(preprocess_text)
    print(f"Dataset loaded: {len(df)} messages | Spam: {df['label_num'].sum()} | Ham: {(df['label_num']==0).sum()}")
    return df


# ─── Train Model ─────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """Vectorize text and train Naive Bayes classifier."""
    X = df['clean_message']
    y = df['label_num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✅ Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, vectorizer


# ─── Save / Load ─────────────────────────────────────────────────────────────

def save_model(model, vectorizer, model_path='model/model.pkl', vec_path='model/vectorizer.pkl'):
    """Persist trained model and vectorizer."""
    import os
    os.makedirs('model', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"\n💾 Model saved to {model_path}")


def load_model(model_path='model/model.pkl', vec_path='model/vectorizer.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


# ─── Predict ─────────────────────────────────────────────────────────────────

def predict(text: str, model, vectorizer) -> dict:
    """Classify a single SMS message."""
    clean = preprocess_text(text)
    vec   = vectorizer.transform([clean])
    pred  = model.predict(vec)[0]
    prob  = model.predict_proba(vec)[0]
    return {
        'message':    text,
        'prediction': 'SPAM' if pred == 1 else 'HAM',
        'confidence': float(max(prob)),
        'spam_prob':  float(prob[1]),
        'ham_prob':   float(prob[0]),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = load_data()
    model, vectorizer = train_model(df)
    save_model(model, vectorizer)

    # Quick demo predictions
    samples = [
        "Congratulations! You've won a FREE iPhone. Click here to claim now!!!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account has been compromised. Call 0800-FREE now!",
        "Can you pick up some milk on your way home?",
    ]
    print("\n─── Sample Predictions ───")
    for msg in samples:
        result = predict(msg, model, vectorizer)
        icon = "🚨" if result['prediction'] == 'SPAM' else "✅"
        print(f"{icon} [{result['prediction']}] ({result['confidence']:.1%}) — {msg[:60]}...")
