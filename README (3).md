# 🛡️ SMS Spam Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-TF--IDF-orange?logo=scikitlearn)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A machine learning model that classifies SMS messages as **spam** or **ham (legitimate)** using **TF-IDF vectorization** and **Naive Bayes** classification, achieving **~96% accuracy** on the UCI SMS Spam Collection dataset of **5,574 messages**.

---

## 📸 Demo

> Run locally with Streamlit — dark-themed interactive web UI with real-time predictions.

---

## ✨ Features

- ✅ **96% accuracy** on UCI SMS Spam Collection (5,574 messages)
- 🔤 Full NLP preprocessing pipeline (tokenization, stopword removal, TF-IDF)
- 📊 Confidence scores and probability breakdown for each prediction
- 🌐 Interactive **Streamlit web app** for real-time classification
- 💾 Trained model saved via Pickle for instant reuse
- 🧪 Sample messages included for quick testing

---

## 🗂️ Project Structure

```
sms-spam-detector/
│
├── spam_detector.py        # Core ML pipeline (train, evaluate, predict)
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
│
├── data/
│   └── spam.csv            # UCI SMS Spam Collection dataset
│
├── model/
│   ├── model.pkl           # Saved Naive Bayes model (generated after training)
│   └── vectorizer.pkl      # Saved TF-IDF vectorizer (generated after training)
│
├── notebooks/
│   └── EDA_and_Training.ipynb   # Jupyter Notebook with EDA + model training
│
└── README.md
```

---

## 🧠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| Pandas | Data loading & manipulation |
| NumPy | Numerical operations |
| Scikit-learn | TF-IDF, Naive Bayes, metrics |
| NLTK | Tokenization, stopword removal |
| Streamlit | Web app interface |
| Pickle | Model serialization |
| Jupyter Notebook | EDA & experimentation |

---

## ⚙️ How It Works

```
Raw SMS → Lowercase → Remove special chars → Tokenize
        → Remove stopwords → TF-IDF Vectorization
        → Multinomial Naive Bayes → SPAM / HAM
```

### NLP Preprocessing Steps
1. **Lowercasing** — normalize case
2. **Special character removal** — strip punctuation & symbols
3. **Number normalization** — replace digits with `NUM` token
4. **Tokenization** — split into individual words (NLTK)
5. **Stopword removal** — filter common English words
6. **TF-IDF Vectorization** — 5,000 features, bigrams (1,2)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/sms-spam-detector.git
cd sms-spam-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Get the [UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place `spam.csv` inside the `data/` folder.

### 4. Train the model
```bash
python spam_detector.py
```
This trains the model and saves `model.pkl` and `vectorizer.pkl` to the `model/` directory.

### 5. Launch the web app
```bash
streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **96.1%** |
| Precision (Spam) | 97.3% |
| Recall (Spam) | 91.8% |
| F1-Score (Spam) | 94.5% |

---

## 📦 requirements.txt contents

See `requirements.txt` for the full list of dependencies.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 🙋‍♂️ Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)
