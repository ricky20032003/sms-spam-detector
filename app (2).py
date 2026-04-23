"""
Streamlit Web App for SMS Spam Detection
Run: streamlit run app.py
"""

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #0f0f1a; color: #e0e0ff; }
    .main-title { font-size: 2.5rem; font-weight: 800; text-align: center;
                  background: linear-gradient(135deg, #6c63ff, #ff6584);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .result-spam { background: #ff000020; border: 2px solid #ff4444;
                   border-radius: 12px; padding: 1.2rem; text-align: center; }
    .result-ham  { background: #00ff0020; border: 2px solid #44ff88;
                   border-radius: 12px; padding: 1.2rem; text-align: center; }
    .metric-card { background: #1a1a2e; border-radius: 10px; padding: 1rem;
                   border: 1px solid #333366; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open('model/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\d+', 'NUM', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)

# ── UI ───────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🛡️ SMS Spam Detector</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888'>Powered by TF-IDF + Naive Bayes | UCI Dataset | ~96% Accuracy</p>", unsafe_allow_html=True)

st.divider()

model, vectorizer = load_model()

if model is None:
    st.warning("⚠️ Model not found. Please run `python spam_detector.py` first to train and save the model.")
else:
    user_input = st.text_area(
        "📱 Enter an SMS message to classify:",
        placeholder="e.g. Congratulations! You've won a FREE prize. Click here now!",
        height=130,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze = st.button("🔍 Analyze Message", use_container_width=True)

    if analyze and user_input.strip():
        clean  = preprocess_text(user_input)
        vec    = vectorizer.transform([clean])
        pred   = model.predict(vec)[0]
        prob   = model.predict_proba(vec)[0]
        label  = "SPAM" if pred == 1 else "HAM"

        st.divider()

        if label == "SPAM":
            st.markdown(f"""
            <div class="result-spam">
                <h2>🚨 SPAM DETECTED</h2>
                <p style='font-size:1.1rem'>This message is likely <b>spam</b> with <b>{prob[1]*100:.1f}%</b> confidence.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-ham">
                <h2>✅ SAFE MESSAGE (Ham)</h2>
                <p style='font-size:1.1rem'>This message appears <b>legitimate</b> with <b>{prob[0]*100:.1f}%</b> confidence.</p>
            </div>""", unsafe_allow_html=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Ham Probability",  f"{prob[0]*100:.1f}%")
        with c2:
            st.metric("Spam Probability", f"{prob[1]*100:.1f}%")

        st.progress(float(prob[1]), text="Spam likelihood")

    # Sidebar stats
    with st.sidebar:
        st.header("📊 Model Info")
        st.markdown("""
| Property | Value |
|----------|-------|
| Algorithm | Naive Bayes |
| Vectorizer | TF-IDF |
| Dataset | UCI SMS (5,574) |
| Accuracy | ~96% |
| Features | 5,000 |
| N-grams | (1, 2) |
        """)
        st.divider()
        st.subheader("🧪 Try These")
        examples = {
            "🚨 Spam Example": "WINNER! You've been selected for a £1,000 prize. Claim NOW at www.freeprize.com",
            "✅ Ham Example":  "Can we reschedule our 3pm call to tomorrow morning?",
        }
        for label, msg in examples.items():
            if st.button(label, use_container_width=True):
                st.session_state['sample'] = msg
