import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import nltk
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

nltk.download("stopwords", quiet=True)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp { background-color: #0d0d0d; color: #e8e8e8; }

    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 48px 40px;
        margin-bottom: 32px;
        border: 1px solid #1e3a5f;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 8px 0;
    }
    .hero p {
        color: #8bafd4;
        font-size: 1.05rem;
        margin: 0;
    }

    .metric-card {
        background: #141414;
        border: 1px solid #222;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-card .value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #4fc3f7;
    }
    .metric-card .label {
        color: #888;
        font-size: 0.85rem;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .result-positive {
        background: linear-gradient(135deg, #0a2e1a, #0d3d24);
        border: 1px solid #1b6b3a;
        border-radius: 12px;
        padding: 24px 28px;
        margin-top: 16px;
    }
    .result-negative {
        background: linear-gradient(135deg, #2e0a0a, #3d0d0d);
        border: 1px solid #6b1b1b;
        border-radius: 12px;
        padding: 24px 28px;
        margin-top: 16px;
    }
    .result-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .result-positive .result-label { color: #4ade80; }
    .result-negative .result-label { color: #f87171; }
    .result-confidence {
        color: #aaa;
        font-size: 0.9rem;
        margin-top: 6px;
    }

    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #222;
    }

    .stTextArea textarea {
        background-color: #141414 !important;
        color: #e8e8e8 !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stButton > button {
        background: #1565c0;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        width: 100%;
    }
    .stButton > button:hover { background: #1976d2; }
    .stFileUploader { border: 1px dashed #333; border-radius: 8px; padding: 8px; }
    .stProgress > div > div { background: #4fc3f7; }
    div[data-testid="stSidebar"] { background-color: #111 !important; }
    .stSelectbox select, .stSlider { color: #e8e8e8; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()
MODEL_PATH = "/home/claude/sentiment_model.pkl"


def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)           # strip HTML tags
    text = re.sub(r"[^a-zA-Z]", " ", text)       # keep only letters
    text = text.lower().split()
    text = [stemmer.stem(w) for w in text if w not in STOP_WORDS]
    return " ".join(text)


@st.cache_resource(show_spinner=False)
def train_model(df):
    df = df.copy()
    df["cleaned"] = df["review"].astype(str).apply(clean_text)
    X = df["cleaned"]
    y = df["sentiment"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    pipeline = Pipeline([
        ("vect", TfidfVectorizer()),
        ("chi", SelectKBest(chi2, k=10000)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    y_pred = pipeline.predict(X_test)
    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=["positive", "negative"])
    dist = y.value_counts()

    return pipeline, train_acc, test_acc, report, cm, dist, len(df)


def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎬 IMDB Sentiment Analyzer</h1>
    <p>Train a Logistic Regression model on IMDB reviews · Explore model metrics · Predict any review</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Setup")
    st.markdown("Upload your `imdb.csv` to train the model.")
    uploaded = st.file_uploader("Upload IMDB CSV", type=["csv"])
    st.markdown("---")
    st.markdown("**Model:** Logistic Regression")
    st.markdown("**Features:** TF-IDF + Chi² (top 10k)")
    st.markdown("**Preprocessing:** Stopword removal + Porter Stemmer")
    st.markdown("---")
    st.markdown("Built by **Ragnar** · Broadway Infosys Final Project")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Train & Metrics", "🔍 Predict", "📖 About"])

# ── TAB 1 — Train & Metrics ───────────────────────────────────────────────────
with tab1:
    if uploaded is None:
        st.info("👈 Upload your IMDB CSV in the sidebar to get started.")
    else:
        try:
            df_raw = pd.read_csv(uploaded, encoding="latin1", engine="python",
                                 on_bad_lines="skip").astype(str)

            if "review" not in df_raw.columns or "sentiment" not in df_raw.columns:
                st.error("CSV must have `review` and `sentiment` columns.")
                st.stop()

            with st.spinner("Training model on 50,000 reviews... this takes ~60 seconds"):
                model, train_acc, test_acc, report, cm, dist, n_rows = train_model(df_raw)

            st.success("✅ Model trained successfully!")

            # ── Metrics row
            st.markdown('<p class="section-header">Model Performance</p>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="value">{test_acc*100:.1f}%</div>
                    <div class="label">Test Accuracy</div></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                    <div class="value">{train_acc*100:.1f}%</div>
                    <div class="label">Train Accuracy</div></div>""", unsafe_allow_html=True)
            with c3:
                prec = report["weighted avg"]["precision"]
                st.markdown(f"""<div class="metric-card">
                    <div class="value">{prec*100:.1f}%</div>
                    <div class="label">Precision</div></div>""", unsafe_allow_html=True)
            with c4:
                rec = report["weighted avg"]["recall"]
                st.markdown(f"""<div class="metric-card">
                    <div class="value">{rec*100:.1f}%</div>
                    <div class="label">Recall</div></div>""", unsafe_allow_html=True)

            st.markdown("")

            # ── Charts row
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown('<p class="section-header">Sentiment Distribution</p>', unsafe_allow_html=True)
                fig1, ax1 = plt.subplots(figsize=(5, 3.5))
                fig1.patch.set_facecolor("#141414")
                ax1.set_facecolor("#141414")
                colors = ["#4fc3f7", "#f87171"]
                bars = ax1.bar(dist.index, dist.values, color=colors, width=0.5, edgecolor="none")
                for bar in bars:
                    ax1.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 200,
                             f"{int(bar.get_height()):,}",
                             ha="center", va="bottom", color="#aaa", fontsize=9)
                ax1.set_xlabel("Sentiment", color="#888", fontsize=9)
                ax1.set_ylabel("Count", color="#888", fontsize=9)
                ax1.tick_params(colors="#888")
                ax1.spines[:].set_color("#333")
                plt.tight_layout()
                st.pyplot(fig1)
                plt.close()

            with col_b:
                st.markdown('<p class="section-header">Confusion Matrix</p>', unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                fig2.patch.set_facecolor("#141414")
                ax2.set_facecolor("#141414")
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["positive", "negative"],
                    yticklabels=["positive", "negative"],
                    ax=ax2, linewidths=0.5, linecolor="#333",
                    annot_kws={"color": "white", "size": 12}
                )
                ax2.set_xlabel("Predicted", color="#888", fontsize=9)
                ax2.set_ylabel("Actual", color="#888", fontsize=9)
                ax2.tick_params(colors="#888")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

            # ── Per-class report
            st.markdown('<p class="section-header">Per-Class Report</p>', unsafe_allow_html=True)
            report_df = pd.DataFrame(report).transpose().round(3)
            report_df = report_df.loc[["positive", "negative", "weighted avg"]]
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

            # ── Overfitting note
            gap = train_acc - test_acc
            if gap > 0.05:
                st.warning(f"⚠️ Train accuracy ({train_acc*100:.1f}%) is {gap*100:.1f}% higher than test accuracy. Mild overfitting — expected with TF-IDF + Logistic Regression on this dataset.")

        except Exception as e:
            st.error(f"Error: {e}")

# ── TAB 2 — Predict ───────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">Predict Sentiment</p>', unsafe_allow_html=True)

    model_loaded = load_model()

    if model_loaded is None:
        st.info("Train the model first in the **Train & Metrics** tab.")
    else:
        review_input = st.text_area(
            "Paste any movie review below:",
            height=160,
            placeholder="e.g. This film was an absolute masterpiece. The acting was phenomenal..."
        )

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            predict_btn = st.button("Analyze Sentiment")

        if predict_btn:
            if not review_input.strip():
                st.warning("Please enter a review first.")
            else:
                cleaned = clean_text(review_input)
                prediction = model_loaded.predict([cleaned])[0]
                proba = model_loaded.predict_proba([cleaned])[0]
                classes = model_loaded.classes_
                confidence = dict(zip(classes, proba))
                conf_val = max(proba) * 100

                if prediction == "positive":
                    st.markdown(f"""
                    <div class="result-positive">
                        <div class="result-label">😊 Positive</div>
                        <div class="result-confidence">Confidence: {conf_val:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-negative">
                        <div class="result-label">😞 Negative</div>
                        <div class="result-confidence">Confidence: {conf_val:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("")

                # Confidence bar chart
                fig3, ax3 = plt.subplots(figsize=(5, 2.2))
                fig3.patch.set_facecolor("#141414")
                ax3.set_facecolor("#141414")
                labels = list(confidence.keys())
                vals = [v * 100 for v in confidence.values()]
                bar_colors = ["#4ade80" if l == "positive" else "#f87171" for l in labels]
                ax3.barh(labels, vals, color=bar_colors, height=0.4, edgecolor="none")
                for i, v in enumerate(vals):
                    ax3.text(v + 0.5, i, f"{v:.1f}%", va="center", color="#aaa", fontsize=9)
                ax3.set_xlim(0, 115)
                ax3.set_xlabel("Confidence %", color="#888", fontsize=9)
                ax3.tick_params(colors="#888")
                ax3.spines[:].set_color("#333")
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()

        # ── Try examples
        st.markdown('<p class="section-header">Try an Example</p>', unsafe_allow_html=True)
        examples = {
            "Glowing positive review": "This movie was absolutely brilliant. The direction, acting, and storyline were all exceptional. One of the best films I've seen in years.",
            "Harsh negative review": "Terrible waste of time. The plot made no sense, the acting was wooden, and the ending was a complete letdown. Avoid at all costs.",
            "Mixed review": "The cinematography was stunning but the script felt rushed and the characters were underdeveloped. Decent watch but not memorable.",
        }
        choice = st.selectbox("Pick an example:", list(examples.keys()))
        if st.button("Run Example"):
            ex_text = examples[choice]
            st.text_area("Example review:", value=ex_text, height=80, disabled=True)
            cleaned = clean_text(ex_text)
            pred = model_loaded.predict([cleaned])[0]
            proba = model_loaded.predict_proba([cleaned])[0]
            conf = max(proba) * 100
            emoji = "😊" if pred == "positive" else "😞"
            color = "#4ade80" if pred == "positive" else "#f87171"
            st.markdown(f"**Result:** <span style='color:{color}'>{emoji} {pred.capitalize()} ({conf:.1f}% confidence)</span>", unsafe_allow_html=True)

# ── TAB 3 — About ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">How It Works</p>', unsafe_allow_html=True)
    st.markdown("""
**Dataset**
50,000 IMDB movie reviews labeled as positive or negative (balanced dataset).

**Preprocessing Pipeline**
1. Strip HTML tags (reviews contain `<br />` tags)
2. Remove non-alphabetic characters
3. Convert to lowercase
4. Remove English stopwords (NLTK)
5. Apply Porter Stemmer — reduces words to root form (e.g. *running → run*, *movies → movi*)

**Feature Extraction**
- **TF-IDF Vectorizer** — converts text to numerical vectors based on word frequency weighted by how rare the word is across all documents
- **Chi² Feature Selection** — keeps the top 10,000 most statistically significant features

**Model**
Logistic Regression — a linear classifier that learns a weight for each word feature. Despite the name, it's a classifier not a regressor.

**Why Logistic Regression for text?**
Text data with TF-IDF produces very sparse, high-dimensional vectors. Logistic Regression handles this well and is fast to train. More complex models (Random Forest, Neural Net) don't always outperform it on bag-of-words representations.
    """)
    st.markdown('<p class="section-header">Known Limitations</p>', unsafe_allow_html=True)
    st.markdown("""
- **Sarcasm** — "Oh great, another terrible sequel" will likely be predicted positive
- **Context blindness** — bag-of-words loses word order, so "not good" and "good" look similar
- **Mild overfitting** — train accuracy is higher than test; the model memorizes some noise
    """)
