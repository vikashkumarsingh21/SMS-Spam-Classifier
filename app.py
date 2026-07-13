"""
Email & SMS Spam Detection
A clean, minimal SaaS-style Streamlit application.

IMPORTANT: The machine learning pipeline (transform_text, vectorizer loading,
model loading, and prediction logic) below is UNCHANGED from the original
implementation. Only the interface, layout, and user experience have been
redesigned.
"""

import re
import time
import string
import platform

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

# =====================================================================================
# NLTK SETUP (safe, silent download so the app never crashes on a fresh machine)
# =====================================================================================
for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(
            f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}"
        )
    except LookupError:
        nltk.download(resource, quiet=True)

ps = PorterStemmer()


# =====================================================================================
# ORIGINAL MACHINE LEARNING PIPELINE -- DO NOT MODIFY
# =====================================================================================
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Loads the original vectorizer and model exactly as before (cached so the
    pickle files are only read once per session, not on every rerun)."""
    tfidf_local = pickle.load(open('vectorizer.pkl', 'rb'))
    model_local = pickle.load(open('model.pkl', 'rb'))
    return tfidf_local, model_local


PIPELINE_LOAD_ERROR = None
try:
    tfidf, model = load_pipeline()
except Exception as e:
    tfidf, model = None, None
    PIPELINE_LOAD_ERROR = str(e)
# =====================================================================================
# END OF ORIGINAL MACHINE LEARNING PIPELINE
# =====================================================================================


# =====================================================================================
# PAGE CONFIGURATION
# =====================================================================================
st.set_page_config(
    page_title="Spam Detection",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================================
# SESSION STATE (analytics counters persist for the life of the session)
# =====================================================================================
defaults = {
    "messages_processed": 0,
    "spam_detected": 0,
    "legit_detected": 0,
    "response_times": [],
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================================================
# GLOBAL THEME / CSS
# A restrained, light SaaS-dashboard aesthetic: one accent color, white/light-gray
# surfaces, thin borders, small radii, and near-invisible motion.
# =====================================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg:        #F6F8FA;
        --surface:   #FFFFFF;
        --border:    #D8DEE4;
        --border-soft: #E9ECEF;
        --text:      #1F2328;
        --text-soft: #59636E;
        --text-faint:#8B949E;
        --accent:    #1F6FEB;
        --accent-hover: #1A5FD1;
        --accent-soft: #EDF3FE;
        --success:   #1A7F37;
        --success-bg:#F0FAF3;
        --danger:    #CF222E;
        --danger-bg: #FDF1F1;
        --warning:   #9A6700;
        --radius:    11px;
        --radius-sm: 8px;
    }

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    .stApp { background: var(--bg); color: var(--text); }
    #MainMenu, header, footer { visibility: hidden; }

    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border-soft);
    }

    /* ---------- Barely-noticeable entrance ---------- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeIn 0.35s ease-out both; }

    /* ---------- Top bar ---------- */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 4px 16px 4px;
        border-bottom: 1px solid var(--border-soft);
        margin-bottom: 24px;
    }
    .topbar-brand {
        font-weight: 600;
        font-size: 0.98rem;
        color: var(--text);
        letter-spacing: -0.01em;
    }
    .topbar-status {
        font-size: 0.82rem;
        color: var(--text-soft);
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .status-dot {
        height: 7px; width: 7px; border-radius: 50%;
        background: var(--success);
        display: inline-block;
    }

    /* ---------- Page heading ---------- */
    .page-eyebrow {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--text-faint);
        margin-bottom: 6px;
    }
    .page-title {
        font-size: 1.55rem;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.01em;
        margin-bottom: 4px;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: var(--text-soft);
        margin-bottom: 22px;
        max-width: 620px;
        line-height: 1.5;
    }

    /* ---------- Card ---------- */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 20px 22px;
        box-shadow: 0 1px 2px rgba(31,35,40,0.04);
        transition: box-shadow 0.15s ease, border-color 0.15s ease;
        margin-bottom: 16px;
    }
    .card:hover { box-shadow: 0 2px 6px rgba(31,35,40,0.06); border-color: var(--text-faint); }

    .section-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text);
        margin: 4px 0 12px 0;
    }

    /* ---------- Result status card ---------- */
    .result-card {
        border-radius: var(--radius);
        padding: 22px 24px;
        display: flex;
        align-items: flex-start;
        gap: 14px;
        border: 1px solid var(--border);
    }
    .result-card.safe  { background: var(--success-bg); border-color: #B4E2C1; }
    .result-card.spam  { background: var(--danger-bg);  border-color: #F1C0C4; }
    .result-icon-wrap {
        width: 34px; height: 34px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0; font-size: 1.05rem; font-weight: 700;
    }
    .result-icon-wrap.safe { background: #DCF4E3; color: var(--success); }
    .result-icon-wrap.spam { background: #FBE1E2; color: var(--danger); }
    .result-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 3px; }
    .result-title.safe { color: #14532D; }
    .result-title.spam { color: #7A1620; }
    .result-desc { font-size: 0.9rem; color: var(--text-soft); line-height: 1.5; }
    .result-confidence {
        font-size: 0.82rem;
        color: var(--text-soft);
        margin-top: 6px;
        font-variant-numeric: tabular-nums;
    }

    /* ---------- KPI ---------- */
    .kpi {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 14px 16px;
    }
    .kpi-value { font-size: 1.4rem; font-weight: 600; color: var(--text); font-variant-numeric: tabular-nums; }
    .kpi-label { font-size: 0.78rem; color: var(--text-soft); margin-top: 2px; }

    /* ---------- Workflow strip ---------- */
    .flow-strip { display: flex; align-items: center; flex-wrap: wrap; gap: 0; }
    .flow-step {
        font-size: 0.82rem;
        font-weight: 500;
        color: var(--text-soft);
        background: var(--surface);
        border: 1px solid var(--border-soft);
        border-radius: var(--radius-sm);
        padding: 7px 12px;
        white-space: nowrap;
    }
    .flow-sep { color: var(--text-faint); padding: 0 8px; font-size: 0.85rem; }

    /* ---------- Buttons ---------- */
    .stButton>button {
        background: var(--accent);
        color: #FFFFFF;
        font-weight: 500;
        font-size: 0.9rem;
        border: 1px solid var(--accent);
        border-radius: var(--radius-sm);
        padding: 0.5rem 1.1rem;
        transition: background 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton>button:hover {
        background: var(--accent-hover);
        border-color: var(--accent-hover);
        color: #FFFFFF;
        box-shadow: 0 1px 3px rgba(31,111,235,0.25);
    }
    .stButton>button:active { background: #164BAC; }

    /* Secondary-looking buttons (used for nav-style actions) */
    div[data-testid="stButton"] button[kind="secondary"] {
        background: var(--surface);
        color: var(--text);
        border: 1px solid var(--border);
    }

    /* ---------- Text area (Gmail / ChatGPT style composer) ---------- */
    .stTextArea textarea {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text) !important;
        font-size: 0.95rem !important;
        line-height: 1.55 !important;
        padding: 14px !important;
        box-shadow: 0 1px 2px rgba(31,35,40,0.03) !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }

    /* ---------- Tabs (minimal underline nav) ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 22px;
        border-bottom: 1px solid var(--border-soft);
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        color: var(--text-soft);
        font-size: 0.9rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: var(--text) !important;
    }

    /* ---------- Footer ---------- */
    .app-footer {
        text-align: center;
        padding: 22px 10px 10px 10px;
        color: var(--text-faint);
        font-size: 0.8rem;
        border-top: 1px solid var(--border-soft);
        margin-top: 28px;
    }

    /* ---------- Misc text helpers ---------- */
    .muted { color: var(--text-soft); font-size: 0.85rem; }
    .divider-space { margin: 22px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================================
# SIDEBAR
# =====================================================================================
with st.sidebar:
    st.markdown("**Spam Detection**")
    st.caption("Message classification tool")
    st.divider()

    st.markdown("**Project**")
    st.caption("Version 1.0")
    st.caption("Maintained by Your Name")
    st.divider()

    with st.expander("How to use this", expanded=False):
        st.write(
            "Open **Detection**, paste an email or SMS message, and select "
            "**Analyze message**. Results and a breakdown of the message "
            "appear below the composer."
        )

    st.divider()
    st.markdown("**Model status**")
    if model is not None and tfidf is not None:
        st.caption("Model and vectorizer loaded")
    else:
        st.caption("Model files not found")
        st.caption("Place model.pkl and vectorizer.pkl in the app directory.")


# =====================================================================================
# TOP BAR (minimal)
# =====================================================================================
st.markdown(
    """
    <div class="topbar">
        <div class="topbar-brand">Spam Detection</div>
        <div class="topbar-status"><span class="status-dot"></span>Model ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)

detection_tab, analytics_tab, about_tab = st.tabs(["Detection", "Analytics", "About"])


# =====================================================================================
# DETECTION TAB (primary focus)
# =====================================================================================
with detection_tab:
    st.markdown('<div class="page-eyebrow">Message classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Check a message for spam</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Paste an email or SMS below. The model classifies it '
        'as spam or legitimate using the trained pipeline, without sending anything anywhere.</div>',
        unsafe_allow_html=True,
    )

    input_sms = st.text_area(
        "Message",
        height=170,
        placeholder="Paste your email or SMS here…",
        label_visibility="collapsed",
        key="input_sms",
    )

    char_count = len(input_sms)
    word_count = len(input_sms.split()) if input_sms.strip() else 0

    meta_col1, meta_col2, meta_spacer = st.columns([1, 1, 4])
    meta_col1.caption(f"{char_count} characters")
    meta_col2.caption(f"{word_count} words")

    detect_clicked = st.button("Analyze message", use_container_width=False)

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    if detect_clicked:
        if not input_sms or not input_sms.strip():
            st.warning("Enter a message before running the analysis.")
        elif model is None or tfidf is None:
            st.error(
                f"Prediction unavailable — model or vectorizer failed to load. "
                f"({PIPELINE_LOAD_ERROR})"
            )
        else:
            # ---------------- Lightweight, unobtrusive progress feedback ----------------
            status_text = st.empty()
            progress_bar = st.progress(0)
            stages = [
                ("Cleaning text", 25),
                ("Vectorizing", 60),
                ("Running model", 90),
                ("Done", 100),
            ]
            start_time = time.time()
            for stage_text, pct in stages:
                status_text.caption(stage_text)
                progress_bar.progress(pct)
                time.sleep(0.12)

            # ---------------- ORIGINAL PREDICTION LOGIC (UNCHANGED) ----------------
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            # ---------------- END ORIGINAL PREDICTION LOGIC ----------------

            confidence = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(vector_input)[0]
                    confidence = max(proba) * 100
                except Exception:
                    confidence = None

            elapsed = time.time() - start_time
            status_text.empty()
            progress_bar.empty()

            # ---------------- Update session analytics ----------------
            st.session_state["messages_processed"] += 1
            st.session_state["response_times"].append(elapsed)
            if result == 1:
                st.session_state["spam_detected"] += 1
            else:
                st.session_state["legit_detected"] += 1

            # ---------------- Result Display ----------------
            if result == 1:
                st.markdown(
                    f"""
                    <div class="result-card spam fade-in">
                        <div class="result-icon-wrap spam">!</div>
                        <div>
                            <div class="result-title spam">Spam detected</div>
                            <div class="result-desc">This message contains patterns commonly
                            associated with spam or phishing attempts.</div>
                            {f'<div class="result-confidence">Confidence: {confidence:.1f}%</div>' if confidence else ''}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card safe fade-in">
                        <div class="result-icon-wrap safe">✓</div>
                        <div>
                            <div class="result-title safe">Legitimate message</div>
                            <div class="result-desc">No spam characteristics were detected in this message.</div>
                            {f'<div class="result-confidence">Confidence: {confidence:.1f}%</div>' if confidence else ''}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(f"Response time: {elapsed:.2f}s")

            # ---------------- Message Insights (analytical only) ----------------
            st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Message breakdown</div>', unsafe_allow_html=True)

            sentence_count = len(re.split(r'[.!?]+', input_sms)) - 1
            sentence_count = max(sentence_count, 1 if input_sms.strip() else 0)
            uppercase_count = sum(1 for ch in input_sms if ch.isupper())
            digit_count = sum(1 for ch in input_sms if ch.isdigit())
            special_char_count = sum(1 for ch in input_sms if ch in string.punctuation)
            links_found = re.findall(r'(https?://\S+|www\.\S+)', input_sms)
            emails_found = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', input_sms)
            phones_found = re.findall(r'\b\d{10}\b|\+\d{1,3}[\s-]?\d{6,12}', input_sms)

            insight_data = [
                ("Characters", char_count),
                ("Words", word_count),
                ("Sentences", sentence_count),
                ("Uppercase characters", uppercase_count),
                ("Numbers", digit_count),
                ("Special characters", special_char_count),
                ("Links", len(links_found)),
                ("Email addresses", len(emails_found)),
            ]
            insight_cols = st.columns(4)
            for idx, (label, value) in enumerate(insight_data):
                with insight_cols[idx % 4]:
                    st.markdown(
                        f"""<div class="kpi"><div class="kpi-value">{value}</div>
                        <div class="kpi-label">{label}</div></div>""",
                        unsafe_allow_html=True,
                    )

            if phones_found:
                st.caption(f"Phone numbers detected: {len(phones_found)}")

    # ---------------- How it works (secondary, below the fold) ----------------
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    with st.expander("How the classification works"):
        st.markdown(
            """
            <div class="flow-strip">
                <span class="flow-step">Input message</span><span class="flow-sep">→</span>
                <span class="flow-step">Text cleaning</span><span class="flow-sep">→</span>
                <span class="flow-step">Tokenization</span><span class="flow-sep">→</span>
                <span class="flow-step">Stopword removal</span><span class="flow-sep">→</span>
                <span class="flow-step">Stemming</span><span class="flow-sep">→</span>
                <span class="flow-step">TF-IDF vectorization</span><span class="flow-sep">→</span>
                <span class="flow-step">Model</span><span class="flow-sep">→</span>
                <span class="flow-step">Spam / Legitimate</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.caption("Model type")
            st.write(type(model).__name__ if model else "Unavailable")
            st.caption("Vectorizer")
            st.write("TF-IDF")
        with m2:
            st.caption("Prediction type")
            st.write("Binary classification")
            st.caption("Language")
            st.write("English")


# =====================================================================================
# ANALYTICS TAB
# =====================================================================================
with analytics_tab:
    st.markdown('<div class="page-eyebrow">Session overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">A running summary of the messages analyzed in this session.</div>',
        unsafe_allow_html=True,
    )

    total = st.session_state["messages_processed"]
    spam = st.session_state["spam_detected"]
    legit = st.session_state["legit_detected"]
    avg_rt = (
        sum(st.session_state["response_times"]) / len(st.session_state["response_times"])
        if st.session_state["response_times"] else 0.0
    )

    k1, k2, k3, k4 = st.columns(4)
    for col, label, value in [
        (k1, "Messages processed", total),
        (k2, "Spam detected", spam),
        (k3, "Legitimate", legit),
        (k4, "Avg. response time", f"{avg_rt:.2f}s"),
    ]:
        with col:
            st.markdown(
                f"""<div class="kpi"><div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div></div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    if total > 0:
        st.markdown('<div class="section-label">Spam ratio this session</div>', unsafe_allow_html=True)
        st.progress(spam / total if total else 0)
        st.caption(f"{spam} of {total} messages classified as spam")
    else:
        st.caption("No messages analyzed yet. Results will appear here once you run the classifier.")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">System</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.caption("Model")
    s1.write("Loaded" if model is not None else "Missing")
    s2.caption("Vectorizer")
    s2.write("Loaded" if tfidf is not None else "Missing")
    s3.caption("Python")
    s3.write(platform.python_version())
    s4.caption("Streamlit")
    s4.write(st.__version__)


# =====================================================================================
# ABOUT TAB
# =====================================================================================
with about_tab:
    st.markdown('<div class="page-eyebrow">Reference</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">About this tool</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Background on how the classifier works and why it matters.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("What is spam detection?", expanded=True):
        st.write(
            "Spam detection identifies unwanted or malicious messages — including "
            "phishing attempts — using rule-based or machine learning techniques."
        )

    with st.expander("How the model works"):
        st.write(
            "The model learns patterns from a labeled set of spam and legitimate "
            "messages during training, then applies those patterns to classify new, "
            "unseen messages."
        )

    with st.expander("TF-IDF vectorization"):
        st.write(
            "TF-IDF converts text into numerical features by weighing words based on "
            "how often they appear in a message relative to how common they are "
            "across the full dataset."
        )

    with st.expander("Text preprocessing"):
        st.write(
            "Before classification, text is lowercased, tokenized, stripped of "
            "stopwords and punctuation, and stemmed to its root form, producing a "
            "clean representation for the model."
        )

    with st.expander("Why this matters"):
        st.write(
            "Spam and phishing messages are a common vector for fraud and account "
            "compromise. Automated classification helps flag risky messages before "
            "they reach the reader."
        )


# =====================================================================================
# FOOTER
# =====================================================================================
st.markdown(
    """
    <div class="app-footer">
        Built with Python, Streamlit, NLTK and scikit-learn · Version 1.0
    </div>
    """,
    unsafe_allow_html=True,
)