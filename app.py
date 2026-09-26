
import streamlit as st
import joblib

from WordCharFeatures_transformer import WordCharFeatures


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Hate Speech Detection",
    page_icon="🛡️",
    layout="wide"
)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load("hate_speech_word_char_lr.pkl")


model = load_model()


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* ---------- MAIN BACKGROUND ---------- */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, #dbeafe 0%, transparent 30%),
        radial-gradient(circle at 90% 10%, #ede9fe 0%, transparent 30%),
        linear-gradient(135deg, #f8fbff, #eef2ff);
}


/* ---------- MAIN CONTAINER ---------- */

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ---------- HERO ---------- */

.hero {
    padding: 32px;
    border-radius: 25px;
    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 15px 40px rgba(59,130,246,0.20);
}

.hero h1 {
    color: white !important;
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    color: #f1f5f9 !important;
    font-size: 18px;
}


/* ---------- GENERAL CARDS ---------- */

.card {
    background: #ffffff !important;
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 8px 30px rgba(15,23,42,0.08);
    margin-bottom: 20px;
    color: #111827 !important;
}

.card h3 {
    color: #111827 !important;
    font-size: 22px;
}

.card p {
    color: #374151 !important;
    font-size: 16px;
    line-height: 1.6;
}

.card b {
    color: #111827 !important;
}


/* ---------- METRIC CARDS ---------- */

.metric-card {
    text-align: center;
    background: #ffffff !important;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.06);
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #2563eb !important;
}

.metric-label {
    color: #475569 !important;
}


/* ---------- INPUT ---------- */

textarea {
    border-radius: 15px !important;
    color: #111827 !important;
    background-color: #ffffff !important;
}


/* ---------- BUTTONS ---------- */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 48px;
    font-weight: 600;
    color: #111827 !important;
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
}

.stButton > button:hover {
    border-color: #2563eb !important;
    color: #2563eb !important;
}


/* ---------- RESULT CARDS ---------- */

.result-hate {
    background: #fee2e2 !important;
    border-left: 7px solid #ef4444;
    padding: 25px;
    border-radius: 15px;
    color: #7f1d1d !important;
}

.result-hate h2,
.result-hate p {
    color: #7f1d1d !important;
}


.result-offensive {
    background: #fef3c7 !important;
    border-left: 7px solid #f59e0b;
    padding: 25px;
    border-radius: 15px;
    color: #78350f !important;
}

.result-offensive h2,
.result-offensive p {
    color: #78350f !important;
}


.result-neutral {
    background: #dcfce7 !important;
    border-left: 7px solid #22c55e;
    padding: 25px;
    border-radius: 15px;
    color: #14532d !important;
}

.result-neutral h2,
.result-neutral p {
    color: #14532d !important;
}


/* ---------- STREAMLIT HEADINGS ---------- */

h1, h2, h3, h4 {
    color: #111827 !important;
}


/* ---------- NORMAL TEXT ---------- */

p, label, span {
    color: #374151;
}


/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    color: #475569 !important;
    padding: 25px;
    font-size: 14px;
}


/* ---------- DIVIDER ---------- */

hr {
    border-color: #cbd5e1 !important;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# HERO
# ==================================================

st.markdown("""
<div class="hero">

<h1>🛡️ Hate Speech Detection</h1>

<p>
Analyze tweets and text using a traditional Machine Learning
pipeline to classify content as Hate Speech, Offensive Language,
or Neither.
</p>

</div>
""", unsafe_allow_html=True)


# ==================================================
# METRICS
# ==================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">3</div>
        <div class="metric-label">Classes</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">89%</div>
        <div class="metric-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">0.7485</div>
        <div class="metric-label">Macro F1</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">ML</div>
        <div class="metric-label">Model Type</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")


# ==================================================
# MAIN INPUT
# ==================================================

left, right = st.columns([2.2, 1])


with left:

    st.markdown("""
    <div class="card">

    <h3>✍️ Enter Text</h3>

    <p>
    Enter a tweet or any text and let the machine learning
    model classify it.
    </p>

    </div>
    """, unsafe_allow_html=True)

    text = st.text_area(
        "Text",
        height=180,
        placeholder="Type or paste your tweet here...",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns(2)

    with col1:
        predict = st.button(
            "🔍 Predict",
            use_container_width=True
        )

    with col2:
        clear = st.button(
            "🗑️ Clear",
            use_container_width=True
        )


# ==================================================
# CLASS INFORMATION
# ==================================================

with right:

    st.markdown("""
    <div class="card">

    <h3>📚 Class Categories</h3>

    <p>
    🔴 <b>Hate Speech</b><br>
    Content targeting individuals or groups.
    </p>

    <p>
    🟠 <b>Offensive Language</b><br>
    Abusive or offensive language without necessarily
    targeting a protected group.
    </p>

    <p>
    🟢 <b>Neither</b><br>
    Neutral or non-offensive content.
    </p>

    </div>
    """, unsafe_allow_html=True)


# ==================================================
# EXAMPLES
# ==================================================

st.markdown("""
<div class="card">

<h3>💡 Try an Example</h3>

<p>
Select an example below to test the model.
</p>

</div>
""", unsafe_allow_html=True)


examples = [
    "I really hate people who spread hateful messages.",
    "You are such an idiot, stop being annoying.",
    "The weather is really nice today."
]


e1, e2, e3 = st.columns(3)

with e1:
    if st.button("🔴 Example 1"):
        text = examples[0]

with e2:
    if st.button("🟠 Example 2"):
        text = examples[1]

with e3:
    if st.button("🟢 Example 3"):
        text = examples[2]


# ==================================================
# PREDICTION
# ==================================================

if predict:

    if not text.strip():

        st.warning("Please enter some text first.")

    else:

        prediction = model.predict([text])[0]

        probabilities = model.predict_proba([text])[0]

        labels = {
            0: "Hate Speech",
            1: "Offensive Language",
            2: "Neither"
        }

        result = labels[prediction]

        st.markdown("---")

        st.subheader("📊 Prediction Result")


        if prediction == 0:

            st.markdown(
                f"""
                <div class="result-hate">

                <h2>🔴 {result}</h2>

                <p>
                The model classified this text as Hate Speech.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        elif prediction == 1:

            st.markdown(
                f"""
                <div class="result-offensive">

                <h2>🟠 {result}</h2>

                <p>
                The model classified this text as Offensive Language.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        else:

            st.markdown(
                f"""
                <div class="result-neutral">

                <h2>🟢 {result}</h2>

                <p>
                The model classified this text as Neither.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ==================================================
        # CONFIDENCE
        # ==================================================

        st.subheader("📈 Prediction Confidence")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.metric(
                "Hate Speech",
                f"{probabilities[0] * 100:.2f}%"
            )
            st.progress(float(probabilities[0]))

        with p2:
            st.metric(
                "Offensive",
                f"{probabilities[1] * 100:.2f}%"
            )
            st.progress(float(probabilities[1]))

        with p3:
            st.metric(
                "Neither",
                f"{probabilities[2] * 100:.2f}%"
            )
            st.progress(float(probabilities[2]))


# ==================================================
# MODEL INFORMATION
# ==================================================

st.markdown("---")

st.subheader("⚙️ Model Information")


m1, m2 = st.columns(2)


with m1:

    st.markdown("""
    <div class="card">

    <h3>Machine Learning Pipeline</h3>

    <p>
    <b>Feature Extraction</b><br>
    Word + Character TF-IDF
    </p>

    <p>
    <b>Class Balancing</b><br>
    Random Oversampling
    </p>

    <p>
    <b>Classifier</b><br>
    Logistic Regression
    </p>

    </div>
    """, unsafe_allow_html=True)


with m2:

    st.markdown("""
    <div class="card">

    <h3>Model Performance</h3>

    <p>
    <b>Accuracy:</b> 89%
    </p>

    <p>
    <b>Macro Precision:</b> 0.73
    </p>

    <p>
    <b>Macro Recall:</b> 0.77
    </p>

    <p>
    <b>Macro F1:</b> 0.7485
    </p>

    </div>
    """, unsafe_allow_html=True)


# ==================================================
# FOOTER
# ==================================================

st.markdown("""
<div class="footer">

🛡️ Hate Speech Detection • Traditional Machine Learning Project

<br><br>

Word + Character TF-IDF • Random Oversampling • Logistic Regression

</div>
""", unsafe_allow_html=True)



# ```python
# import streamlit as st
# import joblib

# from word_char_features import WordCharFeatures


# # Load model
# model = joblib.load("hate_speech_word_char_lr.pkl")


# # Page configuration
# st.set_page_config(
#     page_title="Hate Speech Detector",
#     page_icon="🛡️",
#     layout="centered"
# )


# # Title
# st.title("🛡️ Hate Speech Detection")
# st.write("Enter a tweet or text to classify it.")


# # Text input
# text = st.text_area(
#     "Enter your text:",
#     placeholder="Type your tweet here..."
# )


# # Prediction
# if st.button("Predict"):

#     if not text.strip():
#         st.warning("Please enter some text.")

#     else:
#         prediction = model.predict([text])[0]

#         labels = {
#             0: "Hate Speech",
#             1: "Offensive Language",
#             2: "Neither"
#         }

#         result = labels[prediction]

#         if prediction == 0:
#             st.error(f"Prediction: {result}")

#         elif prediction == 1:
#             st.warning(f"Prediction: {result}")

#         else:
#             st.success(f"Prediction: {result}")
# ```
