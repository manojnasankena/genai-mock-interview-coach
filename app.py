"""
GenAI Mock-Interview Coach (Gemini API Version)
--------------------------------------------------
Paste your resume and a target job description, get tailored interview
questions, type your answers in the browser, and get a score + feedback
for each one.

This version uses Google's Gemini API and plain Python text parsing
(no JSON).

Run:
    streamlit run app.py

Requires:
    pip install streamlit google-genai
    Set the GEMINI_API_KEY environment variable to your API key.
    Get a key at: https://aistudio.google.com/app/apikey
"""

import os
import hashlib
import streamlit as st
from google import genai
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-flash-latest"

st.set_page_config(
    page_title="GenAI Mock-Interview Coach",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom styling - gives the app a consistent, professional color scheme
# and cleaner spacing instead of Streamlit's plain defaults.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

    :root {
        --burgundy-deep: #2B0A12;
        --burgundy: #6B1E2B;
        --burgundy-light: #8C2F3F;
        --cream: #F0E6D2;
        --cream-soft: rgba(240, 230, 210, 0.15);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #3B0F1B 0%, #170609 55%);
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, #2B0A12 0%, #6B1E2B 55%, #12294F 100%);
        border: 1px solid #8C2F3F;
        border-radius: 16px;
        padding: 32px 32px 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 30px rgba(43, 10, 18, 0.5);
    }
    .hero h1 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 800 !important;
        color: white !important;
        margin: 0 !important;
        font-size: 2.1rem !important;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: rgba(255,255,255,0.82);
        font-size: 1.02rem;
        margin-top: 8px;
        margin-bottom: 0;
        line-height: 1.55;
        max-width: 640px;
    }

    /* Section labels above text areas */
    .field-label {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        color: #F3F4F6;
        margin-bottom: 8px;
        margin-top: 4px;
    }

    /* Native bordered containers used as cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #25080F !important;
        border: 1px solid #4A1622 !important;
        border-radius: 14px !important;
    }

    /* Text areas */
    .stTextArea textarea {
        background-color: #170609 !important;
        border: 1px solid #4A1622 !important;
        border-radius: 8px !important;
        font-size: 0.92rem !important;
        color: #E5E7EB !important;
    }
    .stTextArea textarea:focus {
        border: 1px solid var(--cream) !important;
        box-shadow: 0 0 0 1px var(--cream) !important;
    }

    /* Slider label */
    .stSlider label p {
        font-family: 'Poppins', sans-serif;
        font-weight: 600 !important;
        color: #F3F4F6 !important;
    }

    /* Primary button - burgundy to cream accent */
    .stButton > button {
        background: linear-gradient(90deg, #6B1E2B 0%, #8C2F3F 100%) !important;
        color: white !important;
        border: 1px solid var(--cream) !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.4rem !important;
        font-weight: 700 !important;
        font-family: 'Poppins', sans-serif !important;
        letter-spacing: 0.2px;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(240, 230, 210, 0.35);
        border: 1px solid var(--cream) !important;
    }

    /* Question card during the answering stage */
    .question-card {
        background: linear-gradient(135deg, #25080F 0%, #2B0A12 100%);
        border-left: 4px solid var(--cream);
        border-radius: 10px;
        padding: 20px 22px;
        margin-bottom: 18px;
        font-size: 1.08rem;
        line-height: 1.55;
        color: #F3F4F6;
    }

    /* Progress label */
    .progress-label {
        color: var(--cream);
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }

    /* Score badge in the report */
    .score-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        color: white;
        font-family: 'Poppins', sans-serif;
    }

    /* Keyword pills */
    .keyword-pill {
        display: inline-block;
        background: linear-gradient(90deg, #6B1E2B 0%, #2B0A12 100%);
        color: #E5E7EB;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 3px 5px 3px 0;
        border: 1px solid var(--cream);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #25080F;
        border: 1px solid #4A1622;
        border-radius: 12px;
        padding: 14px 10px;
    }

    hr {
        margin: 1.2rem 0 !important;
        border-color: #4A1622 !important;
    }

    /* Login page specific styling */
    .login-wrapper {
        max-width: 420px;
        margin: 60px auto 0 auto;
    }
    .login-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 1.6rem;
        color: white;
        text-align: center;
        margin-bottom: 4px;
    }
    .login-subtitle {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.92rem;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Set the GEMINI_API_KEY environment variable before running this app.")
        st.stop()
    return genai.Client(api_key=api_key)


def ask_gemini(client, prompt):
    """Send a prompt to Gemini and return the plain text reply as a string."""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return response.text or ""


def generate_questions(client, resume, job_description, num_questions):
    prompt = (
        "You are a senior technical interviewer. Given a candidate's resume and a "
        "target job description, write interview questions that test whether the "
        "candidate can actually do the job. Mix background questions, technical "
        "questions based on the job description, and one behavioral question. "
        "Reply with ONLY the questions, one per line, with no numbering, no "
        "introduction, and no extra text.\n\n"
        f"RESUME:\n{resume}\n\n"
        f"JOB DESCRIPTION:\n{job_description}\n\n"
        f"Write exactly {num_questions} interview questions, one per line."
    )
    reply_text = ask_gemini(client, prompt)

    questions = []
    for line in reply_text.split("\n"):
        line = line.strip()
        if line:
            questions.append(line)
    return questions


def score_answer(client, question, answer, job_description):
    prompt = (
        "You are a strict but fair technical interviewer. Score the candidate's "
        "answer against the job description on relevance, correctness, and "
        "completeness. Reply in EXACTLY this plain text format, three lines, "
        "nothing else:\n"
        "SCORE: <a whole number from 0 to 10>\n"
        "FEEDBACK: <2-3 direct, actionable sentences>\n"
        "MISSING: <comma-separated list of up to 3 missing keywords or concepts, "
        "or NONE if nothing important was missing>\n\n"
        f"JOB DESCRIPTION:\n{job_description}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CANDIDATE ANSWER:\n{answer}\n\n"
        "Score this answer using the exact format described."
    )
    reply_text = ask_gemini(client, prompt)

    score = 0
    feedback = ""
    missing_keywords = []

    for line in reply_text.split("\n"):
        line = line.strip()
        if line.startswith("SCORE:"):
            score_text = line.replace("SCORE:", "").strip()
            try:
                score = int(score_text)
            except ValueError:
                score = 0
        elif line.startswith("FEEDBACK:"):
            feedback = line.replace("FEEDBACK:", "").strip()
        elif line.startswith("MISSING:"):
            missing_text = line.replace("MISSING:", "").strip()
            if missing_text.upper() != "NONE":
                missing_keywords = [word.strip() for word in missing_text.split(",")]

    return score, feedback, missing_keywords


# ---------------------------------------------------------------------------
# Authentication - simple hashed-password login gate. The plaintext password
# is never stored anywhere; only its SHA-256 hash is compared. Credentials
# come from environment variables (APP_USERNAME, APP_PASSWORD_HASH), which
# you set via .env locally or Streamlit Cloud's Secrets panel - never hard-
# code real credentials directly in this file.
# ---------------------------------------------------------------------------

def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def check_credentials(username, password):
    correct_username = os.environ.get("APP_USERNAME", "")
    correct_password_hash = os.environ.get("APP_PASSWORD_HASH", "")
    if not correct_username or not correct_password_hash:
        return False
    return username == correct_username and hash_password(password) == correct_password_hash


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


def render_login_page():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700;800&family=Inter:wght@400;500&display=swap');
        .stApp {
            background: radial-gradient(circle at 50% 0%, #3B0F1B 0%, #170609 60%);
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🔒 GenAI Mock-Interview Coach</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Sign in to continue</div>', unsafe_allow_html=True)

    with st.container(border=True):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Log In", use_container_width=True):
            if check_credentials(username, password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    st.markdown('</div>', unsafe_allow_html=True)


if not st.session_state.authenticated:
    render_login_page()
    st.stop()


# ---------------------------------------------------------------------------
# Session state - Streamlit re-runs the whole script on every click, so we
# store progress in st.session_state to remember where we are.
# ---------------------------------------------------------------------------

if "stage" not in st.session_state:
    st.session_state.stage = "setup"
if "questions" not in st.session_state:
    st.session_state.questions = []
if "q_index" not in st.session_state:
    st.session_state.q_index = 0
if "scores" not in st.session_state:
    st.session_state.scores = []
if "feedback_list" not in st.session_state:
    st.session_state.feedback_list = []
if "missing_keywords_all" not in st.session_state:
    st.session_state.missing_keywords_all = []


st.markdown("""
<div class="hero">
    <h1>🎤 GenAI Mock-Interview Coach</h1>
    <p>Paste your resume and a target job description. Get tailored questions,
    answer them, and see exactly where you'd lose points in a real interview.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.write(f"Logged in")
    if st.button("Log Out"):
        st.session_state.authenticated = False
        st.rerun()


# ---------------------------------------------------------------------------
# Stage 1: setup - collect resume + job description
# ---------------------------------------------------------------------------
if st.session_state.stage == "setup":
    with st.container(border=True):
        st.markdown('<div class="field-label">📄 Your Resume</div>', unsafe_allow_html=True)
        resume = st.text_area("Resume", height=180, label_visibility="collapsed", placeholder="Paste your resume as plain text...")

    with st.container(border=True):
        st.markdown('<div class="field-label">🎯 Target Job Description</div>', unsafe_allow_html=True)
        job_description = st.text_area("Job Description", height=180, label_visibility="collapsed", placeholder="Paste the job description you're targeting...")

    num_questions = st.slider("Number of questions", 3, 8, 5)

    if st.button("Generate Questions →", use_container_width=True):
        if resume.strip() == "" or job_description.strip() == "":
            st.warning("Please paste both your resume and a job description.")
        else:
            client = get_client()
            with st.spinner("Generating tailored interview questions..."):
                questions = generate_questions(client, resume, job_description, num_questions)

            st.session_state.questions = questions
            st.session_state.job_description = job_description
            st.session_state.q_index = 0
            st.session_state.scores = []
            st.session_state.feedback_list = []
            st.session_state.missing_keywords_all = []
            st.session_state.stage = "answering"
            st.rerun()


# ---------------------------------------------------------------------------
# Stage 2: answering - show one question at a time
# ---------------------------------------------------------------------------
elif st.session_state.stage == "answering":
    index = st.session_state.q_index
    total = len(st.session_state.questions)
    question = st.session_state.questions[index]

    st.markdown(f'<div class="progress-label">Question {index + 1} of {total}</div>', unsafe_allow_html=True)
    st.progress((index) / total)
    st.markdown(f'<div class="question-card">{question}</div>', unsafe_allow_html=True)

    answer = st.text_area("Your answer", height=150, key=f"answer_box_{index}", placeholder="Type your answer here...")

    if st.button("Submit Answer →", use_container_width=True):
        if answer.strip() == "":
            st.warning("Type an answer before submitting.")
        else:
            client = get_client()
            with st.spinner("Scoring your answer..."):
                score, feedback, missing_keywords = score_answer(
                    client, question, answer, st.session_state.job_description
                )

            st.session_state.scores.append(score)
            st.session_state.feedback_list.append(feedback)
            st.session_state.missing_keywords_all.extend(missing_keywords)

            if index + 1 < total:
                st.session_state.q_index += 1
            else:
                st.session_state.stage = "report"
            st.rerun()


# ---------------------------------------------------------------------------
# Stage 3: report - show the final scored report
# ---------------------------------------------------------------------------
elif st.session_state.stage == "report":
    st.markdown('<div class="progress-label">Interview Report</div>', unsafe_allow_html=True)
    st.title("📊 Your Results")

    scores = st.session_state.scores
    if len(scores) > 0:
        average_score = sum(scores) / len(scores)
    else:
        average_score = 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Score", f"{average_score:.1f} / 10")
    with col2:
        st.metric("Questions Answered", f"{len(scores)}")
    with col3:
        unique_kw_count = len(set(st.session_state.missing_keywords_all))
        st.metric("Gaps Identified", f"{unique_kw_count}")

    st.markdown("### Per-Question Breakdown")

    def score_color(score):
        if score >= 7:
            return "#1E8E3E"  # green
        elif score >= 4:
            return "#B8860B"  # amber
        else:
            return "#C0392B"  # red

    for i in range(len(st.session_state.questions)):
        question = st.session_state.questions[i]
        score = st.session_state.scores[i]
        feedback = st.session_state.feedback_list[i]
        color = score_color(score)
        with st.expander(f"Q{i + 1}: {question[:70]}"):
            st.markdown(
                f'<span class="score-badge" style="background-color:{color};">{score}/10</span>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.write(feedback)

    if st.session_state.missing_keywords_all:
        unique_keywords = sorted(set(st.session_state.missing_keywords_all))
        st.markdown("### 🔑 Keywords to Study")
        pills_html = "".join(f'<span class="keyword-pill">{kw}</span>' for kw in unique_keywords)
        st.markdown(pills_html, unsafe_allow_html=True)

    st.write("")
    if st.button("Start New Session", use_container_width=True):
        st.session_state.stage = "setup"
        st.session_state.questions = []
        st.session_state.q_index = 0
        st.session_state.scores = []
        st.session_state.feedback_list = []
        st.session_state.missing_keywords_all = []
        st.rerun()