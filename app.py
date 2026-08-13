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
    :root {
        --navy: #1F3864;
        --navy-light: #2E4E8C;
    }

    /* Overall app background and font */
    .stApp {
        background-color: #0E1117;
    }

    /* Main title */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        padding-bottom: 0px !important;
    }

    /* Subtitle / intro text */
    .subtitle-text {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-top: -8px;
        margin-bottom: 28px;
        line-height: 1.5;
    }

    /* Card wrapper for input sections */
    .card {
        background-color: #161B22;
        border: 1px solid #262D38;
        border-radius: 12px;
        padding: 24px 24px 8px 24px;
        margin-bottom: 20px;
    }

    /* Section labels above text areas */
    .field-label {
        font-weight: 600;
        font-size: 0.95rem;
        color: #E5E7EB;
        margin-bottom: 6px;
    }

    /* Text areas */
    .stTextArea textarea {
        background-color: #0E1117 !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        font-size: 0.92rem !important;
    }
    .stTextArea textarea:focus {
        border: 1px solid var(--navy-light) !important;
        box-shadow: 0 0 0 1px var(--navy-light) !important;
    }

    /* Primary button */
    .stButton > button {
        background-color: var(--navy) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important;
        transition: background-color 0.15s ease;
    }
    .stButton > button:hover {
        background-color: var(--navy-light) !important;
    }

    /* Question card during the answering stage */
    .question-card {
        background-color: #161B22;
        border-left: 4px solid var(--navy-light);
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 18px;
        font-size: 1.05rem;
        line-height: 1.5;
    }

    /* Progress label */
    .progress-label {
        color: #9CA3AF;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }

    /* Score badge in the report */
    .score-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        color: white;
    }

    /* Keyword pills */
    .keyword-pill {
        display: inline-block;
        background-color: #1F3864;
        color: #E5E7EB;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 3px 4px 3px 0;
    }

    /* Divider spacing tweak */
    hr {
        margin: 1.2rem 0 !important;
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


st.title("🎤 GenAI Mock-Interview Coach")
st.markdown(
    '<p class="subtitle-text">Paste your resume and a target job description. '
    'Get tailored questions, answer them, and see exactly where you\'d lose '
    'points in a real interview.</p>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Stage 1: setup - collect resume + job description
# ---------------------------------------------------------------------------
if st.session_state.stage == "setup":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="field-label">📄 Your Resume</div>', unsafe_allow_html=True)
    resume = st.text_area("Resume", height=180, label_visibility="collapsed", placeholder="Paste your resume as plain text...")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="field-label">🎯 Target Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area("Job Description", height=180, label_visibility="collapsed", placeholder="Paste the job description you're targeting...")
    st.markdown('</div>', unsafe_allow_html=True)

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