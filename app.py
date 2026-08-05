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
st.set_page_config(page_title="GenAI Mock-Interview Coach", page_icon="🎤")
 
 
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
st.write("Paste your resume and a target job description. Get tailored questions, answer them, and see exactly where you'd lose points in a real interview.")
 
 
# ---------------------------------------------------------------------------
# Stage 1: setup - collect resume + job description
# ---------------------------------------------------------------------------
if st.session_state.stage == "setup":
    resume = st.text_area("Your resume (paste as plain text)", height=200)
    job_description = st.text_area("Target job description", height=200)
    num_questions = st.slider("Number of questions", 3, 8, 5)
 
    if st.button("Generate Questions"):
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
 
    st.subheader(f"Question {index + 1} of {total}")
    st.write(question)
 
    answer = st.text_area("Your answer", height=150, key=f"answer_box_{index}")
 
    if st.button("Submit Answer"):
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
    st.subheader("📊 Interview Report")
 
    scores = st.session_state.scores
    if len(scores) > 0:
        average_score = sum(scores) / len(scores)
    else:
        average_score = 0
 
    st.metric("Overall Score", f"{average_score:.1f} / 10")
 
    st.write("### Per-question scores")
    for i in range(len(st.session_state.questions)):
        question = st.session_state.questions[i]
        score = st.session_state.scores[i]
        feedback = st.session_state.feedback_list[i]
        with st.expander(f"Q{i + 1}: {question[:70]} — Score: {score}/10"):
            st.write(feedback)
 
    if st.session_state.missing_keywords_all:
        unique_keywords = sorted(set(st.session_state.missing_keywords_all))
        st.write("### 🔑 Keywords to study")
        st.write(", ".join(unique_keywords))
 
    if st.button("Start New Session"):
        st.session_state.stage = "setup"
        st.session_state.questions = []
        st.session_state.q_index = 0
        st.session_state.scores = []
        st.session_state.feedback_list = []
        st.session_state.missing_keywords_all = []
        st.rerun()