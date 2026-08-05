"""
Test & Log Script for GenAI Mock-Interview Coach
----------------------------------------------------
This script runs your app's core logic multiple times against sample
data, measures real performance numbers, and saves them to a report file.
 
Use the numbers this produces on your resume - they come from actually
running your code, not guesses.
 
Run:
    python test_and_log.py
 
Requires:
    pip install google-genai
    Set the GEMINI_API_KEY environment variable first.
"""
 
import os
import time
import statistics
from google import genai
from google.genai import errors as genai_errors
from dotenv import load_dotenv
load_dotenv()
 
MODEL = "gemini-flash-latest"
 
# Free tier allows only 5 requests per minute. We wait this long between
# every single API call so we never hit that limit.
SECONDS_BETWEEN_CALLS = 13
 
_last_call_time = [0.0]  # stored in a list so it can be updated inside a function
 
 
def wait_for_rate_limit():
    """Pause just long enough since the last call to stay under 5 requests/minute."""
    elapsed_since_last_call = time.time() - _last_call_time[0]
    remaining_wait = SECONDS_BETWEEN_CALLS - elapsed_since_last_call
    if remaining_wait > 0:
        time.sleep(remaining_wait)
    _last_call_time[0] = time.time()
 
# ---------------------------------------------------------------------------
# Core logic (same as app.py, without any Streamlit dependency so it can
# run as a plain script)
# ---------------------------------------------------------------------------
 
def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set the GEMINI_API_KEY environment variable first.")
        exit(1)
    return genai.Client(api_key=api_key)
 
 
def ask_gemini(client, prompt, max_retries=3):
    """Call Gemini, respecting the rate limit and retrying if we still get rate-limited."""
    for attempt in range(max_retries):
        wait_for_rate_limit()
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            return response.text or ""
        except genai_errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                print(f"  Rate limit hit, waiting 20s before retry {attempt + 2}/{max_retries}...")
                time.sleep(20)
            else:
                raise
    return ""
 
 
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
    return [line.strip() for line in reply_text.split("\n") if line.strip()]
 
 
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
            try:
                score = int(line.replace("SCORE:", "").strip())
            except ValueError:
                score = 0
        elif line.startswith("FEEDBACK:"):
            feedback = line.replace("FEEDBACK:", "").strip()
        elif line.startswith("MISSING:"):
            missing_text = line.replace("MISSING:", "").strip()
            if missing_text.upper() != "NONE":
                missing_keywords = [w.strip() for w in missing_text.split(",")]
    return score, feedback, missing_keywords
 
 
# ---------------------------------------------------------------------------
# Sample test data
# ---------------------------------------------------------------------------
 
SAMPLE_RESUME = """
Manojna Sankena
AI/ML undergraduate with hands-on experience in end-to-end machine learning
pipelines, supervised learning, NLP, and data-driven web development.
Skilled in Python, SQL, Scikit-learn, Pandas, EDA, and feature engineering.
Built a job market trend analyzer scraping 1,000+ postings and a subscription
anomaly-detection system with a normalized SQL schema.
"""
 
SAMPLE_JD = """
AI/ML Intern. Responsibilities: build and evaluate ML models using Scikit-learn
or TensorFlow, work with NLP for text classification, clean and analyze data
with Pandas, deploy models via REST APIs, visualize results with Power BI.
Requirements: strong Python and SQL, understanding of supervised learning and
model evaluation, exposure to a deep learning framework and REST APIs is a plus.
"""
 
# A few sample answers of varying quality, used to test the scoring function
# with a realistic spread of responses (not just one easy case repeated).
SAMPLE_ANSWERS = [
    "I used Pandas to clean the dataset, removed nulls and duplicates, then "
    "trained a Scikit-learn classifier and evaluated it with cross-validation.",
    "I just used some Python to look at the data and made a model with sklearn.",
    "For text classification I would use a bag-of-words or embeddings approach, "
    "train a classifier like logistic regression, and evaluate using precision, "
    "recall, and F1-score since class imbalance is common in real text data.",
]
 
CONSISTENCY_TEST_ANSWER = SAMPLE_ANSWERS[2]
CONSISTENCY_TEST_QUESTION = "How would you approach a text classification task with imbalanced classes?"
 
 
# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
 
def run_tests(num_sessions=2, questions_per_session=3, consistency_trials=3):
    client = get_client()
 
    question_gen_times = []
    scoring_times = []
    all_scores = []
    all_missing_keywords = []
    total_questions_generated = 0
    total_answers_scored = 0
 
    total_calls = num_sessions * (1 + questions_per_session) + consistency_trials
    estimated_minutes = (total_calls * SECONDS_BETWEEN_CALLS) / 60
    print(f"Running {num_sessions} test sessions, {questions_per_session} questions each...")
    print(f"This will make {total_calls} API calls, spaced to respect the free-tier rate limit.")
    print(f"Estimated time: ~{estimated_minutes:.1f} minutes. Please let it run without interrupting.\n")
 
    for session in range(1, num_sessions + 1):
        print(f"Session {session}/{num_sessions}...")
 
        start = time.time()
        questions = generate_questions(client, SAMPLE_RESUME, SAMPLE_JD, questions_per_session)
        elapsed = time.time() - start
        question_gen_times.append(elapsed)
        total_questions_generated += len(questions)
 
        for i, question in enumerate(questions):
            sample_answer = SAMPLE_ANSWERS[i % len(SAMPLE_ANSWERS)]
 
            start = time.time()
            score, feedback, missing = score_answer(client, question, sample_answer, SAMPLE_JD)
            elapsed = time.time() - start
            scoring_times.append(elapsed)
 
            all_scores.append(score)
            all_missing_keywords.extend(missing)
            total_answers_scored += 1
 
    # --- Consistency test: score the SAME answer multiple times ---
    print(f"\nRunning consistency test ({consistency_trials} repeated trials on the same answer)...")
    consistency_scores = []
    for _ in range(consistency_trials):
        score, _, _ = score_answer(client, CONSISTENCY_TEST_QUESTION, CONSISTENCY_TEST_ANSWER, SAMPLE_JD)
        consistency_scores.append(score)
 
    # --- Build report ---
    unique_keywords = sorted(set(all_missing_keywords))
 
    report_lines = []
    report_lines.append("GenAI Mock-Interview Coach - Test Report")
    report_lines.append("=" * 50)
    report_lines.append(f"Test sessions run: {num_sessions}")
    report_lines.append(f"Total interview questions generated: {total_questions_generated}")
    report_lines.append(f"Total candidate answers scored: {total_answers_scored}")
    report_lines.append("")
    report_lines.append(f"Average question-generation response time: {statistics.mean(question_gen_times):.2f}s")
    report_lines.append(f"Average answer-scoring response time: {statistics.mean(scoring_times):.2f}s")
    report_lines.append("")
    report_lines.append(f"Score range across all test answers: {min(all_scores)}-{max(all_scores)} out of 10")
    report_lines.append(f"Average score across all test answers: {statistics.mean(all_scores):.1f}/10")
    report_lines.append("")
    report_lines.append(f"Consistency test (same answer scored {consistency_trials} times): {consistency_scores}")
    if len(consistency_scores) > 1:
        report_lines.append(f"Consistency std deviation: {statistics.stdev(consistency_scores):.2f} points")
        report_lines.append(f"Consistency range: {max(consistency_scores) - min(consistency_scores)} points")
    report_lines.append("")
    report_lines.append(f"Unique missing-keywords surfaced across all tests: {len(unique_keywords)}")
    report_lines.append(f"Keywords: {', '.join(unique_keywords)}")
 
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
 
    with open("test_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("\nSaved full report to test_report.txt")
 
 
if __name__ == "__main__":
    run_tests()