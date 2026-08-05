"""
GenAI Mock-Interview Coach - REST API
----------------------------------------
Exposes the same core logic as app.py through a FastAPI REST API, so the
project can be used by other programs/frontends, not just the Streamlit UI.

Run:
    uvicorn api.py:app --reload

Then open http://127.0.0.1:8000/docs to see and test the API in your browser.

Requires:
    pip install fastapi uvicorn google-genai
    Set the GEMINI_API_KEY environment variable first.
"""

import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import errors as genai_errors

from dotenv import load_dotenv
load_dotenv()

MODEL = "gemini-flash-latest"
SECONDS_BETWEEN_CALLS = 13
_last_call_time = [0.0]

app = FastAPI(
    title="GenAI Mock-Interview Coach API",
    description="Generate tailored interview questions and score candidate answers using Gemini.",
    version="1.0.0",
)


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set on the server.")
    return genai.Client(api_key=api_key)


def wait_for_rate_limit():
    elapsed = time.time() - _last_call_time[0]
    remaining = SECONDS_BETWEEN_CALLS - elapsed
    if remaining > 0:
        time.sleep(remaining)
    _last_call_time[0] = time.time()


def ask_gemini(client, prompt, max_retries=3):
    for attempt in range(max_retries):
        wait_for_rate_limit()
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            return response.text or ""
        except genai_errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                time.sleep(20)
            else:
                raise HTTPException(status_code=429, detail="Gemini API rate limit exceeded. Try again shortly.")
    return ""


# ---------------------------------------------------------------------------
# Request/response schemas - these define exactly what JSON shape the API
# accepts and returns. FastAPI uses these to auto-generate docs and validate
# incoming requests for you.
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    resume: str
    job_description: str
    num_questions: int = 5


class QuestionResponse(BaseModel):
    questions: list[str]


class ScoreRequest(BaseModel):
    question: str
    answer: str
    job_description: str


class ScoreResponse(BaseModel):
    score: int
    feedback: str
    missing_keywords: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "GenAI Mock-Interview Coach API is running. Visit /docs to try it out."}


@app.post("/generate-questions", response_model=QuestionResponse)
def generate_questions_endpoint(request: QuestionRequest):
    client = get_client()
    prompt = (
        "You are a senior technical interviewer. Given a candidate's resume and a "
        "target job description, write interview questions that test whether the "
        "candidate can actually do the job. Mix background questions, technical "
        "questions based on the job description, and one behavioral question. "
        "Reply with ONLY the questions, one per line, with no numbering, no "
        "introduction, and no extra text.\n\n"
        f"RESUME:\n{request.resume}\n\n"
        f"JOB DESCRIPTION:\n{request.job_description}\n\n"
        f"Write exactly {request.num_questions} interview questions, one per line."
    )
    reply_text = ask_gemini(client, prompt)
    questions = [line.strip() for line in reply_text.split("\n") if line.strip()]
    return QuestionResponse(questions=questions)


@app.post("/score-answer", response_model=ScoreResponse)
def score_answer_endpoint(request: ScoreRequest):
    client = get_client()
    prompt = (
        "You are a strict but fair technical interviewer. Score the candidate's "
        "answer against the job description on relevance, correctness, and "
        "completeness. Reply in EXACTLY this plain text format, three lines, "
        "nothing else:\n"
        "SCORE: <a whole number from 0 to 10>\n"
        "FEEDBACK: <2-3 direct, actionable sentences>\n"
        "MISSING: <comma-separated list of up to 3 missing keywords or concepts, "
        "or NONE if nothing important was missing>\n\n"
        f"JOB DESCRIPTION:\n{request.job_description}\n\n"
        f"QUESTION:\n{request.question}\n\n"
        f"CANDIDATE ANSWER:\n{request.answer}\n\n"
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

    return ScoreResponse(score=score, feedback=feedback, missing_keywords=missing_keywords)