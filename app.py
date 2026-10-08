from flask import Flask, render_template, request, jsonify
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import time

load_dotenv()

app = Flask(__name__)

# =========================
# Gemini API
# =========================

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(
    api_key=api_key
)

# =========================
# ChromaDB
# =========================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


# =========================
# Home Page
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# Chat API
# =========================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_message = data.get("message", "").strip()

    education = data.get("education", "")
    skills = data.get("skills", "")
    interests = data.get("interests", "")
    level = data.get("level", "")
    goal = data.get("goal", "")

    if user_message == "":
        return jsonify({
            "reply": "Please enter a question."
        })

    # =========================
    # Search ChromaDB
    # =========================

    try:

        results = db.similarity_search(
            user_message,
            k=3
        )

    except Exception as e:

        print("ChromaDB error:", e)

        return jsonify({
            "reply": "Error searching career information."
        }), 500

    # =========================
    # Create Context
    # =========================

    context = "\n\n".join(
        document.page_content
        for document in results
    )

    # =========================
    # Prompt
    # =========================

    prompt = f"""
You are an AI Career Guidance Chatbot.

You help students choose suitable technology careers
and create learning roadmaps.

Use the career information retrieved from the
Career Guidance PDF as your main knowledge source.

=========================
STUDENT PROFILE
=========================

Education:
{education}

Skills:
{skills}

Interests:
{interests}

Skill Level:
{level}

Career Goal:
{goal}

=========================
CAREER INFORMATION
=========================

{context}

=========================
STUDENT QUESTION
=========================

{user_message}

=========================
INSTRUCTIONS
=========================

Give a simple, clear and practical answer.

Consider the student's profile when giving advice.

When suitable, include:

1. Recommended career
2. Why it suits the student
3. Required skills
4. Skills the student should improve
5. Learning roadmap
6. Projects to build
7. Possible job roles

Do not invent information from the PDF.

If the required information is not available in the
knowledge base, clearly say that it is not available
in the knowledge base.

Use simple English suitable for a college student.
"""

    # =========================
    # Gemini Response
    # =========================

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            return jsonify({
                "reply": response.text
            })

        except Exception as e:

            print("Gemini error:", e)

            if attempt < 2:
                time.sleep(2)

            else:
                return jsonify({
                    "reply": "Gemini is temporarily busy. Please try again."
                }), 503


# =========================
# Run Application
# =========================

if __name__ == "__main__":
    app.run(debug=True)