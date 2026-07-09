import os
import streamlit as st
import pdfplumber
from groq import Groq
from dotenv import load_dotenv

# This script builds a simple Streamlit app that uploads a resume,
# extracts its text, and sends it to a Groq LLM for analysis.

# ----------------- PDF READING ----------------- #

def read_pdf(file):
    """Extract text from a PDF file page by page."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"

    return text


# ----------------- LLM ANALYSIS ----------------- #

def analyze_resume(text):
    """Send the resume text to the Groq LLM and return the analysis result."""
    prompt = f"""
    You are an expert resume parser.

    Task:
    1. Extract EXPERIENCE in clean bullet points (concise, professional)
    2. SIMPLIFY SKILLS into categories (Programming, Tools, Cloud, Database, Domain)

    Resume:
    {text}

    Output format:

    Experience:
    - ...

    Skills:
    - Programming:
    - Tools:
    - Cloud:
    - Database:
    - Domain:
    """

    # Load environment variables from .env so the API key stays private.
    load_dotenv()

    # Create a Groq client using the API key from the environment.
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


# ----------------- STREAMLIT UI ----------------- #

# Configure the Streamlit page title and layout.
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Display the app title.
st.title("🤖 AI Resume Analyzer (LLM Powered)")

# Let the user upload a resume file in PDF or TXT format.
uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "txt"])

# Main logic for handling the uploaded resume.
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        resume_text = read_pdf(uploaded_file)
    else:
        resume_text = uploaded_file.read().decode("utf-8")

    st.success("Resume uploaded successfully!")

    # Trigger the AI analysis when the user clicks the button.
    if st.button("Analyze with AI"):
        with st.spinner("Processing with LLM..."):
            result = analyze_resume(resume_text)

        st.subheader("📊 AI Output")
        st.text_area("Result", result, height=400)

# Uncomment the next line if you want to test the analyzer outside the UI.
# print(analyze_resume("Python developer with AWS and SQL experience"))