import PyPDF2
from docx import Document
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Clean text
def clean_text(text):
    doc = nlp(text)
    cleaned_text = " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])
    return cleaned_text

# Extract keywords using GPT-4
def extract_keywords(resume_text):
    openai.api_key = "your_openai_api_key"
    prompt = f"""
    Extract the following details from the resume:
    - Skills
    - Years of experience
    - Educational qualifications
    - Certifications

    Return the output in JSON format.

    Resume Text: {resume_text}
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text

# Calculate similarity
def calculate_similarity(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

# Example usage
if __name__ == "__main__":
    resume_text = extract_text_from_pdf("resume.pdf")
    cleaned_resume_text = clean_text(resume_text)
    with open("job_description.txt", "r") as file:
        job_description = file.read()
    keywords = extract_keywords(cleaned_resume_text)
    similarity_score = calculate_similarity(cleaned_resume_text, job_description)
    print("Extracted Keywords:", keywords)
    print("Similarity Score:", similarity_score)
