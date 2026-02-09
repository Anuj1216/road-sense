# ------------------------------
# SkillGate / HireSense Prototype
# ------------------------------

# 1️⃣ Install required libraries:
# pip install pdfplumber scikit-learn nltk

import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ------------------------------
# Step 1: Read PDF Resume
# ------------------------------
def read_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text.lower()  # lowercase for easy comparison

# ------------------------------
# Step 2: Define Job Requirements
# ------------------------------
must_have = [
    "bachelor",
    "2 years experience"
]

optional_skills = [
    "sales",
    "communication",
    "crm"
]

# ------------------------------
# Step 3: Check Mandatory Requirements
# ------------------------------
def check_mandatory(resume_text, must_have):
    missing = []
    for requirement in must_have:
        if requirement not in resume_text:
            missing.append(requirement)
    return missing  # empty list → all requirements satisfied

# ------------------------------
# Step 4: Score Optional Requirements
# ------------------------------
def score_optional(resume_text, optional_skills):
    score = 0
    for skill in optional_skills:
        if skill in resume_text:
            score += 1
    return score

# ------------------------------
# Step 5: Semantic Similarity (TF-IDF + Cosine)
# ------------------------------
def semantic_similarity(resume_text, requirement_text):
    documents = [resume_text, requirement_text]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    sim = cosine_similarity(vectors[0], vectors[1])
    return sim[0][0]

# ------------------------------
# Step 6: Process All Resumes
# ------------------------------
resume_folder = "resumes"  # folder containing PDFs
results = []

for filename in os.listdir(resume_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(resume_folder, filename)
        resume_text = read_pdf(file_path)

        # Mandatory check
        missing_mandatory = check_mandatory(resume_text, must_have)
        if missing_mandatory:
            results.append({
                "resume": filename,
                "status": "Rejected",
                "missing": missing_mandatory,
                "score": 0
            })
            continue  # skip scoring

        # Optional scoring
        opt_score = score_optional(resume_text, optional_skills)

        # Semantic similarity (bonus)
        sem_scores = []
        for req in must_have + optional_skills:
            sim = semantic_similarity(resume_text, req)
            sem_scores.append(sim)
        semantic_score = sum(sem_scores) / len(sem_scores)

        # Final Score (simple weighted)
        final_score = 0.6 + 0.3 * opt_score + 0.1 * semantic_score  # mandatory pass = 0.6
        results.append({
            "resume": filename,
            "status": "Accepted",
            "missing": [],
            "score": round(final_score, 2)
        })

# ------------------------------
# Step 7: Rank Resumes
# ------------------------------
ranked = sorted(results, key=lambda x: x["score"], reverse=True)

# ------------------------------
# Step 8: Display Results
# ------------------------------
print("\n======= Resume Screening Results =======\n")
for res in ranked:
    print(f"Resume: {res['resume']}")
    print(f"Status: {res['status']}")
    if res['missing']:
        print(f"Missing Mandatory: {', '.join(res['missing'])}")
    print(f"Score: {res['score']}")
    print("-------------------------------------\n")
