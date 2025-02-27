from flask import Flask, request, jsonify
from resume_screening.resume_screening import extract_keywords, calculate_similarity
from sentiment_analysis.sentiment_analysis import sentiment_analyzer, predict_attrition_risk

app = Flask(__name__)

@app.route("/screen_resume", methods=["POST"])
def screen_resume():
    resume_text = request.json["resume_text"]
    job_description = request.json["job_description"]
    keywords = extract_keywords(resume_text)
    similarity_score = calculate_similarity(resume_text, job_description)
    return jsonify({"keywords": keywords, "similarity_score": similarity_score})

@app.route("/analyze_sentiment", methods=["POST"])
def analyze_sentiment():
    feedback = request.json["feedback"]
    sentiment, score = sentiment_analyzer(feedback)
    attrition_risk = predict_attrition_risk(feedback)
    return jsonify({"sentiment": sentiment, "score": score, "attrition_risk": attrition_risk})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
