import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# Load dataset
data = pd.read_csv("employee_feedback.csv")

# Preprocess text
def preprocess_text(text):
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()

# Sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis")

# Train attrition risk model
X = data["feedback"]
y = data["attrition_risk"]

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict attrition risk
def predict_attrition_risk(feedback):
    feedback_vectorized = vectorizer.transform([feedback])
    prediction = model.predict(feedback_vectorized)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    feedback = "I feel undervalued and overworked."
    sentiment, score = sentiment_analyzer(feedback)
    attrition_risk = predict_attrition_risk(feedback)
    print("Sentiment:", sentiment)
    print("Sentiment Score:", score)
    print("Attrition Risk:", attrition_risk)
