import joblib
import os

# Function to load a model and vectorizer
def load_model_and_vectorizer(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Function to make a prediction
def make_prediction(model, vectorizer, text):
    processed_tweet = vectorizer.transform([text])
    prediction = model.predict(processed_tweet)
    return prediction

# Paths to models and vectorizers
current_directory = os.getcwd()
hate_model_path = os.path.join(current_directory, "NLP", "models", "best_hate_model.joblib")
hate_vectorizer_path = os.path.join(current_directory, "NLP", "models", "hate_vectorizer.joblib")
emotion_model_path = os.path.join(current_directory, "NLP", "models", "best_model_emotion.joblib")
emotion_vectorizer_path = os.path.join(current_directory, "NLP", "models", "vectorizer_emotion.joblib")

# Load models and vectorizers
hate_model, hate_vectorizer = load_model_and_vectorizer(hate_model_path, hate_vectorizer_path)
emotion_model, emotion_vectorizer = load_model_and_vectorizer(emotion_model_path, emotion_vectorizer_path)

# Labels for predictions
hate_label_mapping = {0: "hate speech", 1: "offensive language", 2: "neutral"}
emotion_label_mapping = {0: "anger", 1: "fear", 2: "joy", 3: "sadness"}

# Test tweets
sample_tweets = ["fuck you bitch!", "I wanna cry"]

# Testing Hate Speech Model
print("Testing Hate Speech Model")
for tweet in sample_tweets:
    prediction = make_prediction(hate_model, hate_vectorizer, tweet)
    print(f"Tweet: {tweet}")
    print(f"Prediction: {hate_label_mapping[prediction[0]]}\n")

# Testing Emotion Model
print("Testing Emotion Model")
for tweet in sample_tweets:
    prediction = make_prediction(emotion_model, emotion_vectorizer, tweet)
    print(f"Tweet: {tweet}")
    print(f"Prediction: {emotion_label_mapping[prediction[0]]}\n")
