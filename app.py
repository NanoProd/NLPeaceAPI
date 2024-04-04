from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import spacy
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download the model if it's not already installed
try:
    spacy.load("en_core_web_sm")
    logger.info("Successfully loaded SpaCy model.")
except Exception as e:
    logger.error(f"Failed to load SpaCy model: {str(e)}")
    from spacy.cli import download
    download("en_core_web_sm")
    spacy.load("en_core_web_sm")

#configure directory
current_directory = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Current directory set to {current_directory}")

#load models
model_path = os.path.join(current_directory, "NLP", "models", "best_hate_model.joblib")
try:
    model = joblib.load(model_path)
    logger.info("Successfully loaded the model.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Load vectorizer
vectorizer_path = os.path.join(current_directory, "NLP", "models", "hate_vectorizer.joblib")
try:
    vectorizer = joblib.load(vectorizer_path)
    logger.info("Successfully loaded the vectorizer.")
except Exception as e:
    logger.error(f"Error loading vectorizer: {str(e)}")
    raise

# Load Emotion Model
emotion_model_path = os.path.join(current_directory, "NLP", "models", "best_model_emotion.joblib")
try:
    emotion_model = joblib.load(emotion_model_path)
    logger.info("Successfully loaded the emotion model.")
except Exception as e:
    logger.error(f"Error loading emotion model: {str(e)}")
    raise

# Load Emotion Vectorizer
emotion_vectorizer_path = os.path.join(current_directory, "NLP", "models", "vectorizer_emotion.joblib")
try:
    emotion_vectorizer = joblib.load(emotion_vectorizer_path)
    logger.info("Successfully loaded the emotion vectorizer.")
except Exception as e:
    logger.error(f"Error loading emotion vectorizer: {str(e)}")
    raise

#define app
app = FastAPI()


#define tweet class
class Tweet(BaseModel):
    text: str

label_mapping = {0: "hate speech", 1: "offensive language", 2: "neutral"}
emotion_label_mapping = {0: "anger", 1: "fear", 2: "joy", 3: "sadness"}

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/classify/")
async def classify_tweet(tweet: Tweet):
    if not tweet.text:
        return {"prediction": -1, "description": "Invalid tweet"}
    try:
        # Preprocess and vectorize the tweet text
        processed_tweet = vectorizer.transform([tweet.text])
        prediction = model.predict(processed_tweet)
        # Translate the numeric prediction to a label
        prediction_label = label_mapping.get(prediction[0], "Unknown")
        return {"prediction": prediction.tolist(), "description": prediction_label}
    except Exception as e:
        # Log the tweet text and the exception
        logger.error(f"Error processing tweet: {tweet.text}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/classify_emotion/")
async def classify_emotion(tweet: Tweet):
    if not tweet.text:
        return {"prediction": -1, "description": "Invalid tweet"}
    try:
        # Preprocess and vectorize the tweet text
        processed_tweet = emotion_vectorizer.transform([tweet.text])
        prediction = emotion_model.predict(processed_tweet)
        # Translate the numeric prediction to an emotion label
        prediction_label = emotion_label_mapping.get(prediction[0], "Unknown")
        return {"prediction": prediction.tolist(), "description": prediction_label}
    except Exception as e:
        # Log the tweet text and the exception
        logger.error(f"Error processing tweet: {tweet.text}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")