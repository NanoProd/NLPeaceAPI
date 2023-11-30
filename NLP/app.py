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
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    spacy.load("en_core_web_sm")

#configure directory
current_directory = os.path.dirname(os.path.abspath(__file__))

#load models
model_path = os.path.join(current_directory, "models", "best_model.joblib")
model = joblib.load(model_path)

#load vectorizer
vectorizer_path = os.path.join(current_directory, "models", "vectorizer.joblib")
vectorizer = joblib.load(vectorizer_path)

#define app
app = FastAPI()

#define tweet class
class Tweet(BaseModel):
    text: str


@app.post("/classify/")
async def classify_tweet(tweet: Tweet):
    if not tweet.text:
        return {"prediction": -1}
    try:
        #preprocess and vectorize the tweet text
        processed_tweet = vectorizer.transform([tweet.text])
        prediction = model.predict(processed_tweet)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        # Log the tweet text and the exception
        logger.error(f"Error processing tweet: {tweet.text}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
