from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import spacy
import logging
import data_processing as dp
import tensorflow as tf
import pickle
import numpy as np


#custom loss function
import tensorflow.keras.backend as K

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download the Spacy model if it's not already installed
try:
    spacy.load("en_core_web_sm")
    logger.info("Successfully loaded SpaCy model.")
except Exception as e:
    logger.error(f"Failed to load SpaCy model: {str(e)}")
    from spacy.cli import download
    download("en_core_web_sm")
    spacy.load("en_core_web_sm")

# Configure directory
current_directory = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Current directory set to {current_directory}")

# Load neural network models
hate_model_path = os.path.join(current_directory, "NLP", "models", "best_hate_model")
emotion_model_path = os.path.join(current_directory, "NLP", "models", "best_emotion_model")

try:
    hate_model = tf.keras.models.load_model(hate_model_path, custom_objects={'f1_score': f1_score})
    logger.info("Successfully loaded the hate model.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

try:
    emotion_model = tf.keras.models.load_model(emotion_model_path, custom_objects={'f1_score': f1_score})
    logger.info("Successfully loaded the emotion model.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Load tokenizers
hate_tokenizer_path = os.path.join(current_directory, "NLP", "models", "hate_tokenizer.pickle")
emotion_tokenizer_path = os.path.join(current_directory, "NLP", "models", "emotion_tokenizer.pickle")

try:
    with open(hate_tokenizer_path, 'rb') as handle:
        hate_tokenizer = pickle.load(handle)
    logger.info("Successfully loaded the hate tokenizer.")

    with open(emotion_tokenizer_path, 'rb') as handle:
        emotion_tokenizer = pickle.load(handle)
    logger.info("Successfully loaded the emotion tokenizer.")
except Exception as e:
    logger.error(f"Error loading tokenizers: {str(e)}")
    raise

# Define class labels
hate_class_names = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither"
}

emotion_class_names = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear"
}

# Define app
app = FastAPI()

# Define tweet class
class Tweet(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Helper function to prepare text for neural network
def prepare_text_for_nn(text, tokenizer):
    processed_text = dp.preprocess(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=280)
    return padded_sequence

@app.post("/classify/")
async def classify_hatespeech(tweet: Tweet):
    if not tweet.text:
        return {"prediction": -1, "class_name": "Invalid Input"}
    try:
        padded_sequence = prepare_text_for_nn(tweet.text, hate_tokenizer)
        prediction = hate_model.predict(padded_sequence)
        class_label = int(np.argmax(prediction, axis=1)[0])

        # Adjust the mapping of predictions
        if class_label == 1:
            class_label = 2

        class_name = hate_class_names.get(class_label, "Unknown")
        return {"prediction": class_label, "class_name": class_name}
    except Exception as e:
        logger.error(f"Error processing tweet: {tweet.text}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/classify-hatespeech/")
async def classify_hatespeech(tweet: Tweet):
    if not tweet.text:
        return {"prediction": -1, "class_name": "Invalid Input"}
    try:
        padded_sequence = prepare_text_for_nn(tweet.text, hate_tokenizer)
        prediction = hate_model.predict(padded_sequence)
        class_label = int(np.argmax(prediction, axis=1)[0])

        # Adjust the mapping of predictions
        if class_label == 1:
            class_label = 2

        class_name = hate_class_names.get(class_label, "Unknown")
        return {"prediction": class_label, "class_name": class_name}
    except Exception as e:
        logger.error(f"Error processing tweet: {tweet.text}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.post("/classify-emotion/")
async def classify_emotion(tweet: Tweet):
    if not tweet.text:
        return {"prediction": -1, "class_name": "Invalid Input"}
    try:
        padded_sequence = prepare_text_for_nn(tweet.text, emotion_tokenizer)
        prediction = emotion_model.predict(padded_sequence)
        class_label = int(np.argmax(prediction, axis=1)[0])
        class_name = emotion_class_names.get(class_label, "Unknown")
        return {"prediction": class_label, "class_name": class_name}
    except Exception as e:
        logger.error(f"Error processing tweet: {tweet.text}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

