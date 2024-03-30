from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import spacy
import logging
import data_processing as dp
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
import pickle
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Function to load a model using TFSMLayer
def load_model_with_tfsmlayer(model_path):
    model = tf.keras.Sequential()
    model.add(TFSMLayer(model_path, call_endpoint='serving_default'))
    return model

# Function to load a Keras model
def load_keras_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        raise

# Load neural network models with TFSMLayer
hate_model_path = os.path.join(current_directory, "NLP", "models", "best_hate_model")
emotion_model_path = os.path.join(current_directory, "NLP", "models", "best_emotion_model")

try:
    hate_model = load_keras_model(hate_model_path)
    logger.info("Successfully loaded the hate model.")
except Exception as e:
    logger.error(f"Error loading hate model: {str(e)}")
    raise

try:
    emotion_model = load_keras_model(emotion_model_path)
    logger.info("Successfully loaded the emotion model.")
except Exception as e:
    logger.error(f"Error loading emotion model: {str(e)}")
    raise

# Load tokenizers
hate_tokenizer_path = os.path.join(current_directory, "NLP", "models", "hate_tokenizer.pickle")
emotion_tokenizer_path = os.path.join(current_directory, "NLP", "models", "emotion_tokenizer.pickle")
#tokenizer_path = os.path.join(current_directory, "NLP", "models", "tokenizer.pickle")
try:
    with open(hate_tokenizer_path, 'rb') as handle:
        hate_tokenizer = pickle.load(handle)
    logger.info("Successfully loaded the hate tokenizer.")

    with open(emotion_tokenizer_path, 'rb') as handle:
       emotion_tokenizer = pickle.load(handle)
    logger.info("Successfully loaded the emotion tokenizer.")
    # with open(tokenizer_path, 'rb') as handle:
    #     tokenizer = pickle.load(handle)
    # logger.info("Successfully loaded the hate tokenizer.")
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

