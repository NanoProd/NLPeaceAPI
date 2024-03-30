import os
import sys
import tensorflow as tf
import pickle
import numpy as np
import data_processing as dp

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

# Configure directory
current_directory = os.path.dirname(os.path.abspath(__file__))

def load_keras_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        raise

# Load models
hate_model_path = os.path.join(current_directory, "models", "best_hate_model.keras")
emotion_model_path = os.path.join(current_directory, "models", "best_emotion_model.keras")

hate_model = load_keras_model(hate_model_path)
emotion_model = load_keras_model(emotion_model_path)

# Load tokenizers
hate_tokenizer_path = os.path.join(current_directory, "models", "hate_tokenizer.pickle")
emotion_tokenizer_path = os.path.join(current_directory, "models", "emotion_tokenizer.pickle")

try:
    with open(hate_tokenizer_path, 'rb') as handle:
        hate_tokenizer = pickle.load(handle)
    print("Successfully loaded the hate tokenizer.")

    with open(emotion_tokenizer_path, 'rb') as handle:
        emotion_tokenizer = pickle.load(handle)
    print("Successfully loaded the emotion tokenizer.")
except Exception as e:
    print(f"Error loading tokenizers: {str(e)}")
    raise

def prepare_text_for_nn(text, tokenizer):
    processed_text = dp.preprocess(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=280)
    return padded_sequence

def classify_text(text, model, tokenizer, class_names):
    padded_sequence = prepare_text_for_nn(text, tokenizer)
    raw_predictions = model.predict(padded_sequence)
    
    # Print class probabilities
    for i, class_name in class_names.items():
        print(f"Probability of '{class_name}': {raw_predictions[0][i]:.4f}")
    
    predicted_class = np.argmax(raw_predictions, axis=1)[0]
    return class_names.get(predicted_class, "Unknown")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_text = " ".join(sys.argv[1:])
        print("Hate speech classification:")
        hate_result = classify_text(test_text, hate_model, hate_tokenizer, hate_class_names)
        print(f"Predicted class: {hate_result}\n")
        
        print("Emotion classification:")
        emotion_result = classify_text(test_text, emotion_model, emotion_tokenizer, emotion_class_names)
        print(f"Predicted class: {emotion_result}\n")
    else:
        print("Please provide text to classify. Usage: python script.py 'Text-To-Classify'")
