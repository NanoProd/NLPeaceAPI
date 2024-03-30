# MIT License

# Copyright (c) 2023 Fatima El Fouladi, Anum Siddiqui, Jeff Wilgus, David Lemme, Mira Aji, Adam Qamar, Shabia Saeed, Raya Maria Lahoud , Nelly Bozorgzad, Joshua-James Nantel-Ouimet .

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# data_processing.py

import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nlpaug.augmenter.word as naw
from sklearn.preprocessing import LabelEncoder
import os

# Constants for Neural Network
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 280

# Initialize NLTK and SpaCy
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

model = 'en_core_web_sm'
if not spacy.util.is_package(model):
    os.system(f"python -m spacy download {model}")

nlp = spacy.load(model, disable=['parser', 'ner'])

def import_hate_data():
    df = pd.read_csv('data/hatespeech.csv')
    df = df[['tweet', 'class']]
    #df['class'] = df['class'].replace(1, 0)
    return df

def import_emotion_data():
    df = pd.read_csv('data/emotion.csv', header=0)
    return df

def remove_emojis(text):
    emoji_pattern = re.compile("[" 
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def preprocess(text):
    text = re.sub(r'@\w+', '', text)
    text = remove_emojis(text)
    text = re.sub(r'\#\w+', '', text)
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    doc = nlp(" ".join(words))
    words = [token.lemma_ for token in doc]
    return " ".join(words)

def tokenize_and_pad_texts(texts):
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    return padded, tokenizer

def augment_text(text, num_augmentations=1):
    aug = naw.SynonymAug(aug_src='wordnet')
    augmented_texts = [aug.augment(text) for _ in range(num_augmentations)]
    return augmented_texts

def get_label_encoder(labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    return label_encoder