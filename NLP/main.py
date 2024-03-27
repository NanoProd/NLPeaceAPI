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

import data_processing as dp
import models
from logger_config import configure_logger
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import class_weight
import numpy as np

logger = configure_logger(__name__)


logger.info("Starting the NLP pipeline for hatespeech...")


df = dp.import_hate_data()

vectorizer = TfidfVectorizer(max_features=5000)

df['tweet'] = df['tweet'].apply(dp.preprocess)

X = vectorizer.fit_transform(df["tweet"])
df['label'] = df['class']
y = df["label"]

# Define class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
class_weights = dict(zip(np.unique(y), class_weights.flatten()))
print("class weight:")
print(class_weights)


#save the vectorizer
dump(vectorizer, 'models/hate_vectorizer.joblib')
logger.info("Vectorizer saved successfully.")

# Train models
rf_model, rf_score = models.train_random_forest(X, y, class_weights=class_weights)
xgb_model, xgb_score = models.train_xgboost(X, y, 3, class_weights=class_weights)
svm_model, svm_score = models.train_svm(X, y, class_weights=class_weights)
naive_model, naive_score = models.train_naive_bayes(X, y, class_weights=class_weights)
knn_model, knn_score = models.train_knn(X, y, class_weights=class_weights)

#find best model
best_model, best_score, best_model_name = None, 0, ''
if rf_score > best_score:
    best_model, best_score, best_model_name = rf_model, rf_score, 'Random Forest'
if xgb_score > best_score:
    best_model, best_score, best_model_name = xgb_model, xgb_score, 'XGBoost'
if svm_score > best_score:
    best_model, best_score, best_model_name = svm_model, svm_score, 'SVM'
if naive_score > best_score:
    best_model, best_score, best_model_name = naive_model, naive_score, 'Naive Bayes'
if knn_score > best_score:
    best_model, best_score, best_model_name = knn_model, knn_score, 'K Nearest Neighbor'


# Save the best model
if best_model:
    model_path = 'models/best_hate_model.joblib'
    dump(best_model, model_path)
    logger.info(f"Saved best model ({best_model_name}) with score: {best_score}")

