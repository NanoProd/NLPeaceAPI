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
from joblib import dump

logger = configure_logger(__name__)


logger.info("Starting the NLP pipeline...")
df = dp.import_hate_data()
df = dp.process_data(df)
X, y, vectorizer = models.vectorize_data(df)

#save the vectorizer
dump(vectorizer, 'models/hate_vectorizer.joblib')
logger.info("Vectorizer saved successfully.")

# Train models on the new dataset
rf_model, rf_score = models.train_random_forest(X, y)
xgb_model, xgb_score = models.train_xgboost(X, y, 3)
nb_model, nb_score = models.train_naive_bayes(X, y)
lr_model, lr_score = models.train_logistic_regression(X, y)
svm_model, svm_score = models.train_svm(X, y)

# Compare all models to find the best
best_model, best_score, best_model_name = None, 0, ''

# Find and save the best model for the emotion dataset
best_model, best_score, best_model_name = None, 0, ''
if rf_score > best_score:
    best_model, best_score, best_model_name = rf_model, rf_score, 'Random Forest'
if xgb_score > best_score:
    best_model, best_score, best_model_name = xgb_model, xgb_score, 'XGBoost'
if nb_score > best_score:
    best_model, best_score, best_model_name = nb_model, nb_score, 'NaiveBayes'
if lr_score > best_score:
    best_model, best_score, best_model_name = lr_model, lr_score, 'LogisticRegression'
if svm_score > best_score:
    best_model, best_score, best_model_name = svm_model, svm_score, 'SVM'

# Save the best model
if best_model:
    model_path = 'models/best_hate_model.joblib'
    dump(best_model, model_path)
    logger.info(f"Saved best model ({best_model_name}) with score: {best_score}")

