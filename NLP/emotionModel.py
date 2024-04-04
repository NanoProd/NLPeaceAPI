# main.py
import data_processing as dp
import models
from logger_config import configure_logger
from joblib import dump

logger = configure_logger(__name__)

# Import and process the new dataset
logger.info("Starting the NLP pipeline for emotion dataset...")
df = dp.import_data('data/emotion.csv')
df = dp.process_emotion_data(df)
X, y, vectorizer = models.vectorize_emotion_data(df)

y, encoding_map = dp.encode_labels(df['label'])
logger.info(f"Label encoding map: {encoding_map}")

# Save the vectorizer
dump(vectorizer, 'models/vectorizer_emotion.joblib')
logger.info("Vectorizer for emotion dataset saved successfully.")

rf_model, rf_score = models.train_random_forest(X, y)
xgb_model, xgb_score = models.train_xgboost(X, y, 4)
nb_model, nb_score = models.train_naive_bayes(X, y)
lr_model, lr_score = models.train_logistic_regression(X, y)
svm_model, svm_score = models.train_svm(X, y)

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

# Save the best model for the emotion dataset
if best_model:
    model_path = 'models/best_model_emotion.joblib'
    dump(best_model, model_path)
    logger.info(f"Saved best emotion model ({best_model_name}) with score: {best_score}")
