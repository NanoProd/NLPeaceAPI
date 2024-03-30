# Import necessary libraries
import data_processing as dp
import models
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from logger_config import configure_logger

# Configure logger
logger = configure_logger(__name__)
logger.info("Starting the NLP pipeline for hate speech...")

# Load and preprocess data
df = dp.import_hate_data()
df['tweet'] = df['tweet'].apply(dp.preprocess)

# Log the number of samples per class
class_distribution = df['class'].value_counts()
logger.info(f"Number of samples per class after preprocessing:\n{class_distribution}")

# Augment data for underrepresented classes
#class_2_df = df[df['class'] == 2]

# Define the number of augmentations per sample for each class
#num_augmentations_class_2 = 3

#augmented_class_2 = class_2_df['tweet'].apply(lambda x: dp.augment_text(x, num_augmentations=num_augmentations_class_2)).explode()

# Create new DataFrame for augmented data and concatenate
#augmented_df_2 = pd.DataFrame({'tweet': augmented_class_2, 'class': 2})
#df = pd.concat([df, augmented_df_2])

# Relabel class 2 to 1
#df['class'] = df['class'].replace(2, 1)

#reset index
#df = df.sample(frac=1).reset_index(drop=True)

# # Log the number of samples per class
# class_distribution = df['class'].value_counts()
# logger.info(f"Number of samples per class after augmentation:\n{class_distribution}")

# Prepare data for neural network model
nn_tweets, tokenizer = dp.tokenize_and_pad_texts(df['tweet'].values)
nn_labels = df['class'].values  # Directly use binary labels

# Train neural network model
nn_model = models.train_neural_network(nn_tweets, nn_labels, num_classes=3)

# Save the neural network model in the SavedModel format
model_path = 'models/best_hate_model.keras'
nn_model.save(model_path)
logger.info("Saved Neural Network model")

# Save the tokenizer
tokenizer_path = 'models/hate_tokenizer.pickle'
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
logger.info("Saved tokenizer for Neural Network model")

# vectorizer = TfidfVectorizer(max_features=5000)
# X = vectorizer.fit_transform(df["tweet"])
# y = df["class"]
# class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
# class_weights = dict(zip(np.unique(y), class_weights.flatten()))
# dump(vectorizer, 'models/hate_vectorizer.joblib')
# rf_model, rf_score = models.train_random_forest(X, y, class_weights=class_weights)
# xgb_model, xgb_score = models.train_xgboost(X, y, len(np.unique(y)), class_weights=class_weights)
# svm_model, svm_score = models.train_svm(X, y, class_weights=class_weights)
# naive_model, naive_score = models.train_naive_bayes(X, y, class_weights=class_weights)
# knn_model, knn_score = models.train_knn(X, y, class_weights=class_weights)
