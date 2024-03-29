import data_processing as dp
import models
from logger_config import configure_logger
from tensorflow.keras.utils import to_categorical
import pickle
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import nlpaug.augmenter.word as naw

logger = configure_logger(__name__)
logger.info("Starting the NLP pipeline for emotion detection...")

# Load and preprocess data
df = dp.import_emotion_data()
df['text'] = df['text'].apply(dp.preprocess)

# Log the number of samples per class before augmentation
class_distribution = df['label'].value_counts()
logger.info(f"Number of samples per class after preprocessing:\n{class_distribution}")

# Define the text augmentation function
#def augment_text(text, num_augmentations=1):
#    aug = naw.SynonymAug(aug_src='wordnet')
#    augmented_texts = [aug.augment(text) for _ in range(num_augmentations)]
#    return augmented_texts

# Augment data for each class
#num_augmentations = 1
#augmented_df = pd.DataFrame()

#for label in df['label'].unique():
 #   class_df = df[df['label'] == label]
  #  augmented_texts = class_df['text'].apply(lambda x: augment_text(x, num_augmentations)).explode()
   # augmented_labels = [label] * len(augmented_texts)
   # augmented_class_df = pd.DataFrame({'text': augmented_texts, 'label': augmented_labels})
   # augmented_df = pd.concat([augmented_df, augmented_class_df])

# Combine original and augmented data
#df = pd.concat([df, augmented_df])
#df = df.sample(frac=1).reset_index(drop=True)

# Log the number of samples per class after augmentation
#class_distribution = df['label'].value_counts()
#logger.info(f"Number of samples per class after augmentation:\n{class_distribution}")

# Encoding labels with LabelEncoder
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
y = df['label']

# Prepare data for neural network model
nn_texts, tokenizer = dp.tokenize_and_pad_texts(df['text'].values)
nn_labels = to_categorical(y)  # One-hot encoding for multi-class

# Train neural network model using the defined emotion neural network function
nn_model = models.train_emotion_neural_network(nn_texts, nn_labels, num_classes=len(np.unique(y)))

# Save the neural network model in the TensorFlow SavedModel format
emotion_model_path = 'models/best_emotion_model.keras'
nn_model.save(emotion_model_path)
logger.info("Saved Neural Network model")

# Save the tokenizer
emotion_tokenizer_path = 'models/emotion_tokenizer.pickle'
with open(emotion_tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
logger.info("Saved tokenizer for Neural Network model")
