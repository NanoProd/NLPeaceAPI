import numpy as np
import tensorflow as tf
# import xgboost as xgb
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Embedding, GlobalAveragePooling1D, Input, MaxPooling1D
from tensorflow.keras.models import Sequential
# from tensorflow.keras.utils import to_categorical
# import tensorflow_addons as tfa
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Import logger
from logger_config import configure_logger
logger = configure_logger(__name__)


# Import logger
from logger_config import configure_logger
logger = configure_logger(__name__)

# #def train_random_forest(X, y, class_weights=None):
# #    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights)
# #    kf = KFold(n_splits=5)
# #    best_f1_score = 0
# #    best_model_rf = None
# #    for train_index, test_index in kf.split(X):
# #        X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#         f1 = f1_score(y_test, y_pred, average='weighted')
#         logger.info(f"RandomForest F1 Score: {f1}")
#         if f1 > best_f1_score:
#             best_f1_score = f1
#             best_model_rf = clf
#     if best_model_rf:
#         return best_model_rf, best_f1_score

# def train_xgboost(X, y, num_classes, class_weights=None):
#     clf_xgb = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes, class_weight=class_weights)
#     kf = KFold(n_splits=5)
#     best_f1_score = 0
#     best_model = None
#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         clf_xgb.fit(X_train, y_train)
#         y_pred = clf_xgb.predict(X_test)
#         f1 = f1_score(y_test, y_pred, average='weighted')
#         logger.info(f"XGBoost F1 Score: {f1}")
#         if f1 > best_f1_score:
#             best_f1_score = f1
#             best_model = clf_xgb
#     if best_model:
#         return best_model, best_f1_score

# def train_svm(X, y, class_weights=None):
#     clf_svm = SVC(kernel='linear', class_weight=class_weights)
#     kf = KFold(n_splits=5)
#     best_f1_score = 0
#     best_model_svm = None
#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         clf_svm.fit(X_train, y_train)
#         y_pred = clf_svm.predict(X_test)
#         f1 = f1_score(y_test, y_pred, average='weighted')
#         logger.info(f"SVM F1 Score: {f1}")
#         if f1 > best_f1_score:
#             best_f1_score = f1
#             best_model_svm = clf_svm
#     if best_model_svm:
#         return best_model_svm, best_f1_score

# def train_naive_bayes(X, y, class_weights=None):
#     if class_weights is None:
#         clf_nb = MultinomialNB()
#     else:
#         class_prior = np.array([class_weights[i] for i in sorted(class_weights.keys())])
#         clf_nb = MultinomialNB(class_prior=class_prior)

#     kf = KFold(n_splits=5)
#     best_f1_score = 0
#     best_model_nb = None
#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         clf_nb.fit(X_train, y_train)
#         y_pred = clf_nb.predict(X_test)
#         f1 = f1_score(y_test, y_pred, average='weighted')
#         logger.info(f"Naive Bayes F1 Score: {f1}")
#         if f1 > best_f1_score:
#             best_f1_score = f1
#             best_model_nb = clf_nb
#     if best_model_nb:
#         return best_model_nb, best_f1_score
    
# def train_knn(X, y, class_weights=None):
#     clf_knn = KNeighborsClassifier(weights='distance')
#     kf = KFold(n_splits=5)
#     best_f1_score = 0
#     best_model_knn = None
#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         clf_knn.fit(X_train, y_train)
#         y_pred = clf_knn.predict(X_test)
#         f1 = f1_score(y_test, y_pred, average='weighted')
#         logger.info(f"KNN F1 Score: {f1}")
#         if f1 > best_f1_score:
#             best_f1_score = f1
#             best_model_knn = clf_knn
#     if best_model_knn:
#         return best_model_knn, best_f1_score

# Constants
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 280
EMBEDDING_DIM = 16  # Embedding dimensions

def train_neural_network(X, y, num_classes):
    # One-hot encode the labels
    y_one_hot = to_categorical(y, num_classes)

    # Splitting data for training and validation
    X_train, X_val, y_train_one_hot, y_val_one_hot = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    # Define the model
    model = Sequential([
        Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM, input_shape=(MAX_SEQUENCE_LENGTH,)),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(5),
        Conv1D(128, 5, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy', 
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.CategoricalAccuracy(),
                           tf.keras.metrics.AUC(name='auc'),
                           tf.keras.metrics.AUC(name='prc', curve='PR')])

    # Add EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train_one_hot, epochs=20, validation_data=(X_val, y_val_one_hot), callbacks=[early_stopping])

    return model

EMOTION_MAX_VOCAB_SIZE = 10000
EMOTION_MAX_SEQUENCE_LENGTH = 280
EMOTION_EMBEDDING_DIM = 20

def train_emotion_neural_network(X, y, num_classes):
    model = Sequential([
        Embedding(EMOTION_MAX_VOCAB_SIZE, EMOTION_EMBEDDING_DIM, input_shape=(MAX_SEQUENCE_LENGTH,)),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(5),
        Conv1D(128, 5, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model for multi-class classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
      tf.keras.metrics.MeanSquaredError(name='Brier score'),
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.Accuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR')])

    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

    return model