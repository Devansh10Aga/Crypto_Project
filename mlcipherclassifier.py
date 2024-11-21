import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Concatenate, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class CipherClassifier:
    def __init__(self, dataset_path):
        # Load and preprocess data
        self.df = pd.read_csv(dataset_path)

        # Encode algorithms
        self.label_encoder = LabelEncoder()
        self.df['Algorithm_Encoded'] = self.label_encoder.fit_transform(self.df['Algorithm'])

        # TF-IDF Vectorization
        self.plaintext_vectorizer = TfidfVectorizer(max_features=5000)
        self.ciphertext_vectorizer = TfidfVectorizer(max_features=5000)

        self.X_plaintext = self.plaintext_vectorizer.fit_transform(self.df['Plaintext']).toarray()
        self.X_ciphertext = self.ciphertext_vectorizer.fit_transform(self.df['Ciphertext']).toarray()

        # Combine features
        self.X = np.concatenate([self.X_plaintext, self.X_ciphertext], axis=1)
        self.y = self.df['Algorithm_Encoded']

        # Train-Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # LSTM Tokenization
        self.plaintext_tokenizer = Tokenizer(num_words=5000)
        self.plaintext_tokenizer.fit_on_texts(self.df['Plaintext'])
        self.plaintext_seq = self.plaintext_tokenizer.texts_to_sequences(self.df['Plaintext'])
        self.plaintext_pad = pad_sequences(self.plaintext_seq, maxlen=100)

        self.ciphertext_tokenizer = Tokenizer(num_words=5000)
        self.ciphertext_tokenizer.fit_on_texts(self.df['Ciphertext'])
        self.ciphertext_seq = self.ciphertext_tokenizer.texts_to_sequences(self.df['Ciphertext'])
        self.ciphertext_pad = pad_sequences(self.ciphertext_seq, maxlen=100)

        # Unique algorithms for reporting
        self.algorithms = self.label_encoder.classes_

        # Train models during initialization
        self.nb_classifier = self._train_naive_bayes()
        self.nn_classifier = self._train_neural_network()
        self.lstm_model = self._train_lstm_model()

    def _train_naive_bayes(self):
        nb_classifier = MultinomialNB()
        nb_classifier.fit(self.X_train, self.y_train)
        return nb_classifier

    def _train_neural_network(self):
        nn_classifier = MLPClassifier(max_iter=500)
        nn_classifier.fit(self.X_train, self.y_train)
        return nn_classifier

    def _train_lstm_model(self):
        # Prepare training data
        X_train_plaintext, X_test_plaintext, _, _ = train_test_split(
            self.plaintext_pad, self.y, test_size=0.2, random_state=42
        )
        X_train_ciphertext, X_test_ciphertext, _, _ = train_test_split(
            self.ciphertext_pad, self.y, test_size=0.2, random_state=42
        )

        # Complex LSTM model with both plaintext and ciphertext inputs
        plaintext_input = Input(shape=(100,))
        ciphertext_input = Input(shape=(100,))

        plaintext_embedding = Embedding(5000, 32, input_length=100)(plaintext_input)
        ciphertext_embedding = Embedding(5000, 32, input_length=100)(ciphertext_input)

        plaintext_lstm = LSTM(64)(plaintext_embedding)
        ciphertext_lstm = LSTM(64)(ciphertext_embedding)

        merged = Concatenate()([plaintext_lstm, ciphertext_lstm])

        dense_layer = Dense(len(np.unique(self.y)), activation='softmax')(merged)

        model = Model(inputs=[plaintext_input, ciphertext_input], outputs=dense_layer)

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(
            [X_train_plaintext, X_train_ciphertext],
            self.y_train,
            epochs=10,
            verbose=0
        )

        return model

    def generate_detailed_report(self):
        # Prepare report dictionary
        detailed_report = {}

        # 1. Naive Bayes Detailed Report
        nb_pred = self.nb_classifier.predict(self.X_test)

        detailed_report['Naive Bayes'] = {
            'Confusion Matrix': confusion_matrix(self.y_test, nb_pred),
            'Classification Report': classification_report(
                self.y_test, nb_pred,
                target_names=self.algorithms
            ),
            'Metrics': {
                'Accuracy': accuracy_score(self.y_test, nb_pred),
                'Precision (Weighted)': precision_score(self.y_test, nb_pred, average='weighted'),
                'Recall (Weighted)': recall_score(self.y_test, nb_pred, average='weighted'),
                'F1 Score (Weighted)': f1_score(self.y_test, nb_pred, average='weighted')
            }
        }

        # 2. Neural Network Detailed Report
        nn_pred = self.nn_classifier.predict(self.X_test)

        detailed_report['Neural Network'] = {
            'Confusion Matrix': confusion_matrix(self.y_test, nn_pred),
            'Classification Report': classification_report(
                self.y_test, nn_pred,
                target_names=self.algorithms
            ),
            'Metrics': {
                'Accuracy': accuracy_score(self.y_test, nn_pred),
                'Precision (Weighted)': precision_score(self.y_test, nn_pred, average='weighted'),
                'Recall (Weighted)': recall_score(self.y_test, nn_pred, average='weighted'),
                'F1 Score (Weighted)': f1_score(self.y_test, nn_pred, average='weighted')
            }
        }

        # 3. LSTM Detailed Report
        lstm_pred = self.lstm_model.predict([
            self.plaintext_pad[self.y_test.index],
            self.ciphertext_pad[self.y_test.index]
        ]).argmax(axis=1)

        detailed_report['LSTM'] = {
            'Confusion Matrix': confusion_matrix(self.y_test, lstm_pred),
            'Classification Report': classification_report(
                self.y_test, lstm_pred,
                target_names=self.algorithms
            ),
            'Metrics': {
                'Accuracy': accuracy_score(self.y_test, lstm_pred),
                'Precision (Weighted)': precision_score(self.y_test, lstm_pred, average='weighted'),
                'Recall (Weighted)': recall_score(self.y_test, lstm_pred, average='weighted'),
                'F1 Score (Weighted)': f1_score(self.y_test, lstm_pred, average='weighted')
            }
        }

        # Print detailed report
        self._print_detailed_report(detailed_report)

        return detailed_report

    def _print_detailed_report(self, report):
        for model_name, model_report in report.items():
            print(f"\n{'='*50}")
            print(f"{model_name} Classifier Report")
            print(f"{'='*50}")

            print("\nConfusion Matrix:")
            print(model_report['Confusion Matrix'])

            print("\nClassification Report:")
            print(model_report['Classification Report'])

            print("\nMetrics Summary:")
            for metric, value in model_report['Metrics'].items():
                print(f"{metric}: {value * 100:.2f}%")

    def predict_cipher(self, plaintext, ciphertext):
        # 1. Naive Bayes Prediction
        combined_features = np.concatenate([
            self.plaintext_vectorizer.transform([plaintext]).toarray(),
            self.ciphertext_vectorizer.transform([ciphertext]).toarray()
        ], axis=1)

        nb_pred = self.nb_classifier.predict(combined_features)
        nb_result = self.label_encoder.inverse_transform(nb_pred)[0]

        # 2. Neural Network Prediction
        nn_pred = self.nn_classifier.predict(combined_features)
        nn_result = self.label_encoder.inverse_transform(nn_pred)[0]

        # 3. LSTM Prediction
        plaintext_seq = self.plaintext_tokenizer.texts_to_sequences([plaintext])
        plaintext_pad = pad_sequences(plaintext_seq, maxlen=100)

        ciphertext_seq = self.ciphertext_tokenizer.texts_to_sequences([ciphertext])
        ciphertext_pad = pad_sequences(ciphertext_seq, maxlen=100)

        lstm_pred = self.lstm_model.predict([
            plaintext_pad,
            ciphertext_pad
        ]).argmax(axis=1)
        lstm_result = self.label_encoder.inverse_transform(lstm_pred)[0]

        # Print predictions
        print("\nPrediction Results:")
        print(f"Naive Bayes Prediction: {nb_result}")
        print(f"Neural Network Prediction: {nn_result}")
        print(f"LSTM Prediction: {lstm_result}")

        return {
            'Naive Bayes': nb_result,
            'Neural Network': nn_result,
            'LSTM': lstm_result
        }