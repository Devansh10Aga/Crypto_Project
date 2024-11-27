Cipher Classification System

Overview
This project implements a system for encrypting text data, creating a dataset, training machine
learning models to classify encryption algorithms, and deploying a Streamlit-based user
interface for model interaction.
Features:

1. Data Encryption (cry.py):
  ○ Implements two encryption methods: Complete Columnar Transposition and
    Compressocrat.
  ○ Generates a dataset of plaintext, ciphertext, and corresponding algorithms.
  ○ Saves the dataset to a CSV file.
2. Machine Learning Models (mlcipherclassifier.py):
  ○ Preprocesses data using TF-IDF vectorization and tokenization.
  ○ Trains three classifiers:
   ■ Naive Bayes (TF-IDF features).
   ■ Neural Network (TF-IDF features).
   ■ LSTM (sequence features).
  ○ Provides a detailed performance report and prediction functionality.
3. Interactive Streamlit App (streamlitapp.py):
  ○ Visualizes model performance metrics and confusion matrices.
  ○ Allows users to input plaintext and ciphertext for algorithm prediction.
  ○ Displays predictions from all trained models.
Files

1. cry.py:-
● Purpose: Generates a dataset with encrypted texts and their corresponding algorithms.
● Key Functions:
  ○ generate_random_string(length): Creates random strings for encryption.
  ○ complete_columnar_transposition(text, period): Encrypts text using
a columnar transposition cipher.
○ compressocrat(text): Implements a substitution cipher.
○ generate_encryption_dataset(num_entries): Creates a dataset of
encrypted text.
○ save_dataset_to_csv(dataset): Saves the dataset to a CSV file.
2. mlcipherclassifier.py:- 
● Purpose: Trains models to classify encryption algorithms.
● Key Components:
  ○ Data preprocessing using TF-IDF and sequence tokenization.
  ○ Models trained:
    ■ Naive Bayes (MultinomialNB).
    ■ Neural Network (MLPClassifier).
    ■ LSTM (TensorFlow/Keras).
○ generate_detailed_report(): Outputs performance metrics for all models.
○ predict_cipher(plaintext, ciphertext): Predicts encryption algorithm
based on input text.
3. streamlitapp.py
● Purpose: Provides a user-friendly interface for interacting with the classifiers.
● Features:
  ○ Loads the dataset and initializes the classifier.
  ○ Displays detailed performance reports.
  ○ Enables real-time predictions for user-provided plaintext and ciphertext.

Usage:-
1. Generate a dataset by running cry.py.
2. Train the models by initializing CipherClassifier from mlcipherclassifier.py.
3. Launch the Streamlit app to view reports or make predictions.
   
Future Enhancements:-
● Add support for additional encryption algorithms.
● Incorporate real-time dataset generation in the Streamlit app.
● Improve model accuracy with hyperparameter tuning and advanced architectures
