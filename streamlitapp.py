import streamlit as st
import pandas as pd
import numpy as np
from mlcipherclassifier import CipherClassifier

def main():
    st.title('Cipher Classification Model')

    # Hardcoded dataset path
    dataset_path = 'encryption_dataset.csv'  # Replace with actual path to your dataset

    # Initialize classifier
    try:
        classifier = CipherClassifier(dataset_path)
        st.success('Dataset loaded successfully!')
    except Exception as e:
        st.error(f'Error loading dataset: {e}')
        return

    # Generate and display detailed report first
    st.header('Model Performance Report')
    
    # Button to generate detailed report
    if st.button('Generate Detailed Report'):
        try:
            detailed_report = classifier.generate_detailed_report()
            
            # Display report sections
            for model_name, model_report in detailed_report.items():
                st.subheader(f"{model_name} Classifier Report")
                
                # Confusion Matrix
                st.write("Confusion Matrix:")
                st.write(model_report['Confusion Matrix'])
                
                # Classification Report
                st.write("Classification Report:")
                st.write(model_report['Classification Report'])
                
                # Metrics
                st.write("Metrics Summary:")
                for metric, value in model_report['Metrics'].items():
                    st.write(f"{metric}: {value * 100:.2f}%")
                
                st.write("---")
        except Exception as e:
            st.error(f'Error generating report: {e}')

    # Prediction Section
    st.header('Cipher Algorithm Prediction')

    # Input fields
    plaintext = st.text_area('Enter Plaintext', height=100)
    ciphertext = st.text_area('Enter Ciphertext', height=100)

    # Prediction button
    if st.button('Predict Cipher Algorithm'):
        if plaintext and ciphertext:
            try:
                # Make prediction
                predictions = classifier.predict_cipher(plaintext, ciphertext)
                
                # Display predictions
                st.subheader('Prediction Results')
                for model, result in predictions.items():
                    st.write(f"{model} Prediction: {result}")
            
            except Exception as e:
                st.error(f'Error in prediction: {e}')
        else:
            st.warning('Please enter both plaintext and ciphertext')

if __name__ == '__main__':
    main()