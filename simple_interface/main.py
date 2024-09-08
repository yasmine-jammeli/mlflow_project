import streamlit as st
import os
from preprocess import process_edf_file
from evaluation import evaluate_features
import pandas as pd
def main():
    st.title("EEG Data Processing and Evaluation")

    # Upload EDF file
    uploaded_file = st.file_uploader("Choose an EDF file", type=["edf"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        edf_file_path = os.path.join("/tmp", uploaded_file.name)
        with open(edf_file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.write("Processing the file...")

        # Process the EDF file to extract features
        features = process_edf_file(edf_file_path)

        # Save the features to a temporary CSV file
        features_csv_path = os.path.join("/tmp", "cnn_features.csv")
        pd.DataFrame(features).to_csv(features_csv_path, index=False, header=False)

        # Evaluate features
        st.write("Evaluating the features...")
        predicted_label = evaluate_features(features_csv_path)

        st.write(f"Predicted Label: {predicted_label}")

if __name__ == "__main__":
    main()
