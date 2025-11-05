# -*- coding: utf-8 -*-
"""
app.py

This script creates a Streamlit web app for invoice anomaly detection.
Run it from your terminal: streamlit run app.py
"""

# --- Cell 1: Imports and Setup ---
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import warnings
from datetime import datetime

# Ignore specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


print("--- Setting up file paths ---")
# Define paths. Assume 'invoices.csv' is in the same directory as 'app.py'
base_path = os.path.dirname(__file__) # This gets the app's directory
invoices_file_path = os.path.join(base_path, 'invoices.csv')
model_save_path = base_path
model_filename = "anomaly_model.pkl"
encoder_filename = "encoders.pkl"
columns_filename = "model_columns.pkl" # We'll also save the column order

model_filepath = os.path.join(model_save_path, model_filename)
encoder_filepath = os.path.join(model_save_path, encoder_filename)
columns_filepath = os.path.join(model_save_path, columns_filename)


# --- Cell 2: Data Loading and Preprocessing Definition ---
# (This function is identical to your original)
def preprocess_data(csv_file_path):
    """Loads, preprocesses, and returns the invoice data, label encoders, and columns used."""
    print(f"Reading CSV from: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path, low_memory=False)
        print(f"Initial DataFrame shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return None, None, None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None, None

    le_dict = {}  # Store label encoders
    categorical_cols = ['first_name', 'last_name', 'email', 'address', 'city', 'stock_code', 'job']
    numeric_cols_base = ['qty', 'amount', 'product_id'] # Base numeric columns
    df_processed = df.copy()

    print("Encoding categorical columns...")
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = df_processed[col].fillna('missing').astype(str)
            df_processed[col] = le.fit_transform(df_processed[col])
            le_dict[col] = le
        else:
            print(f"Warning: Categorical column '{col}' not found. Skipping.")

    print("Processing invoice date...")
    if 'invoice_date' in df_processed.columns:
        df_processed['invoice_date'] = pd.to_datetime(df_processed['invoice_date'], format='%d/%m/%Y', errors='coerce')
        original_rows = len(df_processed)
        df_processed.dropna(subset=['invoice_date'], inplace=True)
        rows_dropped = original_rows - len(df_processed)
        if rows_dropped > 0:
            print(f"Warning: Dropped {rows_dropped} rows due to invalid date formats in 'invoice_date'.")

        if not df_processed.empty:
            df_processed['invoice_day'] = df_processed['invoice_date'].dt.day
            df_processed['invoice_month'] = df_processed['invoice_date'].dt.month
            df_processed['invoice_year'] = df_processed['invoice_date'].dt.year
            df_processed.drop('invoice_date', axis=1, inplace=True)
            date_cols_created = ['invoice_day', 'invoice_month', 'invoice_year']
        else:
            print("Warning: DataFrame became empty after handling invalid dates.")
            return None, le_dict, None
    else:
        print("Warning: 'invoice_date' column not found. Date features not created.")
        date_cols_created = []

    final_model_cols = [col for col in numeric_cols_base if col in df_processed.columns]
    final_model_cols.extend(le_dict.keys())
    final_model_cols.extend(date_cols_created)
    final_model_cols = sorted(list(dict.fromkeys(final_model_cols)))
    final_model_cols = [col for col in final_model_cols if col in df_processed.columns]

    if not final_model_cols:
        print("Error: No valid columns identified for model training after preprocessing.")
        return None, le_dict, None

    print(f"Columns selected for model training: {final_model_cols}")
    df_final = df_processed[final_model_cols].copy()

    for col in df_final.columns:
         df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

    initial_nan_count = df_final.isnull().sum().sum()
    if initial_nan_count > 0:
        print(f"Warning: Found {initial_nan_count} NaN values after processing. Filling with 0.")
        df_final.fillna(0, inplace=True)

    print("Final DataFrame info before returning:")
    df_final.info(verbose=False)

    return df_final, le_dict, df_final.columns


# --- Cell 3: Model Training and Saving Definition (MODIFIED) ---
def train_and_save_model(df, le_dict, model_cols, save_path):
    """
    Trains the Isolation Forest model and saves it, the encoders, AND the column list.
    (This is modified to save the model_cols list)
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or None. Cannot train model.")
        return False

    # (Your data checks remain the same)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if not non_numeric_cols.empty:
        print(f"ERROR: Non-numeric columns detected before fitting: {non_numeric_cols.tolist()}.")
        return False
    if df.isnull().values.any():
        print("Warning: NaN values detected before fitting. Filling with 0.")
        df.fillna(0, inplace=True)

    model = IsolationForest(contamination=0.01, random_state=42)
    print(f"Fitting Isolation Forest model on {df.shape[0]} samples and {df.shape[1]} features...")

    try:
        model.fit(df)
        print("Model fitting complete.")
    except Exception as e:
        print(f"Error fitting the model: {e}")
        return False

    # Define full file paths
    model_out_path = os.path.join(save_path, model_filename)
    encoder_out_path = os.path.join(save_path, encoder_filename)
    columns_out_path = os.path.join(save_path, columns_filename) # New file path

    try:
        with open(model_out_path, 'wb') as f:
            pickle.dump(model, f)
        with open(encoder_out_path, 'wb') as f:
            pickle.dump(le_dict, f)
        # --- NEW: Save the column list ---
        with open(columns_out_path, 'wb') as f:
            pickle.dump(model_cols, f)

        print(f"Model successfully saved to: {model_out_path}")
        print(f"Encoders successfully saved to: {encoder_out_path}")
        print(f"Columns successfully saved to: {columns_out_path}")
        return True
    except Exception as e:
        print(f"Error saving model/encoders/columns to {save_path}: {e}")
        return False

# --- NEW: Function to run training if files are missing ---
def run_training_if_needed():
    """Checks if model files exist. If not, runs preprocessing and training."""
    if not os.path.exists(model_filepath) or not os.path.exists(encoder_filepath) or not os.path.exists(columns_filepath):
        st.warning("🤖 Model files not found. Running first-time setup... (This may take a minute)")

        if not os.path.exists(invoices_file_path):
            st.error(f"FATAL ERROR: 'invoices.csv' not found at '{invoices_file_path}'.")
            st.error("Please place 'invoices.csv' in the same directory as 'app.py' and refresh.")
            return False, "File not found"

        # Run the preprocessing
        with st.spinner("Step 1/2: Preprocessing data..."):
            df_train, le_dict, df_cols_trained = preprocess_data(invoices_file_path)

        if df_train is None:
            st.error("Preprocessing failed. Check console for errors.")
            return False, "Preprocessing failed"

        # Run the training
        with st.spinner("Step 2/2: Training and saving model..."):
            success = train_and_save_model(df_train, le_dict, df_cols_trained, model_save_path)

        if success:
            st.success("✅ First-time setup complete! Model is trained and saved.")
            st.balloons()
            return True, "Training complete"
        else:
            st.error("Model training or saving failed. Check console for errors.")
            return False, "Training failed"
    else:
        print("Model files found. Skipping training.")
        return True, "Files exist"

# --- NEW: Function to load model, encoders, and columns ---
@st.cache_resource # This caches the loaded files so we don't reload on every interaction
def load_model_and_data():
    """Loads the saved model, encoders, and column list from disk."""
    try:
        with open(model_filepath, 'rb') as f:
            model = pickle.load(f)
        with open(encoder_filepath, 'rb') as f:
            le_dict = pickle.load(f)
        with open(columns_filepath, 'rb') as f:
            df_cols_trained = pickle.load(f)

        print("Model, encoders, and columns loaded successfully.")
        return model, le_dict, df_cols_trained
    except Exception as e:
        print(f"Error loading files: {e}")
        st.error(f"Error loading model files: {e}. Try deleting the .pkl files and refreshing.")
        return None, None, None

# --- NEW: Main Streamlit App Interface ---
def main():
    st.set_page_config(page_title="Invoice Anomaly Detection", layout="wide")
    st.title("🧾 Invoice Anomaly Detection")

    # Step 1: Run training if needed
    setup_success, setup_message = run_training_if_needed()

    # If setup failed (e.g., invoices.csv not found), stop the app
    if not setup_success:
        st.stop()

    # If setup just finished, rerun the app to load the cached model
    if setup_message == "Training complete":
        st.rerun()

    # Step 2: Load the model, encoders, and columns
    model, le_dict, df_cols_trained = load_model_and_data()

    if model is None:
        st.stop() # Stop if loading failed

    # Define which columns were categorical vs. numeric for the UI
    categorical_cols = list(le_dict.keys())
    date_cols_created = ['invoice_day', 'invoice_month', 'invoice_year']

    # Infer numeric columns
    numeric_cols = [
        col for col in df_cols_trained
        if col not in categorical_cols and col not in date_cols_created
    ]

    st.header("Check a New Invoice")
    st.write("Enter the details of the invoice you want to check. The model will predict if it's an anomaly.")

    # Use a form to collect all inputs
    with st.form("invoice_form"):
        input_data = {}

        # Create columns for layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Customer Details")
            input_data['first_name'] = st.text_input("First Name")
            input_data['last_name'] = st.text_input("Last Name")
            input_data['email'] = st.text_input("Email")
            input_data['address'] = st.text_input("Address")
            input_data['city'] = st.text_input("City")
            input_data['job'] = st.text_input("Job")

        with col2:
            st.subheader("Invoice Details")
            # Use a real date input
            input_data['invoice_date'] = st.date_input("Invoice Date", value=datetime.today())
            input_data['amount'] = st.number_input("Amount ($)", min_value=0.0, format="%.2f")
            input_data['qty'] = st.number_input("Quantity", min_value=1, step=1)
            input_data['product_id'] = st.number_input("Product ID", min_value=0, step=1)
            input_data['stock_code'] = st.text_input("Stock Code")

        # Submit button for the form
        submitted = st.form_submit_button("Check for Anomaly")

    # --- Step 3: Process and Predict on Submission ---
    if submitted:
        with st.spinner("Analyzing invoice..."):
            # 1. Create the input DataFrame, initialized to 0.0
            df_input = pd.DataFrame(0.0, index=[0], columns=df_cols_trained)

            # 2. Process and fill the DataFrame
            try:
                for col_name, value in input_data.items():
                    if not value: # Skip if user left it blank
                        continue

                    if col_name == 'invoice_date':
                        # Special handling for date
                        if 'invoice_day' in df_input.columns:
                            df_input.loc[0, 'invoice_day'] = float(value.day)
                        if 'invoice_month' in df_input.columns:
                            df_input.loc[0, 'invoice_month'] = float(value.month)
                        if 'invoice_year' in df_input.columns:
                            df_input.loc[0, 'invoice_year'] = float(value.year)

                    elif col_name in le_dict:
                        # Handle categorical
                        le = le_dict[col_name]
                        if value in le.classes_:
                            df_input.loc[0, col_name] = float(le.transform([value])[0])
                        else:
                            # Unseen label. Treat as 'missing' if 'missing' was trained
                            if 'missing' in le.classes_:
                                df_input.loc[0, col_name] = float(le.transform(['missing'])[0])
                            else:
                                df_input.loc[0, col_name] = 0.0 # Default to 0

                    elif col_name in numeric_cols:
                        # Handle numeric
                        df_input.loc[0, col_name] = float(value)

                # 3. Fill any remaining NaNs (should be 0.0, but just in case)
                df_input.fillna(0.0, inplace=True)

                # 4. Make prediction
                prediction = model.predict(df_input[df_cols_trained]) # Ensure column order

                # 5. Show result
                if prediction[0] == -1:
                    st.error("🚨 Anomaly Detected! This invoice is unusual.")
                else:
                    st.success("✅ No Anomaly Detected. This invoice looks normal.")

                # (Optional) Show the processed data
                st.subheader("Processed Input Data (for debugging):")
                st.dataframe(df_input)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")


# --- Standard Python entry point ---
if __name__ == "__main__":
    main()
