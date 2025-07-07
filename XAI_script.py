# XAI_script.py
#
# Project: The Ultimate Interactive AI Strategy Lab: Fraud Detection
# Author: Archi-Dev Harsh, Bhushit, Aakash, Raj Mishra & Sexy Bhai 😘 (and Apex!)
# Version: 2.6 - Golden Master, All Fixes Integrated
#
# Description:
# This script is responsible for generating Explainable AI (XAI) artifacts,
# specifically SHAP values, for the best-performing model identified from
# `experiment_results.csv`. It loads the necessary data, re-trains the
# chosen model, calculates SHAP explanations for its predictions on the
# test set, and then saves these artifacts as `.pkl` files.
#
# These `.pkl` files are then consumed by the `app.py` Streamlit application's
# "XAI Deep-Dive" tab to provide interactive model explanations.
#
# Key Features:
# - Identifies the best model (typically Random Forest with SMOTE) from pre-computed results.
# - Retrains the identified model to ensure compatibility with SHAP.
# - Utilizes the SHAP library to compute global and local explanations.
# - Persists SHAP values, test data, base value, and feature names for `app.py`.
#
# Dependencies:
# - pandas, numpy
# - scikit-learn (for train_test_split, StandardScaler, RandomForestClassifier)
# - imbalanced-learn (for SMOTE)
# - shap
# - ast (for safely parsing hyperparameter strings from CSV)
#
# Usage:
# Run this script directly from your terminal AFTER `run_experiments.py` has
# successfully generated `artifacts/experiment_results.csv` and `data/creditcard.csv` is present.
# Command: `python XAI_script.py`
# ======================================================================================

import pandas as pd
import numpy as np
import pickle
import ast # For safely evaluating string representations of dictionaries
import os # For checking file existence

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # We expect RF to be our best model for XAI
from sklearn.metrics import f1_score # Just for verification during best model identification

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE

# XAI Library
import shap

print("--- XAI Data Generation Started (Robust Mode) ---")

# --- Configuration (FILE PATHS UPDATED HERE) ---
DATA_PATH = 'data/creditcard.csv' # Points to the data folder
RESULTS_FILE = 'artifacts/experiment_results.csv' # Points to the artifacts folder
# Using the same sample fraction as run_experiments.py to ensure data consistency
SAMPLE_SIZE_FRACTION = 0.035 # Adjust this if you used a different fraction in run_experiments.py
TEST_SIZE = 0.3 # Test size for train_test_split
RANDOM_STATE = 42

# --- Step 1 & 2: Load and Prepare Data ---
print(f"1&2. Loading and preparing data from '{DATA_PATH}' and '{RESULTS_FILE}'...")
try:
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"'{RESULTS_FILE}' not found. Please run `run_experiments.py` first to generate it in the 'artifacts/' directory.")
    results_df = pd.read_csv(RESULTS_FILE)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"'{DATA_PATH}' not found. Please ensure it's in the 'data/' directory.")
    df_full = pd.read_csv(DATA_PATH)
except FileNotFoundError as e:
    print(f"❌ ERROR: {e}. Exiting.")
    exit()
except Exception as e:
    print(f"❌ ERROR: An unexpected error occurred during data loading: {e}. Exiting.")
    exit()

X = df_full.drop('Class', axis=1)
y = df_full['Class']

# Take a stratified sample for XAI generation, consistent with `run_experiments.py`
if SAMPLE_SIZE_FRACTION < 1.0:
    _, X_sample, _, y_sample = train_test_split(
        X, y, test_size=SAMPLE_SIZE_FRACTION, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   ...Using a stratified sample of {len(X_sample)} rows ({SAMPLE_SIZE_FRACTION:.1%}).")
else:
    X_sample, y_sample = X, y
    print(f"   ...Using full dataset for XAI (rows: {len(X_sample)}).")

# Split into training and testing sets for the best model retraining
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_sample
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ...Data prepared and scaled.")


# --- Step 3: Identify and Re-train the Best Model ---
print("3. Identifying and re-training the best model (Random Forest + SMOTE)...")
try:
    # Filter for models that used SMOTE and sort by F1-score to find the best
    smote_results = results_df[results_df['sampling_strategy'] == 'SMOTE']
    
    # We specifically target Random Forest as it's typically a strong performer
    # and TreeExplainer in SHAP works very efficiently with tree-based models.
    best_rf_smote_row = smote_results[smote_results['model_name'] == 'Random Forest'].sort_values('f1_score', ascending=False).iloc[0]
    
    best_model_name = best_rf_smote_row['model_name']
    best_f1_score = best_rf_smote_row['f1_score']
    # Use ast.literal_eval for safe parsing of the hyperparameters string
    best_params = ast.literal_eval(best_rf_smote_row['hyperparameters'])
    
    print(f"   ...Best RF+SMOTE model found: '{best_model_name}' with F1-Score of {best_f1_score:.4f}")
    print(f"      Hyperparameters: {best_params}")

except (IndexError, KeyError) as e:
    print(f"❌ ERROR: Could not find 'Random Forest' model trained with 'SMOTE' in '{RESULTS_FILE}' or invalid format: {e}. "
          "Please ensure `run_experiments.py` was run successfully and generated relevant results. Exiting.")
    exit()
except Exception as e:
    print(f"❌ ERROR: An unexpected error occurred while identifying the best model: {e}. Exiting.")
    exit()

# Apply SMOTE to the training data for retraining the best model
smote_sampler = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote_sampler.fit_resample(X_train_scaled, y_train)
print(f"   ...Training data resampled with SMOTE. New shape: {X_train_smote.shape}")

# Re-initialize and train the best model
best_model = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    random_state=RANDOM_STATE,
    n_jobs=-1 # Use all available CPU cores for faster training
)

print(f"   ...Retraining {best_model_name}...")
best_model.fit(X_train_smote, y_train_smote)
print("   ...Best model re-trained successfully.")


# --- Step 4: Generate SHAP Artifacts ---
print("4. Calculating SHAP values for explanations...")

try:
    # Explicitly use TreeExplainer for RandomForestClassifier.
    # This is generally recommended for tree-based models and is efficient.
    explainer = shap.TreeExplainer(best_model, X_train_smote)

    # Calculate SHAP values for our test set.
    # The `explainer()` call returns an Explanation object.
    shap_object = explainer(X_test_scaled)

    # For binary classification with RandomForestClassifier's predict_proba,
    # `shap_object.values` will have a shape of (num_samples, num_features, num_outputs).
    # We want the SHAP values for the positive class (Class 1, 'Fraud'), which is at index 1.
    shap_values_for_fraud_class = shap_object.values[:, :, 1] # Correct slicing for (num_samples, num_features)

    # Sanity Check for the shape of SHAP values
    expected_shap_shape = X_test_scaled.shape # SHAP values should have the same dimensions as input features
    if shap_values_for_fraud_class.shape != expected_shap_shape:
        print(f"   ❌ FATAL SHAP Shape Mismatch! Expected SHAP values shape {expected_shap_shape}, but got {shap_values_for_fraud_class.shape}.")
        print("      This often indicates an issue with SHAP's interpretation of model output or data.")
        exit() # Exit if shape is incorrect
    else:
        print(f"   ...SHAP values shape: {shap_values_for_fraud_class.shape} (Correct)")


    # Get the base value (expected value) for the "Fraud" class.
    # For TreeExplainer, `explainer.expected_value` gives a (num_outputs,) array.
    # So `explainer.expected_value[1]` will correctly give the scalar base value for the positive class.
    base_value_for_fraud_class = explainer.expected_value[1] 

    # --- Important: Add a safety check before printing to prevent the error ---
    # Ensure base_value_for_fraud_class is a scalar (not an array) before formatting it.
    if isinstance(base_value_for_fraud_class, np.ndarray) and base_value_for_fraud_class.ndim > 0:
        print(f"   ❌ Error: Base value is unexpectedly an array ({base_value_for_fraud_class.shape}). Expected a scalar.")
        print(f"      Base value content: {base_value_for_fraud_class}")
        exit() # Exit if base value is not scalar

    print(f"   ...Base value for 'Fraud' class: {base_value_for_fraud_class:.4f}") # Now this print should be safe.

except Exception as e:
    print(f"❌ ERROR: An error occurred during SHAP value calculation: {e}. Exiting.")
    exit()

# --- Step 5: Save all SHAP artifacts (FILE PATHS UPDATED HERE) ---
print("5. Saving SHAP artifacts to .pkl files...")
try:
    # Ensure artifacts directory exists before saving results
    os.makedirs('artifacts', exist_ok=True) # Create 'artifacts' folder if it doesn't exist
    
    # Define file paths for output
    shap_values_file = 'artifacts/shap_values.pkl'
    x_test_scaled_file = 'artifacts/X_test_scaled.pkl'
    shap_base_value_file = 'artifacts/shap_base_value.pkl'
    feature_names_file = 'artifacts/feature_names.pkl'

    # Save SHAP values (for the positive class)
    with open(shap_values_file, 'wb') as f:
        pickle.dump(shap_values_for_fraud_class, f)
    print(f"   - Saved SHAP values to '{shap_values_file}'")

    # Save the scaled test data (needed for plotting)
    with open(x_test_scaled_file, 'wb') as f:
        pickle.dump(X_test_scaled, f)
    print(f"   - Saved X_test_scaled to '{x_test_scaled_file}'")

    # Save the base value for the positive class (needed for force plots)
    with open(shap_base_value_file, 'wb') as f:
        pickle.dump(base_value_for_fraud_class, f)
    print(f"   - Saved SHAP base value to '{shap_base_value_file}'")
    
    # Save feature names (original column names from X_sample)
    feature_names = X_sample.columns.tolist()
    with open(feature_names_file, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"   - Saved feature names to '{feature_names_file}'")

except Exception as e:
    print(f"❌ ERROR: An error occurred while saving SHAP artifacts: {e}. Exiting.")
    exit()

print("\n✅ All Done! Your XAI artifacts have been correctly generated.")
print("You can now run `streamlit run app.py` and check the '🧠 XAI Deep-Dive' tab.")
print("---------------------------------------------------------------------")