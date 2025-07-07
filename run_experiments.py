# run_experiments.py
#
# Project: The Ultimate Interactive AI Strategy Lab: Fraud Detection
# Author: Archi-Dev Harsh, Bhushit, Aakash, Raj Mishra & Sexy Bhai 😘
# Version: 2.6 - Golden Master, All Fixes Integrated
#
# Description:
# This script is the "Research Rig" of the Fraud Detection Strategy Lab.
# It systematically runs a predefined set of machine learning experiments,
# testing various models, data sampling techniques (SMOTE, Undersampling),
# and hyperparameter combinations on a credit card fraud dataset.
#
# The primary output of this script is `experiment_results.csv`, which contains
# the performance metrics (Accuracy, Precision, Recall, F1-Score) for each
# experiment. This CSV file is then consumed by `app.py` for interactive analysis.
#
# Key Features:
# - Automates the training and evaluation of multiple ML models.
# - Integrates data sampling strategies crucial for imbalanced datasets.
# - Manages hyperparameter tuning grids for different algorithms.
# - Saves all results to a structured CSV for later analysis.
# - Utilizes custom models from `models.py` alongside scikit-learn models.
#
# Dependencies:
# - pandas, numpy
# - scikit-learn (for models, splitting, scaling, metrics)
# - imbalanced-learn (for SMOTE, RandomUnderSampler)
# - tensorflow (for Neural Network from models.py)
# - models (local custom module for KNNFromScratch, build_keras_model)
#
# Usage:
# Run this script directly from your terminal: `python run_experiments.py`
# Ensure `creditcard.csv` is in the 'data/' directory and `models.py`
# is in the project root.
# ======================================================================================

import pandas as pd
import numpy as np
import time
import os
import ast # To parse hyperparameter strings back if needed, though we store as string

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Import custom models from our local `models.py` file
try:
    from models import KNNFromScratch, build_keras_model
    MODELS_LOADED = True
except ImportError:
    MODELS_LOADED = False
    print("⚠️ WARNING: Could not import custom models from `models.py`. "
          "KNN (From Scratch) and Neural Network will not be included in experiments.")
except Exception as e:
    MODELS_LOADED = False
    print(f"❌ ERROR: An unexpected error occurred while importing `models.py`: {e}. "
          "Custom models will be skipped.")

# --- Configuration (FILE PATHS UPDATED HERE) ---
DATA_PATH = 'data/creditcard.csv' # Points to the data folder
RESULTS_FILE = 'artifacts/experiment_results.csv' # Points to the artifacts folder
SAMPLE_SIZE_FRACTION = 0.035 # Using a fraction for testing, full data for final run
TEST_SIZE = 0.3
RANDOM_STATE = 42

# --- Model Definitions and Hyperparameter Grids ---
# Define the models and their respective hyperparameters to test.
# For custom models, ensure they are compatible with sklearn's fit/predict interface.

MODELS_CONFIG = {
    "Logistic Regression": {
        "model": LogisticRegression,
        "params_grid": [
            {"solver": "liblinear", "max_iter": 1000, "random_state": RANDOM_STATE},
            # Add more LR param combinations if needed
        ]
    },
    "Random Forest": {
        "model": RandomForestClassifier,
        "params_grid": [
            {"n_estimators": 50, "max_depth": 5, "random_state": RANDOM_STATE, "n_jobs": -1},
            {"n_estimators": 100, "max_depth": 10, "random_state": RANDOM_STATE, "n_jobs": -1},
            {"n_estimators": 150, "max_depth": 15, "random_state": RANDOM_STATE, "n_jobs": -1},
            # You can add more combinations here
        ]
    }
}

if MODELS_LOADED:
    MODELS_CONFIG["KNN (From Scratch)"] = {
        "model": KNNFromScratch,
        "params_grid": [
            {"k": 3, "distance_metric": "Euclidean"},
            {"k": 5, "distance_metric": "Euclidean"},
            {"k": 7, "distance_metric": "Euclidean"},
            {"k": 5, "distance_metric": "Manhattan"},
            {"k": 7, "distance_metric": "Manhattan"},
        ]
    }
    MODELS_CONFIG["Neural Network"] = {
        # Note: build_keras_model is a function, not a class, so handle accordingly
        "model_builder": build_keras_model,
        "params_grid": [
            {"layers": 1, "neurons": 32, "learning_rate": 0.001, "epochs": 20},
            {"layers": 2, "neurons": 64, "learning_rate": 0.001, "epochs": 20},
            {"layers": 2, "neurons": 32, "learning_rate": 0.0005, "epochs": 30},
        ]
    }

SAMPLING_STRATEGIES = {
    "None": None,
    "SMOTE": SMOTE(random_state=RANDOM_STATE),
    "Undersampling": RandomUnderSampler(random_state=RANDOM_STATE)
}


# --- Helper Functions ---

def load_data(file_path: str, sample_fraction: float):
    """
    Loads the credit card fraud dataset and takes a stratified sample.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: Data file '{file_path}' not found. Please ensure it's in the '{os.path.dirname(file_path)}' directory.")
        return None, None
    
    df = pd.read_csv(file_path)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Take a stratified sample to manage computation time, especially for custom models
    if sample_fraction < 1.0:
        _, X_sample, _, y_sample = train_test_split(
            X, y, test_size=sample_fraction, random_state=RANDOM_STATE, stratify=y
        )
        print(f"Loaded full data (rows: {len(df)}). Using a stratified sample of {len(X_sample)} rows ({sample_fraction:.1%}).")
    else:
        X_sample, y_sample = X, y
        print(f"Loaded full data (rows: {len(df)}). Using all data for experiments.")
        
    return X_sample, y_sample

def get_model_instance(model_name: str, params: dict, input_shape: int):
    """Initializes and returns a model instance based on its name and parameters."""
    config = MODELS_CONFIG.get(model_name)
    if not config:
        return None

    if model_name == "Neural Network":
        # For Neural Network, we call the builder function
        return config["model_builder"](input_shape=input_shape, 
                                       layers=params["layers"], 
                                       neurons=params["neurons"], 
                                       learning_rate=params["learning_rate"])
    else:
        # For other models (classes), instantiate directly
        return config["model"](**params)


def run_single_experiment(
    X: pd.DataFrame, 
    y: pd.Series, 
    model_name: str, 
    sampling_strategy_name: str, 
    model_params: dict
) -> dict:
    """
    Runs a single machine learning experiment with specified model, sampling, and hyperparameters.
    """
    start_time = time.time()
    
    # 1. Data Splitting (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 2. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Apply Sampling Strategy
    X_train_resampled, y_train_resampled = X_train_scaled, y_train
    if SAMPLING_STRATEGIES[sampling_strategy_name] is not None:
        sampler = SAMPLING_STRATEGIES[sampling_strategy_name]
        try:
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
            # print(f"   Applied {sampling_strategy_name}: Resampled train data shape {X_train_resampled.shape}")
        except Exception as e:
            print(f"   ❌ Error applying {sampling_strategy_name}: {e}. Skipping this combination.")
            return {} # Return empty dict if sampling fails
    
    # 4. Model Initialization
    model = get_model_instance(model_name, model_params, X_train_resampled.shape[1])
    
    if model is None:
        print(f"   ❌ Failed to initialize model: {model_name}. Skipping.")
        return {} # Return empty dict if model initialization fails

    # 5. Model Training
    try:
        if model_name == "Neural Network":
            # Keras fit method needs epochs and batch_size (can be added to model_params)
            epochs = model_params.get("epochs", 10) # Default epochs if not in params
            model.fit(X_train_resampled, y_train_resampled, epochs=epochs, batch_size=32, verbose=0)
            y_pred_proba = model.predict(X_test_scaled).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_test_scaled)
    except Exception as e:
        print(f"   ❌ Error during training/prediction for {model_name} with {sampling_strategy_name}: {e}. Skipping.")
        return {} # Return empty dict if training fails

    # 6. Evaluation Metrics
    # zero_division=0 avoids warnings/errors when a class has no predicted samples
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    end_time = time.time()
    run_time = end_time - start_time

    results = {
        "model_name": model_name,
        "sampling_strategy": sampling_strategy_name,
        "hyperparameters": str(model_params), # Store params as string to save in CSV
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "runtime_seconds": run_time
    }
    return results


# --- Main Execution Block ---
if __name__ == "__main__":
    print("\n🚀 Starting the Ultimate AI Strategy Lab: Fraud Detection - Research Rig!")
    # Ensure artifacts directory exists before saving results
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True) # Create 'artifacts' folder if it doesn't exist
    
    print(f"Loading data from: {DATA_PATH}")
    print(f"Saving results to: {RESULTS_FILE}")
    print(f"Using test size: {TEST_SIZE * 100:.0f}%")
    
    X_data, y_data = load_data(DATA_PATH, SAMPLE_SIZE_FRACTION)

    if X_data is None or y_data is None:
        print("🔴 Failed to load data. Exiting experiment runner.")
        exit()

    all_experiment_results = []
    total_experiments = 0
    for model_name, model_config in MODELS_CONFIG.items():
        if "params_grid" in model_config:
            total_experiments += len(model_config["params_grid"]) * len(SAMPLING_STRATEGIES)
        elif "model_builder" in model_config: # For Neural Network
            total_experiments += len(model_config["params_grid"]) * len(SAMPLING_STRATEGIES)

    print(f"\n✨ Total experiments to run: {total_experiments}")
    current_experiment_num = 0

    for model_name, model_config in MODELS_CONFIG.items():
        model_params_grid = model_config.get("params_grid", [{}]) # Default to empty dict if no specific grid
        
        for params in model_params_grid:
            for sampling_name in SAMPLING_STRATEGIES.keys():
                current_experiment_num += 1
                print(f"\n--- Running Experiment {current_experiment_num}/{total_experiments} ---")
                print(f"   Model: {model_name}, Sampling: {sampling_name}, Params: {params}")
                
                result = run_single_experiment(X_data, y_data, model_name, sampling_name, params)
                
                if result: # Only add if experiment ran successfully and returned results
                    all_experiment_results.append(result)
                    print(f"   ✅ Finished in {result['runtime_seconds']:.2f}s. F1-Score: {result['f1_score']:.4f}, Recall: {result['recall']:.4f}")
                else:
                    print(f"   ❌ Experiment skipped or failed for {model_name} with {sampling_name} (see errors above).")

    # Save all results to a CSV file
    if all_experiment_results:
        results_df = pd.DataFrame(all_experiment_results)
        results_df.to_csv(RESULTS_FILE, index=False)
        print(f"\n\n🎉 All experiments completed! Results saved to '{RESULTS_FILE}'.")
        print("\n--- Top 5 Models by F1-Score ---")
        print(results_df.sort_values(by='f1_score', ascending=False).head(5)[
            ['model_name', 'sampling_strategy', 'hyperparameters', 'f1_score', 'recall', 'precision']
        ].to_string()) # Use to_string() for better console display
    else:
        print("\n\n⚠️ No experiments were successfully run. No results file generated.")
