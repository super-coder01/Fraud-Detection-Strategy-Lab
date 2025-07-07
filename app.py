# app.py
#
# ======================================================================================
# Fraud Detection Stragety Lab
# By: Archi-Dev Harsh, Bhushit, Aakash, Raj Mishra & Sexy Bhai 😘 
# Version: 2.6 (Golden Master, All Fixes Integrated)
# ======================================================================================

# --- 1. Imports & Page Configuration ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle # For loading .pkl files (XAI artifacts)
import ast # For safely evaluating string representations of dictionaries
import time # For measuring experiment runtime
import io # For handling SHAP force plot HTML output
import os # For checking file existence

# Scikit-learn & Imblearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# TensorFlow (imported only if models.py is loaded)
import tensorflow as tf # Keep at top, but usage will be inside model building

# XAI Library
import shap

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Strategy Lab | Fraud Detection",
    page_icon="🤖",
    layout="wide", # Use wide layout for better visualization space
    initial_sidebar_state="expanded" # Keep sidebar expanded by default
)

# --- 2. Custom Model Loading & Helper Functions ---

# We place model loading here to handle potential import errors gracefully
# If models.py isn't found or has errors, custom models won't be available.
MODELS_LOADED = False
try:
    from models import KNNFromScratch, build_keras_model
    MODELS_LOADED = True
except ImportError:
    st.warning("`models.py` file not found or has errors. Custom models (KNN, NN) "
               "will not be available in the 'Live Experiment Lab'. Please ensure `models.py` is in the same directory.")
except Exception as e:
    st.warning(f"Error importing custom models from `models.py`: {e}. Custom models are disabled.")


@st.cache_data # Cache data loading for performance
def load_all_data():
    """
    Loads all necessary data (creditcard.csv, experiment_results.csv, XAI .pkl files).
    Uses a stratified sample of the main dataset to ensure fast performance in the Live Lab.
    """
    df_full, X_sample, y_sample, results_df, xai_data = None, None, None, None, None
    
    # Load Main Dataset from 'data/' folder
    try:
        df_full = pd.read_csv('data/creditcard.csv') # Path updated to 'data/'
        # Take a stratified sample for faster live experiments. This is crucial for performance.
        # Approx 10,000 rows for the sample.
        sample_size_ratio = min(10000 / len(df_full), 1.0) # Ensure it doesn't exceed 1.0
        X = df_full.drop('Class', axis=1)
        y = df_full['Class']
        # Ensure stratify is used for the sample split too to maintain class balance
        X_sample, _, y_sample, _ = train_test_split(X, y, train_size=sample_size_ratio, random_state=42, stratify=y)
        st.sidebar.info(f"Using a stratified sample of **{len(X_sample)}** rows for live experiments.")
        
    except FileNotFoundError:
        st.error("`data/creditcard.csv` not found. Please ensure it's in the 'data' directory.")
        df_full = None # Set to None to indicate failure
    except Exception as e:
        st.error(f"Error loading `data/creditcard.csv`: {e}")
        df_full = None

    # Load Pre-computed Results from 'artifacts/' folder
    try:
        if os.path.exists("artifacts/experiment_results.csv"): # Path updated to 'artifacts/'
            results_df = pd.read_csv("artifacts/experiment_results.csv") # Path updated to 'artifacts/'
        else:
            st.warning("`artifacts/experiment_results.csv` not found. Please run `run_experiments.py` first to generate pre-computed results.")
            results_df = None
    except Exception as e:
        st.error(f"Error loading `artifacts/experiment_results.csv`: {e}")
        results_df = None

    # Load XAI Data from 'artifacts/' folder
    xai_data = None
    try:
        # Check if all XAI files exist in the 'artifacts/' directory before attempting to load
        xai_file_paths = [
            'artifacts/shap_values.pkl', 
            'artifacts/X_test_scaled.pkl', 
            'artifacts/shap_base_value.pkl', 
            'artifacts/feature_names.pkl'
        ]
        xai_files_exist = all(os.path.exists(f_path) for f_path in xai_file_paths) # <--- THIS IS THE KEY FIX HERE!

        if xai_files_exist:
            with open('artifacts/shap_values.pkl', 'rb') as f: shap_values = pickle.load(f) # Path updated
            with open('artifacts/X_test_scaled.pkl', 'rb') as f: X_test_data = pickle.load(f) # Path updated
            with open('artifacts/shap_base_value.pkl', 'rb') as f: base_value = pickle.load(f) # Path updated
            with open('artifacts/feature_names.pkl', 'rb') as f: feature_names = pickle.load(f) # Path updated
            xai_data = {"shap_values": shap_values, "X_test_data": X_test_data, "base_value": base_value, "feature_names": feature_names}
        else:
            st.warning("XAI data files (`.pkl`) not found in 'artifacts/' directory. Please run `XAI_script.py` to generate them for the XAI tab.")
    except Exception as e:
        st.error(f"Error loading XAI data from 'artifacts/': {e}. Please ensure `XAI_script.py` generated valid files.")
        xai_data = None
        
    return df_full, X_sample, y_sample, results_df, xai_data

def get_model(model_name: str, params: dict, input_shape: int):
    """Initializes a model object based on its name and parameters."""
    if model_name == "Logistic Regression":
        return LogisticRegression(random_state=42, max_iter=1000)
    elif model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42, n_jobs=-1)
    elif model_name == "KNN (From Scratch)" and MODELS_LOADED:
        return KNNFromScratch(k=params["K"], distance_metric=params["distance_metric"])
    elif model_name == "Neural Network" and MODELS_LOADED:
        return build_keras_model(input_shape, params["layers"], params["neurons"], params["learning_rate"])
    return None

# --- 3. MAIN APP LAYOUT & UI ---

st.title("Fraud Detection Strategy Lab")
st.markdown("A comprehensive tool to run live ML experiments and analyze pre-computed results. Created by **Coder Gang**.")

# Load all data upfront
df_full, X_sample, y_sample, results_df, xai_data = load_all_data()

if df_full is None:
    st.error("Cannot proceed without `creditcard.csv`. Please ensure it's in the `data/` directory.")
else:
    # Create the main tabs
    tab1, tab2, tab3 = st.tabs(["**🔬 Live Experiment Lab**", "**📊 Results Analyzer**", "**🧠 XAI Deep-Dive**"])

    # ======================================================================================
    # TAB 1: LIVE EXPERIMENT LAB
    # ======================================================================================
    with tab1:
        st.header("Run a Live Experiment")
        st.markdown("Configure your experiment in the sidebar and click 'Run' to train a model in real-time. "
                    "This tab uses a **sample of the data** for faster execution.⚡")

        # Sidebar Configuration for Live Lab
        st.sidebar.header("🔬 Live Lab Configuration")
        sampling_strategy_live = st.sidebar.selectbox("1. Sampling Strategy", ("None", "SMOTE", "Undersampling"), key='sampling_live')
        
        model_options = ["Logistic Regression", "Random Forest"]
        if MODELS_LOADED: # Only add custom models if successfully loaded
            model_options.extend(["KNN (From Scratch)", "Neural Network"])
        
        model_name_live = st.sidebar.selectbox("2. Model", model_options, key='model_live')
        
        st.sidebar.subheader(f"3. Tune {model_name_live}")
        params_live = {} # Dictionary to hold live experiment hyperparameters
        if model_name_live == "Random Forest":
            params_live["n_estimators"] = st.sidebar.slider("Num. of Trees", 10, 200, 100, 10, key='rf_n_live')
            params_live["max_depth"] = st.sidebar.slider("Max Depth", 2, 20, 10, 1, key='rf_d_live')
        elif model_name_live == "KNN (From Scratch)":
            params_live["K"] = st.sidebar.slider("K Neighbors", 1, 15, 5, 1, key='knn_k_live')
            params_live["distance_metric"] = st.sidebar.selectbox("Distance Metric", ("Euclidean", "Manhattan"), key='knn_dist_live')
        elif model_name_live == "Neural Network":
            params_live["learning_rate"] = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f", key='nn_lr_live')
            params_live["epochs"] = st.sidebar.slider("Epochs", 5, 50, 25, 5, key='nn_ep_live')
            params_live["layers"] = st.sidebar.slider("Num. of Hidden Layers", 1, 5, 2, 1, key='nn_ly_live')
            params_live["neurons"] = st.sidebar.slider("Neurons per Layer", 16, 128, 32, 16, key='nn_nr_live')

        run_button_live = st.sidebar.button("🚀 Run Live Experiment", key='live_run')

        if run_button_live:
            # Check if custom model is selected but not loaded
            if model_name_live in ["KNN (From Scratch)", "Neural Network"] and not MODELS_LOADED:
                st.error(f"Cannot run '{model_name_live}'. The `models.py` file could not be loaded. Please check your setup.")
            else:
                # Splitting and scaling the sample data
                X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42, stratify=y_sample)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Apply sampling strategy to training data
                y_train_resampled = y_train # Initialize with original y_train
                if sampling_strategy_live == "SMOTE": 
                    X_train_scaled, y_train_resampled = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)
                    st.info(f"SMOTE applied. Training data now has {len(X_train_scaled)} samples (before: {len(y_train)}).")
                elif sampling_strategy_live == "Undersampling": 
                    X_train_scaled, y_train_resampled = RandomUnderSampler(random_state=42).fit_resample(X_train_scaled, y_train)
                    st.info(f"Undersampling applied. Training data now has {len(X_train_scaled)} samples (before: {len(y_train)}).")
                
                model = get_model(model_name_live, params_live, X_train_scaled.shape[1])
                
                if model is None:
                    st.error(f"Failed to initialize model '{model_name_live}'. This model might not be available or parameters are incorrect.")
                else:
                    st.subheader(f"Training {model_name_live} with {sampling_strategy_live}...")
                    with st.spinner("Model training in progress... Please wait.⏳"):
                        start_time = time.time()
                        history = None
                        if model_name_live == "Neural Network":
                            # For Keras, fit returns a history object
                            history = model.fit(X_train_scaled, y_train_resampled, 
                                                epochs=params_live.get("epochs", 10), 
                                                batch_size=32, # Using a fixed batch size for live lab
                                                validation_split=0.2, # Use part of training data for validation
                                                verbose=0) # Suppress verbose output
                        else:
                            # For scikit-learn and custom models
                            model.fit(X_train_scaled, y_train_resampled)
                    
                    # Display training completion message
                    st.success(f"Training complete in {time.time() - start_time:.2f} seconds! 🎉")
                    
                    # Add clarification for KNN's "lazy" nature
                    if model_name_live == "KNN (From Scratch)":
                        st.info("💡 **KNN is a 'lazy learner'**: Its 'training' involves primarily storing the data, which is very fast. The main computations for classification occur during prediction.")

                    # Get predictions
                    y_pred = model.predict(X_test_scaled)
                    if model_name_live == "Neural Network": 
                        y_pred = (y_pred > 0.5).astype(int).flatten() # Convert probabilities to binary predictions
                    
                    st.subheader(f"Performance Metrics for {model_name_live}")
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    # Display metrics in columns for easy comparison
                    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                    res_col1.metric("Accuracy", f"{acc:.2%}")
                    res_col2.metric("Precision", f"{prec:.2%}")
                    res_col3.metric("Recall", f"{rec:.2%}")
                    res_col4.metric("F1-Score", f"{f1:.2%}")

                    st.subheader("Visual Diagnostics")
                    vis_col1, vis_col2 = st.columns(2)
                    with vis_col1:
                        st.markdown("##### Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(6, 5)); 
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                                    xticklabels=['Predicted: Not Fraud (0)', 'Predicted: Fraud (1)'], 
                                    yticklabels=['Actual: Not Fraud (0)', 'Actual: Fraud (1)'], ax=ax); 
                        plt.ylabel('Actual Label'); plt.xlabel('Predicted Label'); 
                        st.pyplot(fig)
                        plt.clf() # Clear figure to prevent overlap
                    with vis_col2:
                        st.markdown("##### AUC-ROC Curve")
                        y_pred_proba_roc = None
                        # Check if model has predict_proba (for sklearn) or predict (for Keras)
                        if hasattr(model, 'predict_proba'): 
                            y_pred_proba_roc = model.predict_proba(X_test_scaled)[:, 1]
                        elif model_name_live == "Neural Network":
                            y_pred_proba_roc = model.predict(X_test_scaled).flatten()
                        
                        if y_pred_proba_roc is not None and len(np.unique(y_test)) > 1: # Ensure at least two classes for ROC
                            fpr, tpr, _ = roc_curve(y_test, y_pred_proba_roc); roc_auc = auc(fpr, tpr)
                            fig, ax = plt.subplots(figsize=(6, 5)); 
                            ax.plot(fpr, tpr, color='purple', lw=2, label=f'AUC = {roc_auc:.2f}'); 
                            ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--'); 
                            ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); 
                            ax.set_title('Receiver Operating Characteristic (ROC) Curve'); ax.legend(loc="lower right"); 
                            st.pyplot(fig)
                            plt.clf() # Clear figure
                        else:
                            st.info("Probability predictions not available or only one class present in test set to plot ROC curve.")
                    
                    # Display Neural Network training history (loss over epochs)
                    if model_name_live == "Neural Network" and history is not None:
                        st.subheader("Live Training History")
                        fig, ax = plt.subplots(figsize=(8, 5)); 
                        ax.plot(history.history['loss'], label='Training Loss', color='cyan');
                        if 'val_loss' in history.history: 
                            ax.plot(history.history['val_loss'], label='Validation Loss', color='magenta')
                        ax.set_title('Live Training & Validation Loss'); 
                        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); 
                        st.pyplot(fig)
                        plt.clf() # Clear figure
        else:
            st.info("Configure your experiment in the sidebar and click '🚀 Run Live Experiment' to see real-time results.")

    # ======================================================================================
    # TAB 2: RESULTS ANALYZER
    # ======================================================================================
    with tab2:
        st.header("Analyze Pre-computed Experiment Results")
        
        if results_df is not None:
            st.markdown("This section visualizes comprehensive data from `experiment_results.csv`, generated by `run_experiments.py`.")
            
            with st.expander("Show Raw Experiment Data"): 
                st.dataframe(results_df)

            st.subheader("🏆 Best Performers Leaderboard")
            metric_to_sort = st.selectbox("Sort models by:", ("f1_score", "recall", "precision", "accuracy"), key='sort_metric')
            # Display top 10 results sorted by selected metric
            st.dataframe(results_df.sort_values(by=metric_to_sort, ascending=False).head(10).style.highlight_max(axis=0, subset=[metric_to_sort]))

            st.subheader("🔥 Performance Heatmap")
            heatmap_metric = st.selectbox("Metric for Heatmap:", ("f1_score", "recall", "precision"), key='heatmap_metric')
            # Create a pivot table to visualize performance across models and sampling strategies
            pivot_table = results_df.pivot_table(values=heatmap_metric, index='model_name', columns='sampling_strategy')
            
            fig, ax = plt.subplots(figsize=(12, 7)); # Adjust size for better readability
            sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis", linewidths=.5, linecolor='black', ax=ax); 
            ax.set_title(f"Heatmap of {heatmap_metric.replace('_', ' ').title()} across Models & Sampling Strategies"); 
            plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); # Rotate labels for readability
            st.pyplot(fig)
            plt.clf() # Clear figure

            st.subheader("🔬 Hyperparameter Sensitivity Plots")
            st.markdown("See how different hyperparameters impact model performance.")
            
            # Filter for models that actually have hyperparameters tuned in the results
            # Exclude Logistic Regression as its HPs aren't typically tuned for 'sensitivity' in this context
            available_models_for_hp = [m for m in results_df['model_name'].unique().tolist() if m in ["Random Forest", "KNN (From Scratch)", "Neural Network"]]

            if not available_models_for_hp:
                st.info("No models with tunable hyperparameters found in `experiment_results.csv` to analyze here.")
            else:
                model_for_analysis = st.selectbox("Analyze Model:", available_models_for_hp, key='analysis_model')
                # Create a copy to avoid SettingWithCopyWarning
                df_analysis = results_df[results_df['model_name'] == model_for_analysis].copy()
                
                # Helper to safely extract parameters from string dictionary
                def extract_param(param_str, param_name):
                    try: 
                        # Use ast.literal_eval for safe parsing of string representation of dicts
                        params = ast.literal_eval(param_str)
                        return params.get(param_name)
                    except (ValueError, SyntaxError): 
                        return None # Handle malformed strings or non-dict strings

                if model_for_analysis == "Random Forest":
                    param_to_plot = st.radio("Analyze by:", ("max_depth", "n_estimators"))
                    df_analysis[param_to_plot] = df_analysis['hyperparameters'].apply(lambda x: extract_param(x, param_to_plot))
                    df_analysis.dropna(subset=[param_to_plot], inplace=True) # Drop rows where param couldn't be extracted
                    
                    if not df_analysis.empty:
                        fig, ax = plt.subplots(figsize=(10, 6)); 
                        sns.lineplot(data=df_analysis, x=param_to_plot, y='f1_score', hue='sampling_strategy', marker='o', ax=ax);
                        ax.set_title(f"F1-Score vs. {param_to_plot.replace('_', ' ').title()} for {model_for_analysis}"); 
                        ax.set_xlabel(param_to_plot.replace('_', ' ').title()); ax.set_ylabel("F1-Score"); 
                        st.pyplot(fig)
                        plt.clf()
                    else:
                        st.info(f"No data available to plot for {model_for_analysis} and {param_to_plot}. Check `artifacts/experiment_results.csv`.")

                elif model_for_analysis == "KNN (From Scratch)":
                    df_analysis['K'] = df_analysis['hyperparameters'].apply(lambda x: extract_param(x, 'K'))
                    df_analysis['distance_metric'] = df_analysis['hyperparameters'].apply(lambda x: extract_param(x, 'distance_metric'))
                    df_analysis.dropna(subset=['K', 'distance_metric'], inplace=True)
                    
                    if not df_analysis.empty:
                        fig, ax = plt.subplots(figsize=(10, 6)); 
                        sns.lineplot(data=df_analysis, x='K', y='f1_score', hue='sampling_strategy', style='distance_metric', marker='o', ax=ax);
                        ax.set_title(f"F1-Score vs. K for {model_for_analysis}"); 
                        ax.set_xlabel("Number of Neighbors (K)"); ax.set_ylabel("F1-Score"); 
                        st.pyplot(fig)
                        plt.clf()
                    else:
                        st.info(f"No data available to plot for {model_for_analysis}. Check `artifacts/experiment_results.csv`.")
                
                elif model_for_analysis == "Neural Network":
                    param_to_plot = st.radio("Analyze by:", ("layers", "neurons", "learning_rate", "epochs"))
                    df_analysis[param_to_plot] = df_analysis['hyperparameters'].apply(lambda x: extract_param(x, param_to_plot))
                    df_analysis.dropna(subset=[param_to_plot], inplace=True)
                    
                    if not df_analysis.empty:
                        fig, ax = plt.subplots(figsize=(10, 6)); 
                        # Use a list of unique values for x-axis if discrete (layers, neurons, epochs)
                        # or directly plot for continuous (learning_rate)
                        sns.lineplot(data=df_analysis, x=param_to_plot, y='f1_score', hue='sampling_strategy', marker='o', ax=ax);
                        ax.set_title(f"F1-Score vs. {param_to_plot.replace('_', ' ').title()} for {model_for_analysis}"); 
                        ax.set_xlabel(param_to_plot.replace('_', ' ').title()); ax.set_ylabel("F1-Score"); 
                        st.pyplot(fig)
                        plt.clf()
                    else:
                        st.info(f"No data available to plot for {model_for_analysis} and {param_to_plot}. Check `artifacts/experiment_results.csv`.")
        else:
            st.error("`artifacts/experiment_results.csv` not found. Please run `run_experiments.py` first to generate pre-computed results for this section.")

        # ======================================================================================
        # Conceptual Feature: Financial Impact Simulator
        # ======================================================================================
        st.markdown("---")
        st.subheader("💰 Conceptual Financial Impact Simulator (S-Tier WOW Feature!)")
        st.markdown("Translate model performance metrics into estimated business impact. "
                    "This section demonstrates the **business value** of choosing the right model and metrics.")

        if results_df is not None:
            st.info("Select a model and input hypothetical financial costs to see its estimated impact.")
            
            # Select a model from the results_df
            models_for_sim = results_df[['model_name', 'sampling_strategy', 'hyperparameters', 'precision', 'recall', 'f1_score']].copy()
            # Create a combined identifier for easy selection
            models_for_sim['Model_ID'] = models_for_sim.apply(lambda row: f"{row['model_name']} ({row['sampling_strategy']}, {row['f1_score']:.3f} F1)", axis=1)
            
            selected_model_id = st.selectbox("Choose an Experiment to Simulate:", models_for_sim['Model_ID'])
            selected_row = models_for_sim[models_for_sim['Model_ID'] == selected_model_id].iloc[0]
            
            model_precision = selected_row['precision']
            model_recall = selected_row['recall']
            
            st.markdown(f"**Selected Model Stats:** Precision: **{model_precision:.2%}**, Recall: **{model_recall:.2%}**")

            st.markdown("---")
            st.subheader("Input Hypothetical Scenarios:")
            col_inp1, col_inp2 = st.columns(2)
            with col_inp1:
                total_transactions = st.number_input("Total Transactions per Period (e.g., month)", min_value=1000, value=100000, step=1000)
                fraud_rate = st.slider("Actual Fraud Rate (%)", min_value=0.01, max_value=5.0, value=0.1, step=0.01, format="%.2f") / 100
            with col_inp2:
                cost_fp = st.number_input("Cost of a False Positive ($) (e.g., customer inconvenience, investigation)", min_value=0.0, value=5.0, step=0.1)
                cost_fn = st.number_input("Cost of a False Negative ($) (e.g., actual fraud loss)", min_value=0.0, value=1000.0, step=10.0)
                benefit_tp = st.number_input("Benefit of a True Positive ($) (e.g., fraud loss avoided)", min_value=0.0, value=1000.0, step=10.0)

            # Calculate Actuals
            actual_fraud = total_transactions * fraud_rate
            actual_non_fraud = total_transactions * (1 - fraud_rate)

            # Estimate TP, FP, FN based on Precision and Recall
            # Rec = TP / Actual_Fraud  => TP = Rec * Actual_Fraud
            # Prec = TP / (TP + FP)   => TP + FP = TP / Prec => FP = (TP / Prec) - TP
            # FN = Actual_Fraud - TP
            
            estimated_tp = model_recall * actual_fraud
            
            # Handle division by zero for precision if TP is 0 or precision is 0
            estimated_fp = 0.0
            if model_precision > 0 and estimated_tp > 0:
                estimated_fp = (estimated_tp / model_precision) - estimated_tp
            
            estimated_fn = actual_fraud - estimated_tp
            
            # Ensure estimated counts are non-negative
            estimated_tp = max(0, estimated_tp)
            estimated_fp = max(0, estimated_fp)
            estimated_fn = max(0, estimated_fn)
            
            estimated_tn = actual_non_fraud - estimated_fp # For completeness

            # Calculate Financial Impact
            total_benefit = estimated_tp * benefit_tp
            total_fp_cost = estimated_fp * cost_fp
            total_fn_cost = estimated_fn * cost_fn
            
            net_financial_impact = total_benefit - total_fp_cost - total_fn_cost
            
            st.markdown("---")
            st.subheader("Estimated Financial Impact:")
            st.info(f"For {total_transactions:,} transactions with a {fraud_rate:.2%} fraud rate:")
            
            metrics_col1, metrics_col2 = st.columns(2)
            metrics_col1.metric("Estimated True Positives (TP)", f"{estimated_tp:,.0f}")
            metrics_col2.metric("Estimated False Positives (FP)", f"{estimated_fp:,.0f}")
            metrics_col1.metric("Estimated False Negatives (FN)", f"{estimated_fn:,.0f}")
            metrics_col2.metric("Estimated True Negatives (TN)", f"{estimated_tn:,.0f}")

            st.markdown("### **Net Financial Impact**")
            impact_style = "green" if net_financial_impact >= 0 else "red"
            st.markdown(f"<h1 style='text-align: center; color: {impact_style};'>${net_financial_impact:,.2f}</h1>", unsafe_allow_html=True)
            
            st.markdown(f"""
                - **Total Benefit from Detected Fraud (TP):** ${total_benefit:,.2f} (from {estimated_tp:,.0f} TP cases)
                - **Total Cost of False Positives (FP):** -${total_fp_cost:,.2f} (from {estimated_fp:,.0f} FP cases)
                - **Total Cost of Missed Fraud (FN):** -${total_fn_cost:,.2f} (from {estimated_fn:,.0f} FN cases)
            """)

            st.markdown("---")
            st.caption("💡 This simulator provides an estimation based on the chosen model's performance metrics and your hypothetical costs. Real-world impact may vary.")

        else:
            st.info("Run `run_experiments.py` and ensure `artifacts/experiment_results.csv` is generated to use the Financial Impact Simulator.")


    # ======================================================================================
    # TAB 3: XAI DEEP-DIVE
    # ======================================================================================
    with tab3:
        st.header("🧠 Explain the Best Model's Predictions (XAI)")
        
        if xai_data: # Check if XAI data was loaded successfully
            shap_values = xai_data["shap_values"]
            X_test_data = xai_data["X_test_data"]
            base_value = xai_data["base_value"]
            feature_names = xai_data["feature_names"]
            
            # Convert X_test_data (numpy array) to DataFrame with feature names for SHAP plotting
            X_test_df = pd.DataFrame(X_test_data, columns=feature_names)

            st.info("This section uses SHAP (SHapley Additive exPlanations) to explain our best model's predictions. "
                    "The SHAP data is generated by `XAI_script.py` (typically for Random Forest + SMOTE).")

            st.subheader("1. Global Feature Importance (Summary Plot)")
            st.markdown("This plot shows the overall impact and direction of each feature on the model's output (prediction of 'Fraud').")
            st.markdown("""
            - Each dot represents a single prediction for a feature.
            - **X-axis (SHAP value):** How much a feature's value contributes to pushing the prediction higher (positive SHAP) or lower (negative SHAP).
            - **Color (Feature Value):** Red indicates a high feature value, blue indicates a low feature value.
            - **Y-axis (Feature Name):** Features are ordered by their global importance (average absolute SHAP value).
            """)
            
            # Ensure the plot is drawn within a new figure context for Streamlit
            fig_summary_plot = plt.figure(figsize=(10, 7)) # Create a new figure
            # Using X_test_df for feature names and proper alignment
            shap.summary_plot(shap_values, X_test_df, show=False)
            st.pyplot(fig_summary_plot, bbox_inches='tight') # bbox_inches='tight' helps prevent clipping
            plt.close(fig_summary_plot) # Close the figure to free memory

            st.markdown("---")

            st.subheader("2. Local Prediction Explanation (Force Plot)")
            st.markdown("Select a specific prediction from the test set to see a detailed 'force plot' for why that particular prediction was made.")
            
            prediction_index = st.slider("Select a prediction (instance) to explain:", 0, len(X_test_df) - 1, 10, 1)
            
            st.markdown(f"**Explaining Prediction #{prediction_index}:**")
            st.dataframe(X_test_df.iloc[prediction_index:prediction_index+1]) # Show the features of the selected instance
            
            # Generate the SHAP force plot HTML. `matplotlib=False` is crucial for Streamlit compatibility.
            force_plot = shap.force_plot(base_value, shap_values[prediction_index, :], X_test_df.iloc[prediction_index, :], matplotlib=False)
            
            # Display the force plot HTML using Streamlit's components.v1.html
            html_buffer = io.StringIO()
            shap.save_html(html_buffer, force_plot)
            st.components.v1.html(html_buffer.getvalue(), height=200, scrolling=True) # Adjust height as needed

            with st.expander("🤔 How to read this Force Plot?"):
                st.markdown("""
                - The **base value** (E[f(x)]) is the average prediction output across the dataset. This is where the prediction starts.
                - **<span style='color:red;'>Red arrows</span>** (features) represent values that pushed the prediction **HIGHER** (towards 'Fraud').
                - **<span style='color:blue;'>Blue arrows</span>** (features) represent values that pushed the prediction **LOWER** (away from 'Fraud').
                - The length of the arrow indicates the magnitude of the impact.
                - **f(x)** is the final output score for the specific prediction. For a sigmoid output, this is typically the probability of fraud.
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.info("💡 **Deep Dive**: XAI helps build trust in complex models by revealing the reasons behind their decisions, which is critical in high-stakes domains like fraud detection.")

        else:
            st.warning("XAI data files (`.pkl`) not found or could not be loaded. Please run `XAI_script.py` to generate them for this section.")
            st.info("Ensure the following files are present in the `artifacts/` directory: `shap_values.pkl`, `X_test_scaled.pkl`, `shap_base_value.pkl`, `feature_names.pkl`.")