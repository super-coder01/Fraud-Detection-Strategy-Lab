# 🤖Fraud Detection Strategy Lab 🔥

## Project Overview

Welcome, future Data Scientists and ML practitioners! 👋

Ever wondered why achieving **99.8% accuracy** in a machine learning model can sometimes be a **complete failure**? Especially in critical areas like **Fraud Detection**? That's the **"Accuracy Paradox"**, and this project is your interactive guide to mastering it!

This lab is more than just code; it's an **interactive educational tool** and a powerful **analysis dashboard**. Its mission is to build your intuition around:

*   **Handling Highly Imbalanced Data:** Why accuracy falls short, and what metrics truly matter.
*   **Business-Relevant Metrics:** A deep dive into Precision, Recall, and F1-Score, and their real-world implications.
*   **Understanding Tradeoffs:** Visualizing how different models, data sampling techniques, and hyperparameters impact performance.
*   **Explainable AI (XAI):** Unlocking the "why" behind your model's predictions, crucial for trust in high-stakes decisions.

**Who is this Lab for?** Students, junior data scientists, and managers eager to understand the practical challenges and strategic choices in building effective ML systems for fraud.

## ✨ Core Features of the Lab (`app.py`)

Our Streamlit web application (`app.py`) is your interactive playground, featuring three main tabs:

1.  **🔬 Live Experiment Lab:**
    *   **Real-time Training:** Configure a single ML experiment on the fly. Choose your model (Logistic Regression, Random Forest, Custom KNN, Neural Network), data sampling strategy (SMOTE, Undersampling, None), and tune hyperparameters.
    *   **Instant Results:** See performance metrics (Accuracy, Precision, Recall, F1-Score), Confusion Matrix, and ROC Curve immediately after training.
    *   **Fast Execution:** Uses a pre-sampled portion of the dataset for quick iterations, making live exploration a breeze.

2.  **📊 Results Analyzer:**
    *   **Best Performers Leaderboard:** Explore pre-computed results from dozens of experiments, ranked by your chosen metric. Discover top model/sampling/hyperparameter combinations.
    *   **Performance Heatmap:** Get an at-a-glance comparison of model performance across different sampling strategies.
    *   **Hyperparameter Sensitivity:** Visualize how tweaking parameters (e.g., `max_depth` for Random Forest, `K` for KNN, `learning_rate` for Neural Network) influences model effectiveness.
    *   **💰 Conceptual Financial Impact Simulator (S-Tier WOW Feature!):** Translate abstract ML metrics into estimated real-world financial savings and losses. This feature bridges the gap between model performance and business value.

3.  **🧠 XAI Deep-Dive:**
    *   **SHAP Integration:** Understand *why* our best model (typically Random Forest + SMOTE) makes its predictions.
    *   **Global Feature Importance:** A `summary_plot` showing which features are most influential overall and their impact direction.
    *   **Local Prediction Explanation:** Use an interactive `force_plot` to dissect a single prediction, revealing how each feature contributed to that specific outcome. Essential for building trust in ML models.

## ⚙️ Technical Architecture & Stack

The "Fraud Detection Strategy Lab" is structured for clarity and efficiency:

*   **Part 1: The Offline "Research Rig" (`run_experiments.py`)**
    *   Systematically runs a predefined grid of experiments (model, sampling, hyperparameters).
    *   **Output:** `artifacts/experiment_results.csv` (a comprehensive log of all experiment outcomes).
*   **Part 2: The XAI Illuminator (`XAI_script.py`)**
    *   Identifies the best model from `artifacts/experiment_results.csv`.
    *   Generates SHAP explanation artifacts (`.pkl` files) for that best model.
*   **Part 3: The Online "Analysis Dashboard" (`app.py`)**
    *   The interactive Streamlit web application that utilizes `artifacts/experiment_results.csv` and the XAI `.pkl` files for dynamic visualizations and live experiments.

### 🛠️ Tech Stack:

*   **Frontend/UI:** `Streamlit`
*   **Data Handling:** `Pandas`, `NumPy`
*   **ML Libraries:** `Scikit-learn`, `Imbalanced-learn` (for SMOTE/Undersampling)
*   **Deep Learning:** `TensorFlow (Keras)` (for custom Neural Network)
*   **XAI (Explainability):** `SHAP`
*   **Custom Code (`models.py`):** Contains `KNNFromScratch` (for understanding first principles) and `build_keras_model` (for dynamic NN creation).

## 💡 Key Learnings & Insights

By interacting with this lab, you'll gain deep insights into:

*   **The Accuracy Trap:** Understand firsthand why accuracy is a poor metric for imbalanced problems and the importance of focusing on Precision, Recall, and F1-Score.
*   **The Power of Sampling:** Learn how data-level techniques like SMOTE can often have a more significant impact on detecting rare events (like fraud) than complex model tuning.
*   **First Principles vs. Optimization:** The `KNNFromScratch` implementation highlights algorithmic fundamentals and showcases the performance benefits of optimized library versions.
*   **The Value of Explainable AI:** Realize that in high-stakes domains, understanding *why* a model makes a decision is as crucial as the decision itself, fostering trust and enabling informed actions.

## 🚀 Getting Started (Setup & Run)

Follow these steps to set up and run the "Fraud Detection Strategy Lab" on your local machine!

### 1. Prerequisites

*   **Python:** Ensure you have Python 3.8+ installed.
*   **Dataset:** Download the `creditcard.csv` dataset (e.g., from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)) and place it inside the **`data/`** directory of this project. **Note: This large file is not included in the repository and must be downloaded separately.**

### 2. Project Structure

Ensure your project directory is organized as follows:

```
.
├── .gitignore              # Defines files/folders to be ignored by Git
├── LICENSE                 # Project's open-source license
├── README.md               # You are here!
├── requirements.txt        # All Python dependencies
├── app.py                  # The main Streamlit application
├── models.py               # Custom ML model implementations
├── run_experiments.py      # Script to run all experiments
├── XAI_script.py           # Script to generate XAI data
├── data/                   # Folder to store raw datasets
│   └── creditcard.csv      # <--- DOWNLOAD & PLACE THIS FILE HERE
└── artifacts/              # Folder for generated outputs (will be created/populated by scripts)
    # ├── experiment_results.csv
    # ├── shap_values.pkl
    # ├── X_test_scaled.pkl
    # ├── shap_base_value.pkl
    # └── feature_names.pkl
```

### 3. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies:

```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source ./.venv/bin/activate
```

### 4. Install Dependencies

Once your virtual environment is active, install all required Python libraries:

```bash
pip install -r requirements.txt
```
*(Manual installation: `pip install streamlit pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow shap`)*

### 5. Run the Project (Step-by-Step)

Follow these steps in order to fully populate the app with pre-computed results and XAI data. Ensure you are in the root directory of your project (e.g., `Fraud-Detection-Strategy-Lab`) when running these commands.

#### a. Run the Offline Experiment Rig (`run_experiments.py`)

This script will train various models with different sampling strategies and hyperparameters, saving the results to **`artifacts/experiment_results.csv`**. This might take some time depending on your system and the `SAMPLE_SIZE_FRACTION` configured in the script.

```bash
python run_experiments.py
```
*Expected Output:* You will see progress messages for each experiment. Upon completion, an `experiment_results.csv` file will be generated inside the `artifacts/` folder.

#### b. Generate XAI Artifacts (`XAI_script.py`)

This script will identify the best model (typically Random Forest with SMOTE) from `artifacts/experiment_results.csv`, re-train it, and generate the necessary SHAP explanation files (`.pkl`) inside the **`artifacts/`** folder for the "XAI Deep-Dive" tab.

```bash
python XAI_script.py
```
*Expected Output:* Confirmation messages that `shap_values.pkl`, `X_test_scaled.pkl`, `shap_base_value.pkl`, and `feature_names.pkl` have been created inside the `artifacts/` folder.

#### c. Launch the Interactive Lab (`app.py`)

Finally, start the Streamlit web application. This is your main interactive dashboard!

```bash
streamlit run app.py
```
*Expected Output:* Your web browser should automatically open a new tab displaying the "Fraud Detection Strategy Lab". Enjoy exploring!

## 🙏 Acknowledgements

This project was a collaborative effort by the amazing **Coder Gang**: Archi-Dev Harsh, Bhushit, Aakash, Raj Mishra, and Sexy Bhai 😘! Special thanks to the open-source community for the incredible libraries that made this possible.