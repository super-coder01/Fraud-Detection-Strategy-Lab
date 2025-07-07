# models.py
#
# Project: The Ultimate Interactive AI Strategy Lab: Fraud Detection
# Author: Archi-Dev Harsh, Bhushit, Aakash, Raj Mishra & Sexy Bhai 😘 
# Version: 2.6 - Golden Master, All Fixes Integrated
#
# Description:
# This file contains custom implementations of machine learning models, specifically
# a K-Nearest Neighbors (KNN) classifier built from scratch using NumPy, and
# a dynamic Keras Neural Network builder. These custom models are integral to
# demonstrating fundamental algorithmic principles and providing flexibility
# within the "Live Experiment Lab" and "Results Analyzer" sections of the Streamlit app.
#
# Key Features:
# - KNNFromScratch: Illustrates a "lazy learner" where training means storing data
#   and prediction involves on-the-fly distance calculations and majority voting.
#   It's intentionally less optimized than library versions to highlight performance
#   benefits of optimized libraries.
# - build_keras_model: Provides a flexible way to construct simple feed-forward
#   neural networks, allowing users to experiment with different architectures
#   (number of layers, neurons) and learning rates.
#
# Dependencies:
# - numpy: For numerical operations in KNNFromScratch.
# - tensorflow: For building and compiling the Keras Neural Network.
# - pandas: For robust handling of Series/DataFrame types, especially for target labels.
#
# This file is imported by `app.py` and `run_experiments.py`.
# ======================================================================================

import numpy as np
from collections import Counter
from typing import Union # For type hinting multiple possible types

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore
import pandas as pd # Import pandas to use pd.Series constructor


# ======================================================================================
# 1. K-Nearest Neighbors (From Scratch)
# ======================================================================================

class KNNFromScratch:
    """
    A K-Nearest Neighbors (KNN) classifier implemented from scratch using only NumPy.

    This class serves as an educational tool to demystify the fundamental logic of
    instance-based, distance-based classification. It intentionally avoids advanced
    optimizations found in production-ready libraries to highlight the core
    computational steps and performance differences.

    Attributes:
        k (int): The number of neighbors to consider for the majority vote during prediction.
        distance_metric (str): The metric used for calculating distance between data points.
                               Supported: 'Euclidean' or 'Manhattan'.
        X_train (np.ndarray): The training feature data stored during the fit method.
        y_train (pd.Series): The training target labels stored during the fit method.
                             Stored as a pandas Series for consistent `.iloc` access.
    """
    def __init__(self, k: int = 5, distance_metric: str = 'Euclidean'):
        """
        Initializes the K-Nearest Neighbors classifier.

        Args:
            k (int): The number of neighbors (default: 5) to consider when making
                     a prediction for a new data point. Must be a positive integer.
            distance_metric (str): The type of distance metric to use.
                                   Options: 'Euclidean' (default) or 'Manhattan'.
                                   Case-sensitive.
        Raises:
            ValueError: If an unsupported distance_metric is provided during initialization.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if distance_metric not in ['Euclidean', 'Manhattan']:
            raise ValueError("Unsupported distance metric. Choose 'Euclidean' or 'Manhattan'.")
            
        self.k: int = k
        self.distance_metric: str = distance_metric
        self.X_train: np.ndarray = np.array([]) # Initialize as empty arrays
        self.y_train: pd.Series = pd.Series([]) # Initialize as empty Series

    def fit(self, X_train: np.ndarray, y_train: Union[pd.Series, np.ndarray]) -> None:
        """
        "Fits" the K-Nearest Neighbors model by simply storing the training data.

        KNN is a "lazy learner," meaning it does not perform any explicit model
        training or parameter learning during the fit phase. All computation
        occurs during the prediction phase.

        Args:
            X_train (np.ndarray): The training feature data (features of samples).
                                  Expected to be a 2D NumPy array.
            y_train (Union[pd.Series, np.ndarray]): The training target labels (classes).
                                                    Expected to be a 1D NumPy array or pandas Series.
                                                    It will be internally converted to a pandas Series
                                                    for consistent indexing.
        """
        self.X_train = X_train
        # Ensure y_train is a pandas Series for consistent .iloc access in _predict_single
        if isinstance(y_train, np.ndarray):
            self.y_train = pd.Series(y_train)
        else:
            self.y_train = y_train

    def _calculate_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculates the distance between two data points based on the specified metric.

        This is a private helper method used internally by the `predict` function.

        Args:
            x1 (np.ndarray): The first data point (1D NumPy array).
            x2 (np.ndarray): The second data point (1D NumPy array).

        Returns:
            float: The calculated distance between x1 and x2.

        Raises:
            ValueError: If an unsupported distance metric was set during initialization.
                        (This should ideally be caught by __init__, but added for robustness).
        """
        if self.distance_metric == 'Euclidean':
            # Euclidean distance: sqrt(sum((x1 - x2)^2))
            return np.sqrt(np.sum((x1 - x2)**2))
        elif self.distance_metric == 'Manhattan':
            # Manhattan distance: sum(|x1 - x2|)
            return np.sum(np.abs(x1 - x2))
        else:
            # This error should ideally not be reachable if __init__ validates distance_metric
            raise ValueError(f"Unsupported distance metric: '{self.distance_metric}'. "
                             "Choose 'Euclidean' or 'Manhattan'.")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts the class label for each sample in the test set.

        For each test sample, it finds its 'k' nearest neighbors in the training
        data and assigns the class label that is most frequent among these neighbors
        (majority vote).

        Args:
            X_test (np.ndarray): The test feature data for which predictions are to be made.
                                 Expected to be a 2D NumPy array.

        Returns:
            np.ndarray: An array of predicted class labels for each sample in X_test.
                        The array will be 1D and contain integer labels.
        """
        # Use a list comprehension for efficiency and readability to call _predict_single
        # for each sample in the test set.
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions, dtype=int) # Ensure output is a NumPy array of integers

    def _predict_single(self, x: np.ndarray) -> int:
        """
        Helper function to predict the class label for a single data point.

        This is a private method called by the public `predict` function.

        Args:
            x (np.ndarray): A single data point (1D NumPy array) for which to predict the class.

        Returns:
            int: The predicted class label (0 or 1 for binary classification).
        """
        # 1. Calculate the distance from the new point 'x' to all training points
        # This loop-based approach is intentionally less optimized to showcase
        # the performance benefits of vectorized operations in libraries like scikit-learn.
        distances = [self._calculate_distance(x, x_train) for x_train in self.X_train]
        
        # 2. Get the indices of the 'k' nearest neighbors
        # np.argsort returns the indices that would sort an array. We take the first 'k'.
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Retrieve the labels of these k neighbors
        # We use .iloc because self.y_train is guaranteed to be a pandas Series,
        # allowing for robust integer-location based indexing.
        k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]
        
        # 4. Return the most common class label among the k neighbors (majority vote)
        # collections.Counter is used to count occurrences of each label.
        # .most_common(1) returns a list of the single most common element and its count.
        most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
        return int(most_common_label) # Ensure return type is int


# ======================================================================================
# 2. Keras Neural Network Builder
# ======================================================================================

def build_keras_model(input_shape: int, layers: int, neurons: int, learning_rate: float) -> tf.keras.Model:
    """
    Dynamically builds and compiles a Keras Sequential model based on user-defined parameters.

    This function provides a flexible way to create simple feed-forward neural networks
    for binary classification tasks. It uses a ReLU activation function for hidden layers
    and a sigmoid activation for the output layer. The model is compiled with the Adam
    optimizer and binary cross-entropy loss, suitable for binary classification.

    Args:
        input_shape (int): The number of input features (i.e., the dimensionality of
                           the input data). This defines the shape of the first layer.
        layers (int): The number of hidden layers to include in the neural network.
                      Must be a non-negative integer.
        neurons (int): The number of neurons (units) in each hidden layer.
                       Must be a positive integer.
        learning_rate (float): The learning rate for the Adam optimizer.
                               A smaller learning rate means smaller updates to weights
                               during training.

    Returns:
        tf.keras.Model: The compiled Keras model, ready for training.

    Raises:
        ValueError: If `layers` is negative or `neurons` is not positive.
    """
    if layers < 0:
        raise ValueError("Number of layers cannot be negative.")
    if neurons <= 0:
        raise ValueError("Number of neurons must be a positive integer.")
    if not (0 < learning_rate <= 1.0): # Basic check, can be refined based on common LR ranges
        print(f"Warning: Learning rate {learning_rate} is outside typical range (0, 1].") # Using print for now, can be log

    model = Sequential()
    
    # Input layer: Defines the expected shape of the input data.
    model.add(Input(shape=(input_shape,)))
    
    # Hidden layers: Dynamically added based on the 'layers' parameter.
    # Each hidden layer uses the ReLU (Rectified Linear Unit) activation function,
    # which helps with training deep networks.
    for _ in range(layers):
        model.add(Dense(neurons, activation='relu'))
        
    # Output layer: For binary classification, a single neuron with a 'sigmoid'
    # activation function is used. Sigmoid outputs a probability score between 0 and 1.
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model: Configures the model for training.
    # - Optimizer: Adam is a popular and efficient optimization algorithm.
    # - Loss Function: 'binary_crossentropy' is standard for binary classification problems.
    # - Metrics: 'accuracy' is tracked during training to evaluate performance.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


# ======================================================================================
# Test Block: Executed only when models.py is run directly
# ======================================================================================
if __name__ == "__main__":
    print("🚀 Starting self-test for models.py...")
    print("---------------------------------------")

    # --- Setup Synthetic Data for Testing ---
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.datasets import make_classification # For easily generating synthetic data

    print("Generating synthetic dataset for testing...")
    # Create a synthetic dataset that mimics a highly imbalanced fraud scenario
    X_synth, y_synth = make_classification(
        n_samples=500,        # Total samples
        n_features=10,        # Number of features
        n_informative=5,      # Features that are actually useful
        n_redundant=2,        # Redundant features
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        weights=[0.98, 0.02], # Highly imbalanced: 98% class 0, 2% class 1 (fraud)
        flip_y=0,             # Noisy labels
        random_state=42       # For reproducibility
    )

    # Split data
    X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(
        X_synth, y_synth, test_size=0.3, random_state=42, stratify=y_synth
    )

    # Scale features
    scaler_synth = StandardScaler()
    X_train_scaled_synth = scaler_synth.fit_transform(X_train_synth)
    X_test_scaled_synth = scaler_synth.transform(X_test_synth)

    print(f"Synthetic Data Shapes: X_train={X_train_scaled_synth.shape}, y_train={y_train_synth.shape}")
    print(f"                       X_test={X_test_scaled_synth.shape}, y_test={y_test_synth.shape}")
    print(f"Class distribution in y_train: {Counter(y_train_synth)}")
    print(f"Class distribution in y_test: {Counter(y_test_synth)}")
    print("---------------------------------------")


    # --- Test KNNFromScratch ---
    print("🧪 Testing KNNFromScratch...")
    try:
        # Test with Euclidean distance
        knn_euclidean = KNNFromScratch(k=5, distance_metric='Euclidean')
        knn_euclidean.fit(X_train_scaled_synth, y_train_synth)
        y_pred_knn_euclidean = knn_euclidean.predict(X_test_scaled_synth)
        
        print("\n   Results for KNN (Euclidean):")
        print(f"     Accuracy:  {accuracy_score(y_test_synth, y_pred_knn_euclidean):.4f}")
        print(f"     Precision: {precision_score(y_test_synth, y_pred_knn_euclidean, zero_division=0):.4f}")
        print(f"     Recall:    {recall_score(y_test_synth, y_pred_knn_euclidean, zero_division=0):.4f}")
        print(f"     F1-Score:  {f1_score(y_test_synth, y_pred_knn_euclidean, zero_division=0):.4f}")

        # Test with Manhattan distance
        knn_manhattan = KNNFromScratch(k=5, distance_metric='Manhattan')
        knn_manhattan.fit(X_train_scaled_synth, y_train_synth)
        y_pred_knn_manhattan = knn_manhattan.predict(X_test_scaled_synth)

        print("\n   Results for KNN (Manhattan):")
        print(f"     Accuracy:  {accuracy_score(y_test_synth, y_pred_knn_manhattan):.4f}")
        print(f"     Precision: {precision_score(y_test_synth, y_pred_knn_manhattan, zero_division=0):.4f}")
        print(f"     Recall:    {recall_score(y_test_synth, y_pred_knn_manhattan, zero_division=0):.4f}")
        print(f"     F1-Score:  {f1_score(y_test_synth, y_pred_knn_manhattan, zero_division=0):.4f}")

    except Exception as e:
        print(f"   ❌ Error during KNNFromScratch test: {e}")
    print("---------------------------------------")


    # --- Test Keras Neural Network Builder ---
    print("🧪 Testing build_keras_model...")
    try:
        input_dim = X_train_scaled_synth.shape[1]
        nn_model = build_keras_model(
            input_shape=input_dim,
            layers=2,
            neurons=32,
            learning_rate=0.001
        )
        print("\n   Keras Model Summary:")
        nn_model.summary() # Print model architecture

        print("\n   Training Keras Model (briefly)...")
        # Train for a few epochs only for testing purposes
        nn_model.fit(X_train_scaled_synth, y_train_synth, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
        print("   ...Training complete.")

        print("\n   Evaluating Keras Model:")
        y_pred_nn_proba = nn_model.predict(X_test_scaled_synth).flatten()
        y_pred_nn = (y_pred_nn_proba > 0.5).astype(int)

        print(f"     Accuracy:  {accuracy_score(y_test_synth, y_pred_nn):.4f}")
        print(f"     Precision: {precision_score(y_test_synth, y_pred_nn, zero_division=0):.4f}")
        print(f"     Recall:    {recall_score(y_test_synth, y_pred_nn, zero_division=0):.4f}")
        print(f"     F1-Score:  {f1_score(y_test_synth, y_pred_nn, zero_division=0):.4f}")

    except Exception as e:
        print(f"   ❌ Error during Keras Model test: {e}")
    
    print("---------------------------------------")
    print("✅ All models.py self-tests completed. You're ready for the next phase, Archi-Dev!")
    print("---------------------------------------")