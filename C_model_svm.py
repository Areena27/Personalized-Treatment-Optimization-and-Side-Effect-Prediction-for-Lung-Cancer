import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform
import time

def load_data(csv_path):
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError("The specified file path is invalid.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or contains invalid data.")
    
    if data.isnull().values.any():
        print("Warning: Missing values detected. Filling missing values with column means.")
        data.fillna(data.mean(), inplace=True)
    
    try:
        features = data.iloc[:, :768].values  # Feature embeddings: columns 0-767
        treatment_labels = data.iloc[:, 768].values  # Treatment type: column 768
        drug_labels = data.iloc[:, 769].values  # Drug: column 769
        side_effects_labels = data.iloc[:, 770:].values  # Side effects: columns 770-793
    except IndexError:
        raise ValueError("Column indices for features and labels are invalid or out of range.")
    
    return features, treatment_labels, drug_labels, side_effects_labels

def preprocess_data(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    joblib.dump(scaler, "ml_project/C_project/C_scaler.joblib")  # Save the scaler
    return scaled_features, scaler

def log_time(start_time, task_name):
    print(f"{task_name} took {time.time() - start_time:.2f} seconds.")

def train_model(X_train, X_val, y_train, y_val, model_name, param_dist, is_multi_output=False):
    model = SVC(probability=True, random_state=42)
    
    if is_multi_output:
        model = MultiOutputClassifier(model)
        param_dist = {
            'estimator__C': uniform(0.1, 10),
            'estimator__gamma': ['scale', 'auto'],
            'estimator__kernel': ['linear', 'rbf', 'poly']
        }
    else:
        param_dist = {
            'C': uniform(0.1, 10),
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf', 'poly']
        }
    
    randomized_search = RandomizedSearchCV(
        model, param_dist, n_iter=10, random_state=42, n_jobs=-1, scoring="accuracy"
    )
    randomized_search.fit(X_train, y_train)
    
    best_model = randomized_search.best_estimator_
    joblib.dump(best_model, f'{model_name}.joblib')
    print(f"{model_name} saved successfully!")
    
    val_score = randomized_search.score(X_val, y_val)
    print(f"{model_name} Validation Accuracy: {val_score:.4f}")
    
    return best_model

def train_all_models(features, treatment_labels, drug_labels, side_effects_labels):
    start_time = time.time()
    
    # Preprocessing
    features, scaler = preprocess_data(features)
    log_time(start_time, "Data Preprocessing")
    
    # Data splitting
    X_train, X_temp, y_train_treatment, y_temp_treatment = train_test_split(
        features, treatment_labels, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val_treatment, y_test_treatment = train_test_split(
        X_temp, y_temp_treatment, test_size=0.5, random_state=42
    )
    
    _, _, y_train_drug, y_temp_drug = train_test_split(
        features, drug_labels, test_size=0.4, random_state=42
    )
    _, _, y_val_drug, y_test_drug = train_test_split(
        X_temp, y_temp_drug, test_size=0.5, random_state=42
    )
    
    _, _, y_train_side_effects, y_temp_side_effects = train_test_split(
        features, side_effects_labels, test_size=0.4, random_state=42
    )
    _, _, y_val_side_effects, y_test_side_effects = train_test_split(
        X_temp, y_temp_side_effects, test_size=0.5, random_state=42
    )
    
    log_time(start_time, "Data Splitting")
    
    # Hyperparameter tuning parameter distribution
    param_dist = {
        'C': uniform(0.1, 10),
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf', 'poly']
    }
    
    # Train treatment model
    treatment_model = train_model(X_train, X_val, y_train_treatment, y_val_treatment, "C_svm_treatment", param_dist)
    
    # Train drug model
    drug_model = train_model(X_train, X_val, y_train_drug, y_val_drug, "C_svm_drug", param_dist)
    
    # Train side effects model using MultiOutputClassifier
    side_effects_model = train_model(
        X_train, X_val, y_train_side_effects, y_val_side_effects,
        "C_svm_side_effects", param_dist, is_multi_output=True
    )
    
    log_time(start_time, "Model Training and Saving")
    return treatment_model, drug_model, side_effects_model

# Example usage
csv_path = 'ml_project/C_project/C_embeddingc.csv'
features, treatment_labels, drug_labels, side_effects_labels = load_data(csv_path)

print("Treatment Labels Shape:", treatment_labels.shape)
print("Drug Labels Shape:", drug_labels.shape)
print("Side Effects Labels Shape:", side_effects_labels.shape)

# Call the training function
train_all_models(features, treatment_labels, drug_labels, side_effects_labels)
