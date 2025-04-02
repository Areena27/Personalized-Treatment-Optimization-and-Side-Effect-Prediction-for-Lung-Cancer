import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(csv_path):
    """Load and preprocess the dataset."""
    data = pd.read_csv(csv_path)
    features = data.iloc[:, :768].values  # Feature embeddings: columns 0-767
    treatment_labels = data.iloc[:, 768].values  # Treatment type: column 768
    drug_labels = data.iloc[:, 769].values  # Drug: column 769
    side_effects_labels = data.iloc[:, 770:].values  # Side effects: columns 770-793
    return features, treatment_labels, drug_labels, side_effects_labels

def preprocess_data(features):
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def plot_confusion_matrix(y_test, y_pred, model_name, labels):
    """Plot confusion matrix for the evaluation."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def evaluate_model(model, X_test, y_test, model_name, labels, is_multi_output=False):
    """
    Evaluate the model and print performance metrics.
    For multi-output models, metrics are calculated per label.
    """
    y_pred = model.predict(X_test)

    if not is_multi_output:
        # Calculate confusion matrix for single-label models
        plot_confusion_matrix(y_test, y_pred, model_name, labels)

    if is_multi_output:
        # Calculate metrics for each output (side effects)
        print(f"\nEvaluation for {model_name}:")
        total_samples = y_test.shape[0]

        for i in range(y_test.shape[1]):
            print(f"\nSideEffect_{i}:")
            print(classification_report(y_test[:, i], y_pred[:, i], zero_division=0))

        # Calculate overall accuracy with relaxed definition
        incorrect_predictions = np.sum(y_test != y_pred, axis=1)
        relaxed_correct = np.sum(incorrect_predictions <= 1)
        relaxed_accuracy = relaxed_correct / total_samples

        print(f"\nOverall Accuracy: {relaxed_accuracy:.4f}")

    else:
        # Single-label metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"\nEvaluation for {model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

from C_generate_clinical_notes import calculate_treatment_score, calculate_side_effects


def predict_treatment(X_val, treatment_model, drug_model, side_effects_model):
    """
    Generate predictions using ML models and rule-based logic.
    Handles cases where only embeddings are available in X_val.
    """
    print("Generating predictions...")

    # Separate BERT embeddings
    embeddings = X_val[:, :768]  # First 768 columns are embeddings

    # Check if structured columns exist
    if X_val.shape[1] > 768:
        treatment_column = X_val[:, 768]  # Treatment type
        drug_column = X_val[:, 769]      # Drug type
        side_effects_columns = X_val[:, 770:793]  # Side effects
    else:
        treatment_column = None
        drug_column = None
        side_effects_columns = None

    # ML Model Predictions
    treatment_preds_ml = treatment_model.predict(embeddings)
    drug_preds_ml = drug_model.predict(embeddings)
    side_effects_preds_ml = side_effects_model.predict(embeddings)

    # Rule-Based Predictions
    rule_based_predictions = []
    for i, embedding in enumerate(embeddings):
        # Use placeholder values if structured columns are missing
        treatment = treatment_column[i] if treatment_column is not None else None
        drug = drug_column[i] if drug_column is not None else None
        side_effects = side_effects_columns[i] if side_effects_columns is not None else None

        # Apply rule-based logic (skip if structured columns are missing)
        treatment_rule_based, drug_rule_based = calculate_treatment_score(treatment) if treatment is not None else (None, None)
        side_effects_rule_based = calculate_side_effects(side_effects) if side_effects is not None else None

        rule_based_predictions.append({
            "treatment": treatment_rule_based,
            "drug": drug_rule_based,
            "side_effects": side_effects_rule_based
        })

    # Combine ML and Rule-Based Predictions
    combined_predictions = []
    for i in range(len(X_val)):
        combined_predictions.append({
            "treatment": (treatment_preds_ml[i] 
                          if rule_based_predictions[i]["treatment"] is None or 
                          treatment_preds_ml[i] == rule_based_predictions[i]["treatment"]
                          else rule_based_predictions[i]["treatment"]),
            "drug": (drug_preds_ml[i] 
                     if rule_based_predictions[i]["drug"] is None or 
                     drug_preds_ml[i] == rule_based_predictions[i]["drug"]
                     else rule_based_predictions[i]["drug"]),
            "side_effects": (side_effects_preds_ml[i] 
                             if rule_based_predictions[i]["side_effects"] is None or 
                             (side_effects_preds_ml[i] == rule_based_predictions[i]["side_effects"]).all()
                             else rule_based_predictions[i]["side_effects"]),
        })

    return combined_predictions




def evaluate_all_models(csv_path):
    """Load test data, preprocess it, and evaluate all models."""
    print("Loading data...")
    features, treatment_labels, drug_labels, side_effects_labels = load_data(csv_path)

    print("Preprocessing data...")
    features = preprocess_data(features)

    # Split into train, validation, and test sets (consistent splits for all labels)
    X_train, X_temp, y_train_treatment, y_temp_treatment, y_train_drug, y_temp_drug, y_train_side_effects, y_temp_side_effects = train_test_split(
        features, treatment_labels, drug_labels, side_effects_labels, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val_treatment, y_test_treatment, y_val_drug, y_test_drug, y_val_side_effects, y_test_side_effects = train_test_split(
        X_temp, y_temp_treatment, y_temp_drug, y_temp_side_effects, test_size=0.5, random_state=42
    )

    # Load models
    print("Loading models...")
    treatment_model = joblib.load("ml_project/C_project/C_rfc_treatment.joblib")
    drug_model = joblib.load("ml_project/C_project/C_rfc_drug.joblib")
    side_effects_model = joblib.load("ml_project/C_project/C_rfc_side_effects.joblib")

    # Labels for confusion matrix
    treatment_labels_unique = np.unique(y_test_treatment)
    drug_labels_unique = np.unique(y_test_drug)
    side_effects_labels_unique = [f"SideEffect_{i}" for i in range(y_test_side_effects.shape[1])]

    # Evaluate each model
    print("Evaluating RFC Treatment Model...")
    evaluate_model(treatment_model, X_test, y_test_treatment, "Treatment Model", treatment_labels_unique)

    print("Evaluating RFC Drug Model...")
    evaluate_model(drug_model, X_test, y_test_drug, "Drug Model", drug_labels_unique)

    print("Evaluating RFC Side Effects Model...")
    evaluate_model(side_effects_model, X_test, y_test_side_effects, "Side Effects Model", side_effects_labels_unique, is_multi_output=True)

    # Call the predict_treatment function after model evaluation
    combined_predictions = predict_treatment(X_val, treatment_model, drug_model, side_effects_model)

# Example usage
if __name__ == "__main__":
    csv_path = 'ml_project/C_project/C_embeddingc.csv'
    evaluate_all_models(csv_path)

