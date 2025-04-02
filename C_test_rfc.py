import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
import joblib
import numpy as np
import torch
import warnings
import random
import pandas as pd
from transformers import BertTokenizer, BertModel
from prettytable import PrettyTable

# Suppress warnings (optional)
warnings.filterwarnings("ignore", category=UserWarning)

# Load the trained models
treatment_model = joblib.load('ml_project/C_project/C_rfc_treatment.joblib')
drug_model = joblib.load('ml_project/C_project/C_rfc_drug.joblib')
side_effects_model = joblib.load('ml_project/C_project/C_rfc_side_effects.joblib')

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

def validate_clinical_notes(text):
    """Validates if at least half of the required medical details are present."""
    patterns = {
        "age and gender": r'\b\d{1,3}-?year-?old\s+(Male|Female|Other)\b',
        "smoking and alcohol history": r'\b(?:Never Smoked|Former Smoker|Current Smoker|Smoked|no Smoking|Smoking history).*?(?:None|Light|Moderate|Heavy|no alcohol consumption|alcohol consumption)\b',
        "symptoms": r'\b(?:Symptoms include|Symptoms included|presented with|Symptoms:|experienced).+\b',
        "diagnosis": r'\b(?:Diagnosed with|Following a diagnosis of) .+\b',
        "comorbidities": r'\b(?:Comorbidities include|Other diseases include) .+\b',
        "allergies": r'\b(?:Allergies: .+|is Allergic to .+|No known allergies)\b'
    }

    total_fields = len(patterns)
    present_fields = sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in patterns.values())

    # Proceed if at least 50% of details are given
    if present_fields >= total_fields // 2:
        return "✅ Sufficient clinical details provided."
    
    return "⚠️ Insufficient details. Please include more information for better predictions."


def extract_bert_embeddings(data, column_name, tokenizer, model, max_len=128, batch_size=16):
    """Extracts BERT embeddings from clinical notes."""
    features = []
    for i in range(0, len(data), batch_size):
        batch_notes = data[column_name][i:i + batch_size].tolist()
        inputs = tokenizer(
            batch_notes,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # CLS token embeddings
        features.append(cls_embeddings)
    return np.vstack(features)

while True:
    # Prompt user for input
    user_input = input("\nPlease enter the clinical notes: ")

    # Validate input format
    validation_result = validate_clinical_notes(user_input)
    if "❌" in validation_result:
        print(validation_result)
        continue  # Ask for input again if invalid

    print("✅ Input accepted. Processing...\n")

    # Create a DataFrame for feature extraction
    new_data = pd.DataFrame({"Clinical Notes": [user_input]})

    # Extract features using BERT
    new_bert_features = extract_bert_embeddings(new_data, "Clinical Notes", tokenizer, model)

    # Make predictions
    treatment_predictions = treatment_model.predict(new_bert_features)
    drug_predictions = drug_model.predict(new_bert_features)
    side_effects_predictions = side_effects_model.predict(new_bert_features)

    # Define side effect labels
    side_effects_columns = ["nausea", "fatigue", "neutropenia", "kidney toxicity", "vomiting", "hair loss", 
                            "low platelet count", "anemia", "diarrhoea", "infection risk", "low wbc count", 
                            "skin irritation", "loss of appetite", "radiation dermatitis", "radiation burns", 
                            "esophagitis", "fever", "joint pain", "rash", "immune-related pneumonitis", "cough", 
                            "thyroid issues", "colitis"]

    side_effects_results = []
    for i in range(side_effects_predictions.shape[0]):
        side_effects = [side_effects_columns[j] for j in range(side_effects_predictions.shape[1]) if side_effects_predictions[i][j] == 1]
        
        # Randomly select 2-4 side effects if any are predicted
        selected_side_effects = random.sample(side_effects, k=min(random.randint(2, 4), len(side_effects))) if side_effects else ["none"]
        side_effects_results.append(selected_side_effects)    

    # Combine predictions into a DataFrame
    predictions_df = pd.DataFrame({
        "Treatment Type": treatment_predictions,
        "Drug": drug_predictions,
        "Predicted Side Effects": side_effects_results
    })

    # Display results in a structured format
    table = PrettyTable()
    table.field_names = ["Treatment Type", "Drug", "Predicted Side Effects"]

    for _, row in predictions_df.iterrows():
        table.add_row([row["Treatment Type"], row["Drug"], ", ".join(row["Predicted Side Effects"])])

    print("\nPredictions:\n")
    print(table)
    break  # Exit after a valid input is processed
