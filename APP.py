from flask import Flask, render_template, request, flash
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import joblib
import numpy as np
import torch
import random
from transformers import BertTokenizer, BertModel
import pandas as pd
from collections import Counter

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flash messages

# Set environment variable to use CPU only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load models
models = {
    "rfc": {
        "treatment": joblib.load('ml_project/final_codes/treatment_model_rfc.joblib'),
        "drug": joblib.load('ml_project/final_codes/drug_model_rfc.joblib'),
        "side_effects": joblib.load('ml_project/final_codes/side_effects_model_rfc.joblib'),
    },
    "svm": {
        "treatment": joblib.load('ml_project/final_codes/treatment_model_svm.joblib'),
        "drug": joblib.load('ml_project/final_codes/drug_model_svm.joblib'),
        "side_effects": joblib.load('ml_project/final_codes/side_effects_model_svm.joblib'),
    },
    "gbc": {
        "treatment": joblib.load('ml_project/final_codes/treatment_model_gbc.joblib'),
        "drug": joblib.load('ml_project/final_codes/drug_model_gbc.joblib'),
        "side_effects": joblib.load('ml_project/final_codes/side_effects_model_gbc.joblib'),
    },
}

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

# BERT embeddings function (CPU)
def extract_bert_embeddings(data, column_name, tokenizer, model, max_len=128, batch_size=16):
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

# Majority vote function
def majority_vote(predictions):
    return Counter(predictions).most_common(1)[0][0]


# Majority voting for multi-label classification (Side Effects)
def majority_vote_multilabel(predictions):
    predictions = np.array(predictions, dtype=int)  # Ensure numerical data type
    sum_preds = np.sum(predictions, axis=0)  # Sum up 1s for each class
    return (sum_preds >= 2).astype(int)  # At least 2 out of 3 models must agree

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    user_input = request.form.get('clinicalNote')

    # Validate input
    if not user_input or len(user_input.strip()) < 10:
        flash("Invalid input. Please provide a meaningful clinical note.", "error")
        return render_template('index.html')

    # Prepare DataFrame for predictions
    new_data = pd.DataFrame({"Clinical Notes": [user_input]})

    # Extract features using BERT
    new_bert_features = extract_bert_embeddings(new_data, "Clinical Notes", tokenizer, model)

    # Predictions using all models
    treatment_preds = []
    drug_preds = []
    side_effects_preds = []

    for model_type in models:
        treatment_preds.append(models[model_type]["treatment"].predict(new_bert_features)[0])
        drug_preds.append(models[model_type]["drug"].predict(new_bert_features)[0])
        side_effects_preds.append(models[model_type]["side_effects"].predict(new_bert_features)[0])
    
    # Convert side_effects_preds to NumPy array before majority voting
    side_effects_preds = np.array(side_effects_preds, dtype=int)

    # Majority voting
    final_treatment = majority_vote(treatment_preds)
    final_drug = majority_vote(drug_preds)
    final_side_effects = majority_vote_multilabel(side_effects_preds)

    # Side Effects Mapping
    side_effects_columns = [
        "nausea", "fatigue", "neutropenia", "kidney toxicity", "vomiting", "hair loss",
        "low platelet count", "anemia", "diarrhoea", "infection risk", "low wbc count",
        "skin irritation", "loss of appetite", "radiation dermatitis", "radiation burns",
        "esophagitis", "fever", "joint pain", "rash", "immune-related pneumonitis", "cough",
        "thyroid issues", "colitis"
    ]

    # Convert multi-label predictions to side effect names
    final_side_effects_list = [
        side_effects_columns[i] for i in range(len(final_side_effects)) if final_side_effects[i] == 1
    ]
    final_side_effects_list = random.sample(final_side_effects_list, k=min(random.randint(2, 4), len(final_side_effects_list))) if final_side_effects_list else ["None"]

    # Format results for display
    result = {
        "Treatment Prediction": final_treatment,
        "Drug Prediction": final_drug,
        "Side Effects": final_side_effects_list
    }

    return render_template('index.html', clinicalnote=result)

if __name__ == '__main__':
    app.run(debug=True)