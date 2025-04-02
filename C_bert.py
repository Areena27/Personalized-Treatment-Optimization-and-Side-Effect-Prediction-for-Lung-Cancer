import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import re
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA

# Predefined mapping of treatments to drugs
treatment_to_drugs = {
    "chemotherapy": ["cisplatin", "carboplatin", "etoposide"],
    "radiotherapy": ["standard radiotherapy", "high-dose radiotherapy"],
    "immunotherapy": ["pembrolizumab", "nivolumab"]
}

# Load and preprocess data
def load_and_preprocess_data(input_file):
    data = pd.read_csv(input_file)
    data['Clinical Notes'] = data['Clinical Notes'].str.lower().str.strip()  # Lowercase and strip spaces
    return data

# Extract embeddings using BERT
def extract_bert_embeddings(data, column_name, tokenizer, model, max_len=128, batch_size=16):
    features = []
    print("Extracting BERT embeddings...")
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
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
            # CLS token embeddings
            cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()   # CLS token embeddings
        features.append(cls_embeddings)
    return np.vstack(features)

def detect_treatment(text):
    """Detect treatment type from text using regex."""
    text = text.lower()
    if re.search(r'\bchemo(?:therapy)?\b', text):
        return "chemotherapy"
    elif re.search(r'\bradiotherapy\b', text):
        return "radiotherapy"
    elif re.search(r'\bimmunotherapy\b', text):
        return "immunotherapy"
    return "unknown"


def detect_drug(text, treatment_type):
    """Detect drug name based on treatment type."""
    if treatment_type in treatment_to_drugs:
        for drug in treatment_to_drugs[treatment_type]:
            if re.search(r'\b' + re.escape(drug) + r'\b', text, re.IGNORECASE):
                return drug
        return treatment_to_drugs[treatment_type][0]  # Default to first drug
    return "N/A"

# Extract side effects using BERT's attention mechanism
def extract_side_effects_with_attention(text, tokenizer, model):
    """
    Uses BERT's self-attention mechanism to extract potential side effects from clinical notes.
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
    
    # Get attention weights (check if attentions are available)
    with torch.no_grad():
        outputs = model(**inputs)
        if hasattr(outputs, "attentions"):
            attention_weights = outputs.attentions  # Get attention weights from BERT's self-attention layers
            last_layer_attention = attention_weights[-1]  # Extract the last layer's attention
        else:
            print("No attention weights found!")
            return []
    
    # Map side effect keywords to token IDs
    side_effect_keywords = ["nausea", "fatigue", "neutropenia", "kidney toxicity", "vomiting",
                            "hair loss", "low platelet count", "anemia", "diarrhoea", "infection Risk",
                            "low wbc count", "skin irritation", "loss of appetite", "radiation dermatitis",
                            "radiation burns", "esophagitis", "fever", "joint pain", "rash",
                            "immune-related pneumonitis", "cough", "thyroid issues", "colitis"]
    
    # Convert keywords to token ids
    keyword_token_ids = [tokenizer.encode(word, add_special_tokens=False) for word in side_effect_keywords]
    attention_scores = {}  # Dictionary to store attention scores for each keyword

    # For each keyword, check if its token ID is in the input sequence
    for keyword, token_ids in zip(side_effect_keywords, keyword_token_ids):
        for token_id in token_ids:
            token_positions = (inputs.input_ids[0] == token_id).nonzero(as_tuple=True)[0]
            for pos in token_positions:
                attention_scores[keyword] = max(attention_scores.get(keyword, 0), torch.max(last_layer_attention[0, :, pos]).item())
    
    # Extract potential side effects based on maximum attention scores
    potential_side_effects = [word for word, score in attention_scores.items() if score > 0.5]  # Threshold on attention score
    return potential_side_effects if potential_side_effects else ["none"]

# Dimensionality reduction
def reduce_dimensionality(features, n_components=64):
    print("Reducing dimensionality...")
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Explained Variance Retained: {explained_variance:.2f}")
    return reduced_features

# Main function
if __name__ == "__main__":
    # File paths
    input_file = "ml_project/C_project/C_cleaned.csv"
    output_file = "ml_project/C_project/C_embeddingc.csv"

    # Load data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(input_file)
    print(f"Data loaded successfully with {len(data)} records.")

    # Initialize models
    print("Initializing BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

    # Extract BERT embeddings
    bert_features = extract_bert_embeddings(data, "Clinical Notes", tokenizer, model)

    #Dimensionality reduction (optional)
    #use_pca = True
    #if use_pca:
    #    bert_features = reduce_dimensionality(bert_features, n_components=64)

    # Initialize lists to store processed data
    features = bert_features.tolist()

    # Extract treatment and corresponding drug information
    treatment_labels = [detect_treatment(text) for text in data['Clinical Notes']]
    drug_labels = [detect_drug(text, treatment) for text, treatment in zip(data['Clinical Notes'], treatment_labels)]

    # Side effects (multi-label extraction using BERT attention mechanism)
    side_effects_column = ["nausea", "fatigue", "neutropenia", "kidney toxicity","vomiting", "hair loss", "low platelet count", 
                           "anemia", "diarrhoea", "infection Risk", "low wbc count", "skin irritation", "loss of appetite", 
                           "radiation dermatitis", "radiation burns", "esophagitis","fever", "joint pain", "rash", 
                           "immune-related pneumonitis", "cough", "thyroid issues", "colitis"]
    
    side_effects_labels = []
    for text in data["Clinical Notes"]:
        side_effects = extract_side_effects_with_attention(text, tokenizer, model)
        side_effects_labels.append([1 if keyword in side_effects else 0 for keyword in side_effects_column])

    # Create DataFrame for features and labels
    features_df = pd.DataFrame(features)
    labels_df = pd.DataFrame({
        "Treatment_Type": treatment_labels,  # Treatment type (manually extracted or inferred)
        "Drug": drug_labels,
        **{f"SideEffect_{i}": [label[i] for label in side_effects_labels] for i in range(len(side_effects_column))}  # Side effects
    })

    # Combine features and labels
    output_df = pd.concat([features_df, labels_df], axis=1)

    # Save the combined features to CSV
    print(f"Saving extracted features to {output_file}...")
    output_df.to_csv(output_file, index=False)
    print("Feature extraction completed successfully.")
