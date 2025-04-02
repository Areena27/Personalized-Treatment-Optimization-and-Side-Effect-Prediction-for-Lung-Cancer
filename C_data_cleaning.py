import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from contractions import fix

# Download NLTK data files (only needed the first time)
nltk.download('stopwords')
nltk.download('wordnet')

def clean_data(input_file, output_file):
    """
    Cleans and preprocesses clinical notes while preserving multi-word medical terms.
    """
    # Load the dataset
    data = pd.read_csv(input_file)
    
    # Remove duplicates and empty clinical notes
    data.drop_duplicates(inplace=True)
    data.dropna(subset=['Clinical Notes'], inplace=True)

    # Standardize text (lowercase, remove extra spaces)
    data['Clinical Notes'] = data['Clinical Notes'].str.lower().str.strip()

    # Expand contractions
    data['Clinical Notes'] = data['Clinical Notes'].apply(lambda x: fix(x))

    # Preserve important multi-word medical terms BEFORE removing special characters
    medical_phrases = {
        r"\bhigh dose radiotherapy\b": "High-Dose Radiotherapy",
        r"\bstandard radiotherapy\b": "Standard Radiotherapy",
        r"\bnon small cell lung cancer\b": "NSCLC",
        r"\bsmall cell lung cancer\b": "SCLC",
        r"\bchronic obstructive pulmonary disease\b": "COPD",
    }
    for pattern, replacement in medical_phrases.items():
        data['Clinical Notes'] = data['Clinical Notes'].apply(lambda x: re.sub(pattern, replacement, x))

    # Remove special characters except hyphens (to preserve "High-Dose")
    data['Clinical Notes'] = data['Clinical Notes'].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s.,?-]", "", x))

    # Remove extra whitespace
    data['Clinical Notes'] = data['Clinical Notes'].apply(lambda x: re.sub(r"\s+", " ", x).strip())

    # Standardize common medical terms (again after cleaning)
    medical_terms = {
        r"\bnsclc\b": "NSCLC",
        r"\bsclc\b": "SCLC",
        r"\bcopd\b": "COPD",
        r"\bdiabetes\b": "Diabetes",
        r"\bhypertension\b": "Hypertension",
    }
    for pattern, replacement in medical_terms.items():
        data['Clinical Notes'] = data['Clinical Notes'].apply(lambda x: re.sub(pattern, replacement, x))

    # Handle missing or placeholder values
    data['Clinical Notes'] = data['Clinical Notes'].apply(lambda x: re.sub(r"\bnone\b", "None", x))
    data['Clinical Notes'] = data['Clinical Notes'].apply(lambda x: re.sub(r"\bna\b", "NA", x))

    # Tokenization
    data['Clinical Notes'] = data['Clinical Notes'].apply(lambda x: x.split())

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    data['Clinical Notes'] = data['Clinical Notes'].apply(lambda x: [word for word in x if word not in stop_words])

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    data['Clinical Notes'] = data['Clinical Notes'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # Join tokens back to a string
    data['Clinical Notes'] = data['Clinical Notes'].apply(lambda x: ' '.join(x))

    # Save the cleaned data
    data.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

# Example usage
input_file = 'ml_project/C_project/C_clinical_notes.csv'
output_file = 'ml_project/C_project/C_cleaned.csv'
clean_data(input_file, output_file)
