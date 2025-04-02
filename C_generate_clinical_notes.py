import pandas as pd
import random
from faker import Faker
from datetime import datetime

fake = Faker("en_GB")

# Constants for logical rules
stage_probabilities = {"Stage I": 0.3, "Stage II": 0.2, "Stage III": 0.2, "Stage IV": 0.3}
treatment_weights = {"Chemotherapy": 0.33, "Radiotherapy": 0.37, "Immunotherapy": 0.30}
quality_of_life_scores = {"Excellent": 0.1, "Good": 0.3, "Fair": 0.4, "Poor": 0.2}

# Drugs and treatments
drugs_treatments = {
    "Chemotherapy": ["Cisplatin", "Carboplatin", "Etoposide"],
    "Radiotherapy": ["Standard Radiotherapy", "High-Dose Radiotherapy"],
    "Immunotherapy": ["Pembrolizumab", "Nivolumab"],
}

side_effects_map = {
    "Cisplatin": ["Nausea", "Fatigue", "Neutropenia", "Kidney Toxicity"],
    "Carboplatin": ["Vomiting", "Hair Loss", "Low Platelet Count", "Anemia"],
    "Etoposide": [ "Diarrhoea", "Infection Risk", "Low WBC Count"],
    "Standard Radiotherapy": ["Skin Irritation", "Fatigue", "Loss of Appetite", "Radiation Dermatitis"],
    "High-Dose Radiotherapy": ["Nausea", "Fatigue", "Radiation Burns", "Esophagitis"],
    "Pembrolizumab": ["Fever", "Joint Pain", "Rash", "Immune-Related Pneumonitis"],
    "Nivolumab": ["Fatigue", "Cough", "Thyroid Issues", "Colitis"],
}

drug_dosage_ranges = {
    "Cisplatin": (50, 100),  # mg/m² per cycle
    "Carboplatin": (200, 450),  # mg (calculated using AUC)
    "Etoposide": (50, 100),  # mg/m²/day
    "Standard Radiotherapy": (1, 10),  # Gy/session
    "High-Dose Radiotherapy": (10, 20),  # Gy/session
    "Pembrolizumab": (200, 400),  # mg every 3 weeks
    "Nivolumab": (240, 480),  # mg every 2 weeks
}

drug_frequency_options = {
    "Cisplatin": ["Every 3 weeks", "Every 4 weeks"],
    "Carboplatin": ["Every 21 days", "Every 28 days"],
    "Etoposide": ["Daily for 5 days", "Daily for 3 days"],
    "Standard Radiotherapy": ["5 sessions/week for 6 weeks"],
    "High-Dose Radiotherapy": ["1 session/week for 4 weeks"],
    "Pembrolizumab": ["Every 3 weeks"],
    "Nivolumab": ["Every 2 weeks"],
}

# Response prediction weights
response_weights = {
    "Complete Response": 0.2,
    "Partial Response": 0.3,
    "Stable Disease": 0.25,
    "Progressive Disease": 0.25
}

def calculate_treatment_score(stage, diagnosis, comorbidities, age):
    stage_weights = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}
    comorbidity_penalty = len(comorbidities) * 0.5
    age_penalty = 0.1 * max(0, age - 60)

    if stage == "Stage I" and diagnosis == "NSCLC":
        selected_treatment = "Radiotherapy"
    elif stage in ["Stage II", "Stage I"]:
        selected_treatment = "Chemotherapy"
    elif stage == "Stage III":
        selected_treatment = random.choices(
        ["Immunotherapy", "Chemotherapy"], weights=[0.6, 0.4])[0]
    elif stage == "Stage IV":
        selected_treatment = random.choices(["Immunotherapy", "Chemotherapy"], weights=[0.7, 0.3])[0]
    else:
        # Calculate scores
        chemo_score = stage_weights[stage] + (2 if "Peripheral Vascular Disease" not in comorbidities else -3) - age_penalty
        immuno_score = stage_weights[stage] * 0.9 - comorbidity_penalty
        radio_score = stage_weights[stage] * 0.8 - (1 if "COPD" in comorbidities else 0)

        #scores = {"Chemotherapy": chemo_score, "Radiotherapy": radio_score, "Immunotherapy": immuno_score}
        scores = {
            "Chemotherapy": chemo_score * treatment_weights["Chemotherapy"],
            "Radiotherapy": radio_score * treatment_weights["Radiotherapy"],
            "Immunotherapy": immuno_score * treatment_weights["Immunotherapy"],
        }

        selected_treatment = max(scores, key=scores.get)

    drug = random.choice(drugs_treatments[selected_treatment])

    return selected_treatment, drug


def calculate_dosage(base_dosage, age, comorbidities_count):
    #Calculate drug dosage based on age and comorbidities.
    return int(base_dosage * (1 - age / 100) * (1 - 0.2 * comorbidities_count))

#def calculate_side_effects(drug, comorbidities_count):
    #max_dosage = max(drug_dosage_ranges[drug])
    #side_effect_prob = dosage / max_dosage * (1 + 0.1 * comorbidities_count)
    #side_effect_prob =  0.2 * comorbidities_count
    #return random.sample(side_effects_map[drug],k=min(random.randint(1, 3),int(len(side_effects_map[drug]) * side_effect_prob)))
    #return random.sample(side_effects_map[drug],  k=random.randint(1, 3))

def calculate_side_effects(drug, comorbidities_count):
    max_effects = min(4, len(side_effects_map[drug]))
    min_effects = max(2, int(comorbidities_count * 0.5))  # Comorbidities influence the lower bound
    num_effects = random.randint(min_effects, max_effects)
    return random.sample(side_effects_map[drug], k=num_effects)

def calculate_treatment_outcome(stage, comorbidities_count):
    stage_severity = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}[stage]
    complete_response_prob = 1 / (1 + 2.718 ** (-(0.5 * stage_severity - 0.2 * comorbidities_count)))
    response_options = ["Complete Response", "Partial Response", "Stable Disease", "Progressive Disease"]
    response_weights = [
        complete_response_prob,
        (1 - complete_response_prob) * 0.5,
        (1 - complete_response_prob) * 0.3,
        (1 - complete_response_prob) * 0.2,
    ]
    return random.choices(response_options, weights=response_weights, k=1)[0]

def generate_timeline():
    diagnosis_year = random.randint(2010, 2024)
    treatment_year = random.randint(diagnosis_year + 1, min(2025, datetime.now().year)) 
    follow_up_interval = random.randint(2, 6) 

    timeline = f"Patient was diagnosed on {diagnosis_year} and began treatment in {treatment_year}. Follow-up every {follow_up_interval} months."
    return timeline

# Narrative templates
narrative_templates = [
    (
        "The patient, a {age}-year-old {gender}, is a {smoking_status} and consumes alcohol ({alcohol_consumption}). "
        "They have a family history of cancer: {family_history}. Symptoms reported include {symptoms} and is allergic to {allergies}. "
        "Diagnosed with {diagnosis} ({stage}) having {comorbidities}. Underwent {treatment_type} with {drug} ({dosage} mg, {frequency}) for {duration}. "
        "Observed side effects: {side_effects}. Treatment response was {response}. Currently {survival_status}. "
        "Quality of life: {quality_of_life}. Timeline: {timeline}."
    ),
    (
        "{treatment_type} was initiated using {drug} at a dosage of {dosage} mg ({frequency}) for {duration}. "
        "This was following a diagnosis of {diagnosis} ({stage}). Symptoms included {symptoms}. "
        "The patient, a {age}-year-old {gender}, is a {smoking_status} with {alcohol_consumption} alcohol consumption. "
        "Family history: {family_history} and is allergic to {allergies}. Side effects noted were {side_effects} and other diseases include {comorbidities}. Response to treatment: {response}. "
        "Current status: {survival_status}. Quality of life: {quality_of_life}. Timeline: {timeline}."
    ),
    (
        "Symptoms such as {symptoms} were reported by a {age}-year-old {gender} who is a {smoking_status} and consumes "
        "alcohol ({alcohol_consumption}). Family history: {family_history},also has {comorbidities} and is allergic to {allergies}. They were diagnosed with {diagnosis} ({stage}). "
        "Prescribed {drug} as part of {treatment_type} ({dosage} mg, {frequency}) for {duration}. "
        "Side effects observed: {side_effects}. Treatment response: {response}. Status: {survival_status}. "
        "Quality of life: {quality_of_life}. Timeline: {timeline}."
    ),
     (
        "A {age}-year-old {gender}, {smoking_status} with {alcohol_consumption} alcohol consumption, presented with {symptoms}."
        "Family history includes {family_history}. Diagnosed with {diagnosis} ({stage}). Comorbidities include {comorbidities}."
        "Allergies: {allergies}. {treatment_type} was initiated with {drug} ({dosage} mg, {frequency}) for {duration}."
        "Observed side effects: {side_effects}. Treatment response: {response}. Current status: {survival_status}." 
        "Quality of life: {quality_of_life}. Timeline: {timeline}."
    ),
    (
        "{symptoms} were reported by a {age}-year-old {gender}, a {smoking_status} with {alcohol_consumption} alcohol consumption." 
        "Family history: {family_history}. Diagnosed with {diagnosis} ({stage}). Comorbidities include {comorbidities}."
        "Allergies: {allergies}. {treatment_type} was administered with {drug} ({dosage} mg, {frequency}) for {duration}." 
        "Side effects: {side_effects}. Treatment response: {response}. Current status: {survival_status}." 
        "Quality of life: {quality_of_life}. Timeline: {timeline}."
    ),
    (
        "This {age}-year-old {gender}, {smoking_status} with {alcohol_consumption} alcohol consumption, has a family history of {family_history}." 
        "They were diagnosed with {diagnosis} ({stage}) and presented with {symptoms}. Comorbidities include {comorbidities}." 
        "Allergies: {allergies}. {treatment_type} with {drug} ({dosage} mg, {frequency}) was administered for {duration}." 
        "Side effects: {side_effects}. Treatment response: {response}. Current status: {survival_status}." 
        "Quality of life: {quality_of_life}. Timeline: {timeline}."
    )
]

# Generate clinical data with logical dependencies
def generate_clinical_data(num_entries):
    data = []
    comorbidities_list = ["Diabetes", "Hypertension", "Peripheral Vascular Disease", "COPD", "Cognitive Heart Failure", "Renal Disease", "Asthma"]
    
    for _ in range(num_entries):
        # Demographics
        age = random.randint(20, 85)
        gender = random.choice(["Male", "Female", "Other"])
        smoking_status = random.choice(["Current Smoker", "Former Smoker", "Never Smoked"])
        alcohol_consumption = random.choice(["None", "Light", "Moderate", "Heavy"])
        family_history = random.choice(["Yes", "No"])
        allergies = random.sample(["Penicillin", "Dust", "Sulfa Drugs", "Peanuts", "Seafood"], k=random.randint(0, 2))
        #comorbidities = random.sample(comorbidities_list, k=random.randint(1, 3))
        comorbidities = random.sample(comorbidities_list, k=min(len(comorbidities_list), random.randint(1, 3)))

        # Symptoms
        symptoms = random.sample(
            ["Coughing", "Shortness of Breath", "Chest Pain", "Fatigue", "Swallowing Difficulty", "Weight Loss"],
            k=random.randint(1, 3)
        )

        # Diagnosis selection (NSCLC or SCLC)
        diagnosis = "NSCLC" if random.random() > 0.2 else "SCLC"
        
        # Stage influenced by smoking
        stage = random.choices(
            ["Stage I", "Stage II", "Stage III", "Stage IV"],
            weights=[1.0, 1.0, 1.2, 1.5] if smoking_status != "Never Smoked" else [0.8, 0.6, 0.6, 0.7],
            k=1
        )[0]

        # Treatment and drug selection
        treatment_type, drug = calculate_treatment_score(stage, diagnosis, comorbidities, age)
        dosage = calculate_dosage(random.randint(*drug_dosage_ranges[drug]), age, len(comorbidities))
        side_effects = calculate_side_effects(drug, len(comorbidities))
        response = calculate_treatment_outcome(stage, len(comorbidities))
        survival_status = random.choice(["Alive", "Deceased"])
        quality_of_life = random.choices(
            ["Excellent", "Good", "Fair", "Poor"], weights=[0.1, 0.3, 0.4, 0.2], k=1
        )[0]
        duration = random.choice([3, 6, 12])  # Random treatment duration
        timeline = generate_timeline()

        # Construct the narrative
        narrative = random.choice(narrative_templates).format(
            age=age, gender=gender, smoking_status=smoking_status, alcohol_consumption=alcohol_consumption,
            family_history=family_history, symptoms=", ".join(symptoms), allergies=", ".join(allergies), 
            comorbidities=", ".join(comorbidities),diagnosis=diagnosis, stage=stage, treatment_type=treatment_type, drug=drug,
            dosage=dosage, frequency=random.choice(drug_frequency_options[drug]),duration=duration,
            side_effects=", ".join(side_effects), response=response, survival_status=survival_status,
            quality_of_life=quality_of_life, timeline=timeline
        )
        data.append(narrative)
    
    return data

# Generate  records and save to CSV
sample_data = generate_clinical_data(11000)

# Save to CSV file
df = pd.DataFrame(sample_data, columns=["Clinical Notes"])
df.to_csv('ml_project/C_project/C_clinical_notes.csv', index=False)
print(f"Data saved to {'ml_project/C_project/C_clinical_notes.csv'}")

