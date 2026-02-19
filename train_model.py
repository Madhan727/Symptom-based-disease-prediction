import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Expanded Symptoms and Diseases to match a real "Disease & Symptom" dataset structure
SYMPTOMS = [
    'fever', 'cough', 'fatigue', 'headache', 'shortness_of_breath', 
    'sore_throat', 'body_ache', 'nausea', 'vomiting', 'diarrhea',
    'chest_pain', 'loss_of_taste', 'loss_of_smell', 'skin_rash',
    'joint_pain', 'sneezing', 'runny_nose', 'chills', 'dizziness'
]

DISEASE_PROFILES = {
    'Common Cold': ['cough', 'sneezing', 'runny_nose', 'sore_throat'],
    'Flu': ['fever', 'cough', 'fatigue', 'body_ache', 'chills'],
    'COVID-19': ['fever', 'cough', 'fatigue', 'shortness_of_breath', 'loss_of_taste', 'loss_of_smell'],
    'Pneumonia': ['fever', 'cough', 'shortness_of_breath', 'chest_pain', 'fatigue'],
    'Bronchitis': ['cough', 'fatigue', 'shortness_of_breath', 'chest_pain'],
    'Migraine': ['headache', 'nausea', 'dizziness', 'vomiting'],
    'Dengue': ['fever', 'headache', 'body_ache', 'joint_pain', 'skin_rash'],
    'Gastroenteritis': ['nausea', 'vomiting', 'diarrhea', 'fatigue'],
    'Asthma Flare-up': ['shortness_of_breath', 'cough', 'chest_pain']
}

def generate_robust_data(num_samples=5000):
    data = []
    diseases = list(DISEASE_PROFILES.keys())
    
    for _ in range(num_samples):
        # User Demographics & Lifestyle
        age = np.random.randint(1, 95)
        smoking = np.random.choice([0, 1], p=[0.8, 0.2])
        diabetes = np.random.choice([0, 1], p=[0.85, 0.15])
        bp = np.random.choice([0, 1], p=[0.7, 0.3])
        
        disease = np.random.choice(diseases)
        core_symptoms = DISEASE_PROFILES[disease]
        
        symptom_flags = {s: 0 for s in SYMPTOMS}
        
        # Assign core symptoms with high probability
        for s in core_symptoms:
            if np.random.random() < 0.9:
                symptom_flags[s] = 1
        
        # Assign random symptoms with low probability (noise)
        for s in SYMPTOMS:
            if s not in core_symptoms:
                if np.random.random() < 0.05:
                    symptom_flags[s] = 1
        
        # Temporal & Severity Logic
        if disease in ['Common Cold', 'Flu', 'Gastroenteritis']:
            avg_severity = np.random.randint(2, 7)
            avg_duration = np.random.randint(3, 10)
            recurrence_count = 0
        elif disease in ['Migraine', 'Asthma Flare-up']:
            avg_severity = np.random.randint(6, 10)
            avg_duration = np.random.randint(1, 4)
            recurrence_count = np.random.randint(1, 8)
        else: # Serious conditions
            avg_severity = np.random.randint(7, 10)
            avg_duration = np.random.randint(10, 40)
            recurrence_count = 0
            
        row = {
            'age': age,
            'smoking': smoking,
            'diabetes': diabetes,
            'bp': bp,
            'avg_severity': avg_severity,
            'avg_duration': avg_duration,
            'recurrence_count': recurrence_count,
            'disease': disease
        }
        row.update(symptom_flags)
        data.append(row)
    
    return pd.DataFrame(data)

def train_and_save_model():
    print("Generating robust Disease & Symptom dataset...")
    df = generate_robust_data()
    
    X = df.drop('disease', axis=1)
    y = df['disease']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    print(f"Training RandomForestClassifier on {len(X)} samples...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    joblib.dump(clf, 'model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(SYMPTOMS, 'symptoms_list.pkl')
    
    print("Robust model and assets saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
