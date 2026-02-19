from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'medical-prediction-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    smoking = db.Column(db.Boolean, default=False)
    diabetes = db.Column(db.Boolean, default=False)
    bp = db.Column(db.Boolean, default=False)
    symptoms = db.relationship('Symptom', backref='user', lazy=True)

class Symptom(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    duration = db.Column(db.Integer)  # in days
    severity = db.Column(db.Integer)  # 1-10
    status = db.Column(db.String(20))  # Active / Resolved
    date_occurrence = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load ML Model Assets (trained on real Disease & Symptoms dataset)
try:
    model          = joblib.load('model.pkl')           # RandomForestClassifier
    le             = joblib.load('label_encoder.pkl')   # Disease LabelEncoder
    SYMPTOMS_LIST  = joblib.load('symptoms_list.pkl')   # All 377 model feature cols
    UI_SYMPTOMS    = joblib.load('ui_symptoms.pkl')     # Top 50 for dashboard dropdown
except FileNotFoundError:
    model = None
    le = None
    SYMPTOMS_LIST = []
    UI_SYMPTOMS   = []
    print('[WARNING] model artifacts not found — run: python train_model.py')

# Temporal Analysis Helper
def analyze_temporal_patterns(user_id):
    symptoms = Symptom.query.filter_by(user_id=user_id, status='Active').all()
    if not symptoms:
        return None
    
    # Calculate recurrence and total duration
    symptom_counts = {}
    total_duration = 0
    max_severity = 0
    
    for s in symptoms:
        symptom_counts[s.name.lower()] = symptom_counts.get(s.name.lower(), 0) + 1
        total_duration += s.duration
        if s.severity > max_severity:
            max_severity = s.severity
            
    # Pattern Logic
    is_recurring = any(count > 1 for count in symptom_counts.values())
    is_chronic = total_duration > 30
    
    pattern = "Acute"
    if is_chronic:
        pattern = "Chronic Risk"
    elif is_recurring:
        pattern = "Recurring"
        
    return {
        "pattern": pattern,
        "total_duration": total_duration,
        "max_severity": max_severity,
        "symptom_counts": symptom_counts
    }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        smoking = 'smoking' in request.form
        diabetes = 'diabetes' in request.form
        bp = 'bp' in request.form
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
            
        new_user = User(
            username=username,
            password=generate_password_hash(password),
            name=name,
            age=int(age),
            gender=gender,
            smoking=smoking,
            diabetes=diabetes,
            bp=bp
        )
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    symptoms = Symptom.query.filter_by(user_id=current_user.id).order_by(Symptom.date_occurrence.desc()).all()
    # Use the UI-friendly top-50 list for the dropdown; fall back to full list
    dropdown_symptoms = UI_SYMPTOMS if UI_SYMPTOMS else SYMPTOMS_LIST
    return render_template('dashboard.html', symptoms=symptoms, symptoms_list=dropdown_symptoms)

@app.route('/add_symptom', methods=['POST'])
@login_required
def add_symptom():
    name = request.form.get('symptom_name')
    custom_name = request.form.get('custom_symptom')
    if name == 'custom':
        name = custom_name
    
    duration = request.form.get('duration')
    severity = request.form.get('severity')
    status = request.form.get('status')
    
    new_symptom = Symptom(
        user_id = current_user.id,
        name = name,
        duration = int(duration),
        severity = int(severity),
        status = status
    )
    db.session.add(new_symptom)
    db.session.commit()
    if status == 'Resolved':
        flash('Resolved symptom saved. You can delete it from Logged History.', 'info')
    else:
        flash('Active symptom added successfully')
    return redirect(url_for('dashboard'))

@app.route('/analyze')
@login_required
def analyze():
    if model is None:
        flash('Prediction model not loaded. Run: python train_model.py', 'danger')
        return redirect(url_for('dashboard'))

    analysis = analyze_temporal_patterns(current_user.id)
    if not analysis:
        flash('Please add at least one Active symptom before running analysis.', 'warning')
        return redirect(url_for('dashboard'))

    # ── Build feature vector matching the real dataset's 377 symptom columns ──
    # Start with all zeros (one per model feature column)
    symptom_flags = {col: 0 for col in SYMPTOMS_LIST}

    # Map user's logged symptoms to the exact column names used during training
    # Column names in the real dataset use underscores and lowercase
    active_symptoms = Symptom.query.filter_by(user_id=current_user.id, status='Active').all()
    for s in active_symptoms:
        # Normalise the entered name to match dataset column naming
        col_name = (s.name.strip().lower()
                    .replace(' ', '_')
                    .replace('-', '_'))
        if col_name in symptom_flags:
            symptom_flags[col_name] = 1

    # Build DataFrame in the exact column order the model was trained on
    df_features = pd.DataFrame([symptom_flags])[SYMPTOMS_LIST]

    # ── Prediction ────────────────────────────────────────────────────────────
    probs = model.predict_proba(df_features)[0]
    top_indices = np.argsort(probs)[-3:][::-1]
    predictions = []
    for idx in top_indices:
        disease_name = le.inverse_transform([idx])[0]
        predictions.append({
            'disease': disease_name.title(),
            'probability': round(float(probs[idx]) * 100, 2)
        })

    # ── Risk Score (clamped 0–100 for SVG gauge) ──────────────────────────────
    smoking_factor = 10 if current_user.smoking else 0
    risk_score = min(100,
                     (analysis['max_severity'] * 5) +
                     (analysis['total_duration'] * 2) +
                     smoking_factor)
    if risk_score > 60:
        risk_level = "High"
    elif risk_score > 30:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # ── Specialist Recommendation ─────────────────────────────────────────────
    top_disease = predictions[0]['disease'].lower() if predictions else ''
    specialist_map = {
        'covid': 'Pulmonologist / Infectious Disease Specialist',
        'pneumonia': 'Pulmonologist',
        'bronchitis': 'Pulmonologist',
        'asthma': 'Pulmonologist',
        'migraine': 'Neurologist',
        'dengue': 'Infectious Disease Specialist',
        'malaria': 'Infectious Disease Specialist',
        'tuberculosis': 'Pulmonologist',
        'typhoid': 'Infectious Disease Specialist',
        'gastro': 'Gastroenterologist',
        'diabetes': 'Endocrinologist',
        'hypertension': 'Cardiologist',
        'heart': 'Cardiologist',
        'skin': 'Dermatologist',
        'allergy': 'Allergist / Immunologist',
        'anxiety': 'Psychiatrist',
        'depression': 'Psychiatrist',
    }
    specialist = 'General Physician'
    for keyword, spec in specialist_map.items():
        if keyword in top_disease:
            specialist = spec
            break

    return render_template('result.html',
                           analysis=analysis,
                           predictions=predictions,
                           risk_level=risk_level,
                           risk_score=risk_score,
                           specialist=specialist)

@app.route('/delete_symptom/<int:symptom_id>', methods=['POST'])
@login_required
def delete_symptom(symptom_id):
    symptom = Symptom.query.get(symptom_id)
    if symptom and symptom.user_id == current_user.id:
        db.session.delete(symptom)
        db.session.commit()
        flash('Symptom deleted successfully')
    return redirect(url_for('dashboard'))

@app.route('/api/severity_history')
@login_required
def severity_history():
    symptoms = Symptom.query.filter_by(user_id=current_user.id, status='Active').order_by(Symptom.date_occurrence.asc()).all()
    data = {
        'labels': [s.date_occurrence.strftime('%Y-%m-%d') for s in symptoms],
        'values': [s.severity for s in symptoms]
    }
    return jsonify(data)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
