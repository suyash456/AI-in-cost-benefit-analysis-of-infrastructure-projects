# cost_benefit_ai_app/app.py

import os
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, url_for

app = Flask(__name__)

# Load model and scaler
model = joblib.load('pythonProject/models/xgb_classifier.pkl')
scaler = joblib.load('pythonProject/models/scaler.pkl')

# Feature names
feature_names = [
    'Certificates', 'Years of Experience', 'age', 'Time Arrival Strafe',
    'Project Cost', 'Project Proximity', 'Violation Risk Index',
    'Company PCAB Score', 'Weekly Overtime Hours', 'Salary Bracket'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        fields = ['certificates', 'experience', 'age', 'arrival_strafe',
                  'project_cost', 'project_proximity', 'risk_index',
                  'pcab_score', 'overtime_hours', 'salary_bracket']

        values = []
        for field in fields:
            val = float(request.form[field])
            if val < 0:
                return f"Invalid input: {field} must be non-negative."
            values.append(val)

        # Predict
        X_scaled = scaler.transform([values])
        proba = model.predict_proba(X_scaled)[0][1]
        prediction = int(proba >= 0.5)
        score = int(proba * 100)

        # Save feature importance
        os.makedirs("static", exist_ok=True)
        importances = model.feature_importances_
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances, y=feature_names)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("static/feature_importance.png")
        plt.close()

        label = "Good Project ✅" if prediction == 1 else "Not a Good Project ❌"
        return render_template('result.html', prediction=label, score=score)

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=False)
