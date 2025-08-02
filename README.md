# AI in Cost-Benefit Analysis of Infrastructure Projects

## Overview
This project is a **Machine Learning-powered web application** that evaluates infrastructure projects based on various features such as cost, experience, risk index, and more.  
It predicts whether a project is **Good** or **Not Good** and provides a **Cost-Benefit Score** with a visual representation of **feature importance**.

The application is implemented using:
- **XGBoost** for prediction
- **Flask** for backend
- **Bootstrap** for frontend
- **Matplotlib & Seaborn** for data visualization

---

## Features
- Accepts project details via a user-friendly web form
- Predicts if a project is **Good** or **Not Good**
- Displays a **Cost-Benefit Score** (0–100)
- Progress bar visualization for score
- Feature importance chart for model interpretability
- Mobile-responsive design

---

## 📂 Project Structure

```text
cost_benefit_ai_project/
│
├── App.py                          # Flask backend
├── models/
│   ├── xgb_classifier.pkl          # Trained XGBoost model
│   ├── scaler.pkl                   # StandardScaler
│   └── train_model.py               # Script to train the model
│
├── static/
│   ├── feature_importance.png       # Auto-generated feature importance plot
│
├── templates/
│   ├── index.html                   # Input form page
│   └── result.html                  # Prediction result page
│
├── data/
│   └── Civil Engineering Global Project Dataset.csv  # Dataset used
│
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation




