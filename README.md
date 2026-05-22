# 🧠 Mental Health Risk Assessment System

An AI-powered hybrid mental health risk assessment system that combines **XGBoost**, **CatBoost**, and **Fuzzy Logic** to predict mental health risk levels based on lifestyle, academic, professional, and psychological factors.

The project includes:

* Machine Learning based prediction models
* Fuzzy Logic inference system
* Hybrid AI decision-making
* Interactive Streamlit web application
* Analytics dashboard and visualizations

---

## 📌 Features

* ✅ Mental health risk prediction
* ✅ Hybrid AI model (XGBoost + CatBoost + Fuzzy Logic)
* ✅ Real-time risk assessment
* ✅ Confidence score visualization
* ✅ Interactive Streamlit dashboard
* ✅ Feature importance analysis
* ✅ Data preprocessing and encoding pipeline
* ✅ Hybrid model evaluation metrics

---

# 🏗️ Project Architecture

```text
User Input
     ↓
Data Preprocessing
     ↓
 ┌───────────────┐
 │ XGBoost Model │
 └───────────────┘
     ↓

 ┌───────────────┐
 │ CatBoost Model│
 └───────────────┘
     ↓

 ┌───────────────┐
 │ Fuzzy Logic   │
 └───────────────┘
     ↓

Hybrid Decision Engine
     ↓
Final Mental Health Risk Prediction
```

---

# 📂 Project Structure

```text
├── app.py                 # Streamlit Web Application
├── train.py               # Data preprocessing and ML model training
├── evaluate.py            # Model evaluation metrics
├── fuzzy.py               # Fuzzy Logic System
├── hybrid.py              # Hybrid AI model
├── main.py                # Main execution file
├── requirements.txt       # Project dependencies
├── final_depression_dataset_1.csv
└── README.md
```

---

# ⚙️ Technologies Used

## Programming Language

* Python 3.10+

## Libraries & Frameworks

* Streamlit
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* CatBoost
* Scikit-Fuzzy
* Matplotlib

---

# 🧠 Machine Learning Models

## 1. XGBoost

Used for gradient boosting based classification with high accuracy and feature importance analysis.

## 2. CatBoost

Handles categorical data efficiently and improves prediction stability.

## 3. Fuzzy Logic System

Rule-based intelligent inference system used for uncertainty handling and human-like reasoning.

---

# 📊 Fuzzy Logic Rules

The fuzzy system evaluates:

* Sleep duration
* Stress level
* Social interaction score

### Example Rules

```text
IF sleep is LOW AND stress is HIGH
THEN risk is HIGH

IF sleep is HIGH AND stress is LOW
THEN risk is LOW

IF social interaction is LOW
THEN risk is HIGH
```

# 🧪 Dataset Information

The dataset contains various mental health related attributes such as:

* Gender
* Age
* Academic Pressure
* Work Pressure
* Sleep Duration
* Financial Stress
* Dietary Habits
* Job Satisfaction
* Family Mental Illness History
* Suicidal Thoughts
* Work/Study Hours
* CGPA

---

# 🔄 Workflow

## Step 1: Data Preprocessing

* Missing value handling
* Label Encoding
* Feature Scaling
* Train-test split

## Step 2: Train Models

* XGBoost training
* CatBoost training

## Step 3: Fuzzy Inference

* Membership functions
* Rule evaluation
* Risk scoring

## Step 4: Hybrid Decision

* Combines ML confidence scores with fuzzy risk score

## Step 5: Visualization

* Dashboard charts
* Confidence comparison
* Risk analysis

---

# 🚀 Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/mental-health-risk-assessment.git

cd mental-health-risk-assessment
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

## Run Streamlit Application

```bash
streamlit run app.py
```

---

# 📊 Evaluation Metrics

The system evaluates models using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

# 🖥️ Streamlit Dashboard

The application dashboard includes:

* User input form
* Model confidence comparison
* Risk breakdown chart
* ML vs Fuzzy comparison
* Final assessment result

---

# 📌 Sample Output

```text
Final Assessment:
High Risk ⚠️

Immediate attention recommended
Overall Confidence: 87.45%
```

---

# 📚 Future Enhancements

* Deep Learning integration
* Real-time chatbot support
* Cloud deployment
* Multi-language support
* Emotion detection using NLP
* Mobile application integration

---

# 👨‍💻 Authors

* Aditya Mittal

---

# 📄 License

This project is licensed under the MIT License.

---

# 📎 File References

* Streamlit Application Logic 
* Evaluation Metrics Module 
* Fuzzy Logic System 
* Hybrid Model Implementation 
* Main Execution File 
* Requirements File 
* Training Pipeline 
