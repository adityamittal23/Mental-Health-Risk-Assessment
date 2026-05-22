from train import load_and_preprocess, train_models
from fuzzy import create_fuzzy_system, get_fuzzy_score
from hybrid import train_hybrid
from evaluate import evaluate

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess("final_depression_dataset_1.csv")

# Train ML models
xgb, cat = train_models(X_train, y_train)

# Predictions
y_pred_xgb = xgb.predict(X_test)
y_pred_cat = cat.predict(X_test)

# Evaluate
evaluate("XGBoost", y_test, y_pred_xgb)
evaluate("CatBoost", y_test, y_pred_cat)

# Fuzzy system
fuzzy_sim = create_fuzzy_system()

# Hybrid model
y_pred_hybrid = train_hybrid(X_train, X_test, y_train, fuzzy_sim, get_fuzzy_score)

evaluate("Hybrid Model", y_test, y_pred_hybrid)