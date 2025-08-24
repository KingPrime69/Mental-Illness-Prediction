import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath("."))
from src.preprocessing import preprocessing

# Loading the dataset
DATA_PATH = "./data/raw/Patients.csv"
data = pd.read_csv(DATA_PATH, sep=";", header=0)

# Preprocess the data using the custom preprocessing function
df_preprocessed = preprocessing(DATA_PATH, mode="train")

# Define features and target variable
features = df_preprocessed.columns.drop('Mental Illness').tolist()
target = 'Mental Illness'
X = df_preprocessed[features]
y = df_preprocessed[target]

# Handle class imbalance: undersampling and oversampling (SMOTE)
print("Original dataset distribution:", Counter(y))
undersampler = RandomUnderSampler(random_state=42)
X_under, y_under = undersampler.fit_resample(X, y)
print("After undersampling:", Counter(y_under))
smote = SMOTE(random_state=42)
X_over, y_over = smote.fit_resample(X, y)
print("After oversampling (SMOTE):", Counter(y_over))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)

# Utility function to get feature importance (SHAP or default) 
def get_feature_importance(model, X, feature_names, method="auto", top_n=20):
    if method == "shap":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap_sum = np.abs(shap_values).mean(axis=0)
        feat_imp = pd.Series(shap_sum, index=feature_names).sort_values(ascending=False)
    else:
        feat_imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    return feat_imp.head(top_n)

# Utility function to evaluate a model and display metrics and confusion matrix
def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    return accuracy_score(y_test, y_pred), y_pred
    
# Utility function to plot ROC and Precision-Recall curves
def plot_roc_pr_curves(model, X_test, y_test, model_name="Model"):
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'AP = {pr_auc:.2f}')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Utility function for cross-validated accuracy
def cross_val_metrics(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print(f"Cross-validated accuracy ({cv}-fold): {scores.mean():.3f} ± {scores.std():.3f}")

# Define hyperparameter grid for XGBoost
'''param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 5, 10],
    'min_child_weight': [1, 3, 5],
    'learning_rate': [0.1],
    'n_jobs': [-1]
}

# Hyperparameter optimization with GridSearchCV for XGBoost
gr_cv = GridSearchCV(XGBClassifier(random_state=42, n_jobs=-1), param_grid, cv=5, scoring='neg_mean_absolute_error')
gr_cv.fit(X_train, y_train)
best_params = gr_cv.best_params_

# Train XGBoost model with best parameters
print("Starting XGBoost training")
xgb_model = XGBClassifier(**best_params)
xgb_model.fit(X_train, y_train)
print("XGBoost training completed")
''' 
print("Starting Random Forest training")
# Train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=500, max_depth=20, max_features=0.3,
    class_weight=None,
    random_state=42, n_jobs=-1, bootstrap=True, oob_score=False
)
rf_model.fit(X_train, y_train)
print("Random Forest training completed")

# Evaluate both models
#acc_xgb, y_pred_xgb = evaluate_model(xgb_model, X_test, y_test, name="XGBoost")
acc_rf, y_pred_rf = evaluate_model(rf_model, X_test, y_test, name="Random Forest")

# Plot ROC and Precision-Recall curves for both models
#plot_roc_pr_curves(xgb_model, X_test, y_test, model_name="XGBoost")
#plot_roc_pr_curves(rf_model, X_test, y_test, model_name="Random Forest")

# Print summary of accuracy scores
# print(f"XGBoost Accuracy: {acc_xgb:.4f}")
print(f"Random Forest Accuracy: {acc_rf:.4f}")

# Save trained models
joblib.dump(rf_model, "../data/models/random_forest_model.pkl", compress=3)
with open("../data/models/model_features.txt", "w") as f:
    for col in X_train.columns:
        f.write(col + "\n")
#joblib.dump(xgb_model, "../data/models/xgb_model.pkl", compress=3)

# --- Feature Importance ---
# This code identifies the variables that had the greatest influence on the prediction.
# Since it takes a particularly long time to run, it is currently commented out. You can run it if you want to obtain this information.

'''
feat_imp_xgb = get_feature_importance(xgb_model, X_train, X_train.columns, method="shap", top_n=20)
feat_imp_rf = get_feature_importance(rf_model, X_train, X_train.columns, method="shap", top_n=20)

print("\nTop features XGBoost (SHAP):\n", feat_imp_xgb)
print("\nTop features Random Forest:\n", feat_imp_rf)

plt.figure(figsize=(10, 4))
feat_imp_xgb.plot(kind='bar', title='Top 20 Features XGBoost (SHAP)')
plt.show()

plt.figure(figsize=(10, 4))
feat_imp_rf.plot(kind='bar', title='Top 20 Features Random Forest')
plt.show()
'''