# model_pipeline.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# -------------------------------
# 1. Préparation des données
# -------------------------------
def prepare_data(test_size=0.2, random_state=42, stratify=True):
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train shape: {X_train_scaled.shape}")
    print(f"Test shape: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# -------------------------------
# 2. Construction du modèle
# -------------------------------
def build_model(n_estimators=100, max_depth=None, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    print("Random Forest model created.")
    return model


# -------------------------------
# 3. Entraînement
# -------------------------------
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model


# -------------------------------
# 4. Évaluation (SANS AUCUNE VISUALISATION)
# -------------------------------
def evaluate_model(model, X_test, y_test, show_plot=False):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nMatrice de confusion:")
    print(cm)
    print("\nClassification report:")
    print(report)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report
    }


# -------------------------------
# 5. Sauvegarde du modèle
# -------------------------------
def save_model(model, scaler, filepath='rf_model.joblib'):
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

    joblib.dump(model, filepath.replace('.joblib', '_model.joblib'))
    joblib.dump(scaler, filepath.replace('.joblib', '_scaler.joblib'))

    print(f"Model saved to: {filepath.replace('.joblib', '_model.joblib')}")
    print(f"Scaler saved to: {filepath.replace('.joblib', '_scaler.joblib')}")


# -------------------------------
# 6. Chargement du modèle
# -------------------------------
def load_model(filepath='rf_model.joblib'):
    model_path = filepath.replace('.joblib', '_model.joblib')
    scaler_path = filepath.replace('.joblib', '_scaler.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("Model and scaler loaded successfully.")
    return model, scaler


# -------------------------------
# 7. Prédiction sur nouveaux échantillons
# -------------------------------
def predict_new_samples(model, scaler, new_data, class_names=None):
    if class_names is None:
        class_names = ['setosa', 'versicolor', 'virginica']

    new_data_scaled = scaler.transform(new_data)
    predictions = model.predict(new_data_scaled)
    probabilities = model.predict_proba(new_data_scaled)

    predicted_labels = [class_names[p] for p in predictions]

    print("\nPredictions:")
    for i, (pred, prob) in enumerate(zip(predicted_labels, probabilities)):
        print(f"Sample {i+1}: {pred} - Probabilities: {prob}")

    return predictions, probabilities


if __name__ == "__main__":
    print("Testing Random Forest pipeline...")

    X_train, X_test, y_train, y_test, scaler = prepare_data()

    model = build_model()
    train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)

    save_model(model, scaler, "test_rf.joblib")

    print("Pipeline test completed.")