# main.py
import argparse
import sys
import numpy as np
from model_pipeline import (
    prepare_data, build_model, train_model,
    evaluate_model, save_model, load_model,
    predict_new_samples
)

def main():
    parser = argparse.ArgumentParser(description="Pipeline ML Random Forest - Iris Dataset")

    parser.add_argument("--prepare_data", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    parser.add_argument("--predict", action="store_true", help="Faire une prédiction")
    parser.add_argument("--full_pipeline", action="store_true", help="Pipeline complet")

    parser.add_argument("--model_path", type=str, default="rf_model.joblib")
    parser.add_argument("--save_path", type=str, default="rf_model.joblib")
    parser.add_argument("--test_size", type=float, default=0.2)

    args = parser.parse_args()

    try:
        # -----------------------
        # Prepare data
        # -----------------------
        if args.prepare_data:
            print("=== PREPARE DATA ===")
            X_train, X_test, y_train, y_test, scaler = prepare_data(test_size=args.test_size)
            print("Données préparées avec succès.")
            return

        # -----------------------
        # Train model
        # -----------------------
        if args.train:
            print("=== TRAIN MODEL ===")
            X_train, X_test, y_train, y_test, scaler = prepare_data(test_size=args.test_size)
            model = build_model()
            train_model(model, X_train, y_train)
            save_model(model, scaler, args.save_path)
            print("Modèle entraîné et sauvegardé.")
            return

        # -----------------------
        # Evaluate model
        # -----------------------
        if args.evaluate:
            print("=== EVALUATE MODEL ===")
            model, scaler = load_model(args.model_path)
            _, X_test, _, y_test, _ = prepare_data(test_size=args.test_size)
            evaluate_model(model, X_test, y_test)
            return

        # -----------------------
        # Predict
        # -----------------------
        if args.predict:
            print("=== PREDICT MODE ===")
            model, scaler = load_model(args.model_path)

            sample = np.array([
                [5.1, 3.5, 1.4, 0.2],
                [6.7, 3.1, 4.4, 1.4]
            ])

            predict_new_samples(model, scaler, sample)
            return

        # -----------------------
        # Full pipeline
        # -----------------------
        if args.full_pipeline:
            print("=== FULL PIPELINE ===")
            X_train, X_test, y_train, y_test, scaler = prepare_data(test_size=args.test_size)

            model = build_model()
            train_model(model, X_train, y_train)
            evaluate_model(model, X_test, y_test)
            save_model(model, scaler, args.save_path)

            print("Pipeline complet exécuté.")
            return

        # Si aucune commande n’est fournie
        print("Aucune option fournie. Utilise --help pour voir les commandes disponibles.")

    except Exception as e:
        print(f"Erreur : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()