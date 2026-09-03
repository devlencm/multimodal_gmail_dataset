import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import roc_curve

from utils.train_utils import get_splits


def generate_random_adversarial(vector_size, n, modality, feat_cols):
    # If not the random widget modality, generate random vectors where each feature [0, 1]
    if modality != "widget":
        return np.random.uniform(0.0, 1.0, size=(n, vector_size))

    # For widget features, randomly select 1 or 0 for one-hot features, and from range [0, 1] for durations
    adversarial_vectors = np.empty((n, vector_size), dtype=np.float32)

    for i, col in enumerate(feat_cols):
        if "dur" in col.lower():
            # Duration features are continuous
            adversarial_vectors[:, i] = np.random.uniform(0.0, 1.0, size=n)
        else:
            # One-hot widget features are binary
            adversarial_vectors[:, i] = np.random.randint(0, 2, size=n)

    return adversarial_vectors


def load_model(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")

    # Load classifier and scaler
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return clf, scaler


def get_eer_threshold(clf, X_val, y_val):
    # Get model scores
    scores = clf.predict_proba(X_val)[:, 1]

    # Calculate fpr, tpr, and thresholds from ROC curve
    fpr, tpr, thresholds = roc_curve(y_val, scores)
    fnr = 1.0 - tpr

    # Index for EER threshold is where fpr and fnr are equal
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_threshold = thresholds[eer_idx]

    # Calculate EER
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0

    return eer_threshold, eer


def adversarial_inference(clf, adversarial_vectors, eer_threshold):
    # Predict scores for adversarial vectors
    scores = clf.predict_proba(adversarial_vectors)[:, 1]

    # Success is where the model score >= EER threshold
    attack_successes = scores >= eer_threshold

    # Get mean ASR
    ASR = np.mean(attack_successes)

    return ASR


def evaluate_user(data, split_list, model, modality, split_type, user, adversarial_vectors):
    # Find split for user
    user_split = None

    # Get the split for a given user
    for split_user, session, train_idx, val_idx in split_list:
        if split_user == user:
            user_split = (session, train_idx, val_idx)
            break

    if user_split is None:
        raise ValueError(
            f"Could not find split for user {user}"
        )

    session, train_idx, val_idx = user_split

    # Load saved model and scaler
    if split_type == "inter":
        model_dir = os.path.join("models", model, modality, f"user_{user}")

    else:
        if session is None:
            raise ValueError(
                "Expected a session for intra-session evaluation"
            )

        model_dir = os.path.join("models", model, modality, f"user_{user}", f"session_{session}")

    clf, scaler = load_model(model_dir)

    # Create val set
    feat_cols = data.columns.to_list()[4:]
    X_val = data.loc[val_idx, feat_cols].values
    y_val = (data.loc[val_idx, "User_ID"] == user).astype(int).values

    # Scale val data using saved scaler
    X_val_scaled = scaler.transform(X_val)

    # Get EER threshold and EER from val data
    eer_threshold, eer = get_eer_threshold(clf, X_val_scaled, y_val)

    # Calculate ASR using the shared random vectors
    ASR = adversarial_inference(clf, adversarial_vectors, eer_threshold)

    return ASR, eer_threshold, eer


def main():
    models = ["SVM", "GBM", "RF"]
    split_types = ["inter", "intra"]
    modalities = ["keystroke", "mouse", "scroll", "widget"]
    train_sessions = [1, 2]
    n_adversarial = 1_000_000 

    results_dir = "adversarial_results"
    os.makedirs(results_dir, exist_ok=True)

    for modality in modalities:
        print(f"\nGenerating random vectors for {modality}...")

        # Load data once per modality
        data_path = f"{modality}_features.csv"
        data = pd.read_csv(data_path).fillna(0)

        # Get users
        all_users = np.unique(data["User_ID"])
        all_users = sorted([int(user_id) for user_id in all_users])

        # Feature columns
        feat_cols = data.columns.to_list()[4:]

        vector_size = len(feat_cols)

        # Generate one random attack set per modality.
        adversarial_vectors = generate_random_adversarial(
            vector_size=vector_size,
            n=n_adversarial,
            modality=modality,
            feat_cols=feat_cols
        )

        print(f"Generated {n_adversarial:,} random vectors with {vector_size} features")

        for model in models:
            for split_type in split_types:
                print(f"\nEvaluating {model} / {modality} / {split_type}")

                # Recreate original train/val splits
                split_list = list(
                    get_splits(
                        data,
                        all_users,
                        split_type,
                        train_sessions,
                        train_fraction=2 / 3
                    )
                )

                results = []

                for user in tqdm(all_users, desc=f"{model} {split_type}", unit="user"):
                    try:
                        ASR, eer_threshold, eer = evaluate_user(
                            data=data,
                            split_list=split_list,
                            model=model,
                            modality=modality,
                            split_type=split_type,
                            user=user,
                            adversarial_vectors=adversarial_vectors
                        )

                        results.append({
                            "User": user,
                            "ASR": ASR,
                            "Threshold": eer_threshold,
                            "EER": eer
                        })

                    except Exception as e:
                        print(f"Skipping user {user}: {e}")

                # Add mean
                if results:
                    mean_asr = np.mean([result["ASR"] for result in results])
                    mean_threshold = np.mean([result["Threshold"] for result in results])
                    mean_eer = np.mean([result["EER"] for result in results])

                    results.append({
                        "User": "Mean",
                        "ASR": mean_asr,
                        "Threshold": mean_threshold,
                        "EER": mean_eer
                    })

                # Save results
                results_path = os.path.join(results_dir, f"asr_{model}_{modality}_{split_type}.csv")
                pd.DataFrame(results).to_csv(results_path, index=False)

                print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()