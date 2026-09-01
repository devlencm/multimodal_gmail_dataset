import os
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from keras.src.layers import Bidirectional
from tensorflow.keras.metrics import AUC

from sklearn.metrics import roc_curve, roc_auc_score

from scipy.optimize import brentq
from scipy.interpolate import interp1d

from utils.lstm_train_utils import get_splits


np.random.seed(42)

# Compute EER
def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    return eer

# Main evaluation
def test_biLSTM(modality="mouse", data_path = './', model_path="./", split_method="inter", save_scores=True):
    os.makedirs("model_scores", exist_ok=True)  
    os.makedirs("results", exist_ok=True)

    results = []
    all_scores = {}

    # Load data
    X = np.load(os.path.join(data_path, f"{modality}_X.npy"), mmap_mode="r")
    user_ids = np.load(os.path.join(data_path, f"{modality}_y.npy"), mmap_mode="r")
    sessions = np.load(os.path.join(data_path, f"{modality}_sessions.npy"), mmap_mode="r")
    trajectory_times = np.load(os.path.join(data_path, f"{modality}_trajectory_times.npy"), mmap_mode="r")
    all_users = np.unique(user_ids)

    # Get train/val split
    splits = get_splits(
        user_ids,
        sessions,
        all_users,
        split_method,
        train_sessions=[1, 2],
        train_fraction=2 / 3
    )

    # Evaluate the trained models
    for user, session, train_idx, val_idx in splits:
        # Construct model path based on split type
        if split_method == "intra":
            model_file = os.path.join(model_path, f"{user}_session_{session}_best.keras")

        elif split_method == "inter":
            model_file = os.path.join(model_path, f"{user}_best_val_auc.keras")

        else:
            raise ValueError("split_method must be 'inter' or 'intra'")

        # Ensure model path exists
        if not os.path.exists(model_file):
            if split_method == "intra":
                print(f"Missing model for user {user}, session {session}")

            else:
                print(f"Missing model for user {user}")

            continue

        # Initializer model output score storage (used later in fusion)
        if user not in all_scores:
            all_scores[user] = {}

        # Load model
        model = load_model(model_file)

        # Training data and labels
        X_train = X[train_idx]
        y_train = (user_ids[train_idx] == user).astype(np.float32)

        # Validation data and labels
        X_val = X[val_idx]
        y_val = (user_ids[val_idx] == user).astype(np.float32)

        # Check that both classes are present
        if (
            np.sum(y_train == 1) == 0
            or
            np.sum(y_train == 0) == 0
            or
            np.sum(y_val == 1) == 0
            or
            np.sum(y_val == 0) == 0
        ):
            if split_method == "intra":

                print(
                    f"Skipping user {user}, "
                    f"session {session}: "
                    f"missing one or more classes"
                )

            else:

                print(
                    f"Skipping user {user}: "
                    f"missing one or more classes"
                )

            del model

            continue

        # Get model scores on val set
        val_scores = model.predict(X_val, batch_size=1024, verbose=0).ravel()


        # Test metrics
        test_auc = roc_auc_score(y_val, val_scores)
        test_eer = compute_eer(y_val, val_scores)

        # Print results
        if split_method == "intra":
            print(f"User {user}, Session {session}: EER={test_eer:.4f}, AUC={test_auc:.4f}")
        else:
            print(f"User {user}: EER={test_eer:.4f}, AUC={test_auc:.4f}")

        # Save metrics
        if split_method == "intra":

            results.append(
                {
                    "User": user,
                    "Session": session,
                    "EER": test_eer,
                    "AUC": test_auc
                }
            )

        else:

            results.append(
                {
                    "User": user,
                    "EER": test_eer,
                    "AUC": test_auc
                }
            )

        val_idx_array = np.asarray(val_idx)

        score_data = {
            "val": {
                "scores": val_scores,
                "start": trajectory_times[val_idx_array, 0],
                "end": trajectory_times[val_idx_array, 1],
                "labels": user_ids[val_idx_array],
                "session": sessions[val_idx_array]
            }
        }

        # Store scores
        if split_method == "intra":
            all_scores[user][session] = score_data
        else:
            all_scores[user] = score_data

        del model
        del y_train
        del y_val
        del val_scores

    # Create results df
    df_results = pd.DataFrame(results)

    # Add Mean rows
    mean_eer = df_results["EER"].mean()
    mean_auc = df_results["AUC"].mean()

    if split_method == "intra":

        mean_row = {
            "User": "Mean",
            "Session": np.nan,
            "EER": mean_eer,
            "AUC": mean_auc
        }

    else:

        mean_row = {
            "User": "Mean",
            "EER": mean_eer,
            "AUC": mean_auc
        }

    df_results = pd.concat([df_results, pd.DataFrame([mean_row])], ignore_index=True)

    # Save scores
    if save_scores:
        modality_name = modality
        base_dir = os.path.join("model_scores", "LSTM", modality_name)
        os.makedirs(base_dir, exist_ok=True)

        if split_method == "intra":

            # Intra: One score file per USER / SESSION
            for user, user_sessions in all_scores.items():
                user_dir = os.path.join(base_dir, f"user_{user}")
                os.makedirs(user_dir, exist_ok=True)

                for sess, score_data in user_sessions.items():
                    np.save(os.path.join(user_dir, f"session_{sess}.npy"), score_data)

        else:
            # Inter: One score file containing all users
            np.save(os.path.join(base_dir, "user_scores.npy"), all_scores)

    # Save metrics
    output_file = os.path.join("results", f"user_metrics_LSTM_{modality}_{split_method}.csv")
    df_results.to_csv(output_file, index=False)
    print(f"Saved scores and metrics for {modality} ({split_method})")

    return df_results


if __name__ == "__main__":
    for modality in ["mouse", "keystroke", "scroll"]:
        for split in ["intra", "inter"]:
            test_biLSTM(modality=modality,
                        model_path=(f"{modality}_model_{split}/"),
                        split_method=split,
                        save_scores=True
                        )
