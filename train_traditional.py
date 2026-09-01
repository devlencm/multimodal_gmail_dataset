import os
from os.path import split

import joblib
import pandas as pd
import numpy as np

from utils.train_utils import *
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from cuml.svm import SVC as cuSVC
import cupy as cp
import time

np.random.seed(42)


def train(modalities="all", model="GBM", split_type="intra", train_sessions=[1, 2], save_scores=True):
    os.makedirs("model_scores", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Patch Intelex extension for sklearn for GBM and RF for speed
    if model != "SVM":
        from sklearnex import patch_sklearn

        patch_sklearn()

    # Establish queue of feature vector CSV's
    if modalities == "all":
        queue = ["scroll_features.csv", "keystroke_features.csv", "mouse_features.csv", "widget_features.csv"]
    else:
        if isinstance(modalities, list):
            queue = [m + "_features.csv" for m in modalities]
        else:
            raise TypeError("Modalities must be 'all' or of list type, e.g. ['keystroke', 'widget']")

    # Loop through queue
    for file in queue:
        all_scores = {}
        results = []
        skipped_users = []

        # Read modality data CSV
        data = pd.read_csv(file).fillna(value=0)

        # Get user IDs
        all_users = np.unique(data['User_ID'])
        all_users = sorted([int(user) for user in all_users])

        print(f"Training {file.split('_')[0]} Modality, Model Type {model}, {len(all_users)} Users...\n")

        # Generate splits based on split_type
        split_list = list(
            get_splits(
                data,
                all_users,
                split_type,
                train_sessions,
                train_fraction=2 / 3
            )
        )

        # Loop through data splits
        for user, session, train_idx, test_idx in split_list:
            data_users = data.copy()
            feat_cols = data_users.columns.to_list()[4:]

            if user not in all_scores:
                all_scores[user] = {}

            if session is not None:
                all_scores[user][session] = {}

            X_train = data_users.loc[train_idx, feat_cols].values
            y_train = (data_users.loc[train_idx, "User_ID"] == user).astype(int).values
            
            X_test = data_users.loc[test_idx, feat_cols].values
            y_val = (data_users.loc[test_idx, "User_ID"] == user).astype(int).values

            # Oversample genuine samples to size of impostor
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            
            X_gen_train = X_train[y_train == 1]
            X_imp_train = X_train[y_train == 0]
            over_inds = np.random.choice(len(X_gen_train), len(X_imp_train), replace=True)
            X_gen_train_bal = X_gen_train[over_inds]

            # Create balanced set and labels
            X_train_bal = np.vstack((X_gen_train_bal, X_imp_train))
            y_train = np.concatenate((np.ones(len(X_gen_train_bal)), np.zeros(len(X_imp_train))))
            
            if session is None:  # Inter-session, session not set
                print(f"User {user}: Train {len(X_train)} rows, Test {len(X_test)} rows")
            else:  # Intra-session, show session specific output
                print(f"User {user}, Session {session}: Train {len(X_train)} rows, Test {len(X_test)} rows")

            # Scale 0-1
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X_train_bal)
            X_test = scaler.transform(X_test)

            if model == "GBM":
                clf = GradientBoostingClassifier(n_estimators=300, max_depth=5)

            elif model == "SVM":
                X_train = cp.asarray(X_train)
                y_train = cp.asarray(y_train)
                X_test = cp.asarray(X_test)
                
                clf = cuSVC(kernel="rbf", probability=True)

            else:
                clf = RandomForestClassifier(n_estimators=800, max_depth=3, max_features=15, n_jobs=-1)

            clf.fit(X_train, y_train)

            modality_name = file.split("_")[0]

            if session is None:
                save_dir = os.path.join('models', model, modality_name, f"user_{user}")
            else:
                save_dir = os.path.join('models', model, modality_name, f"user_{user}/session_{session}")

            os.makedirs(save_dir, exist_ok=True)

            # Save classifier
            joblib.dump(clf, os.path.join(save_dir, "model.joblib"))

            # Save scaler
            joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))

            # Get val set model output scores
            test_scores = clf.predict_proba(X_test)[:, 1]
            
            # Convert CuPy arrays back to NumPy for sklearn metrics/saving
            if model == "SVM":
                train_scores = cp.asnumpy(train_scores)
                test_scores = cp.asnumpy(test_scores)
                y_train = cp.asnumpy(y_train)
                y_val = cp.asnumpy(y_val)
                
            fpr, tpr, thres = roc_curve(y_train, train_scores)
            train_eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
            train_auc = roc_auc_score(y_train, train_scores)

            fpr, tpr, thres = roc_curve(y_val, test_scores)
            test_eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
            test_auc = roc_auc_score(y_val, test_scores)

            print(
                f"User {user} Session {session} "
                f"Train EER: {train_eer}, "
                f"Train AUC: {train_auc}, "
                f"Test EER: {test_eer * 100:.2f}, "
                f"AUC: {test_auc:.4f}"
            )
            
            results.append({"User": user, "Session": session, "EER": test_eer, "AUC": test_auc})

            score_data = {
                "test": {
                    "scores": test_scores,
                    "start": data_users.loc[test_idx, "start"].values,
                    "end": data_users.loc[test_idx, "end"].values,
                    "labels": data_users.loc[test_idx, "User_ID"].values,
                    "session": data_users.loc[test_idx, "session"].values,
                },
            }

            if session is None:
                all_scores[user] = score_data
            else:
                all_scores[user][session] = score_data

        if save_scores:
            modality_name = file.split("_")[0]
            df_results = pd.DataFrame(results)
        
            mean_eer = df_results["EER"].mean()
            mean_auc = df_results["AUC"].mean()
        
            df_results = pd.concat(
                [
                    df_results,
                    pd.DataFrame([
                        {
                            "User": "Mean",
                            "Session": np.nan,
                            "EER": mean_eer,
                            "AUC": mean_auc
                        }
                    ])
                ],
                ignore_index=True
            )
        
            if split_type == "intra":
                # Intra: Save one score file per USER / SESSION
                base_dir = os.path.join(
                    "model_scores",
                    model,
                    modality_name
                )
        
                for user, user_sessions in all_scores.items():        
                    user_dir = os.path.join(base_dir, f"user_{user}")
                    os.makedirs(user_dir, exist_ok=True)
        
                    for sess, score_data in user_sessions.items():        
                        np.save(os.path.join(user_dir, f"session_{sess}.npy"), score_data)
        
                df_results.to_csv(
                    f"results/user_metrics_{model}_{modality_name}_intra.csv",
                    index=False
                )
        
            else:
                # Inter: one score file containing all users
                base_dir = os.path.join("model_scores", model, modality_name)
                os.makedirs(base_dir, exist_ok=True)
                np.save(os.path.join(base_dir, "user_scores.npy"), all_scores)
        
                df_results.to_csv(f"results/single_sample/user_metrics_{model}_{modality_name}_inter.csv", index=False)
        
            print(f"Saved scores for {modality_name}")
            print(f"Saved metrics for {modality_name}")
            print(f"Users Skipped: {np.unique(skipped_users)}\n")


for model in ["SVM", "GBM", "RF"]:
    train(
        model=model,
        modalities=["mouse", "widget", "keystroke", "scroll"],
        split_type="inter"
    )

    train(
        model=model,
        modalities=["mouse", "widget", "keystroke", "scroll"],
        split_type="intra"
    )
