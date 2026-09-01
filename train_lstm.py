import os
import numpy as np
from keras.src.layers import Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input,
    Masking,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.utils import shuffle
from utils.lstm_train_utils import *
np.random.seed(42)

def build_lstm_model(input_shape):
    model = Sequential(
        [
            Input(shape=(input_shape)),
            Masking(mask_value=0),
            Bidirectional(LSTM(64, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(64)),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", AUC(name="auc")],
    )

    return model


def train_lstm(data_path="./", out_path="./", modalities=[], split_method="inter"):
    os.makedirs(out_path, exist_ok=True)

    for modality in modalities:
        print(f"\nLoading modality: {modality}")

        # Load data
        X = np.load(os.path.join(data_path, f"{modality}_X.npy"), mmap_mode="r")
        user_ids = np.load(os.path.join(data_path, f"{modality}_y.npy"), mmap_mode="r")
        all_users = np.unique(user_ids)
        sessions = np.load(os.path.join(data_path, f"{modality}_sessions.npy"), mmap_mode="r")

        # Get train/val splits
        splits = get_splits(
            user_ids,
            sessions,
            all_users,
            split_method,
            train_sessions=[1, 2],
            train_fraction=2 / 3
        )

        for user, session, train_idx, val_idx in splits:
            print(f"\nTraining user {user}" + (f", session {session}" if session is not None else ""))

            # Get training and validation sets/labels
            X_train = X[train_idx]
            X_val = X[val_idx]
            y_train = (user_ids[train_idx] == user).astype(np.float32)
            y_val = (user_ids[val_idx] == user).astype(np.float32)

            # Oversample genuine/positive samples to match impostors
            gen_mask = y_train == 1
            imp_mask = y_train == 0
            X_gen_train = X_train[gen_mask]
            X_imp_train = X_train[imp_mask]

            # Ensure both classes are present
            if len(X_gen_train) == 0 or len(X_imp_train) == 0:
                print(f"Skipping user {user}: positive={len(X_gen_train)}, negative={len(X_imp_train)}")
                continue

            over_inds = np.random.choice(len(X_gen_train), size=len(X_imp_train), replace=True)
            X_gen_train_bal = X_gen_train[over_inds]

            # Create train and val sets
            X_train = np.concatenate([X_gen_train_bal, X_imp_train], axis=0)
            y_train = np.concatenate([np.ones(len(X_gen_train_bal), dtype=np.float32), np.zeros(len(X_imp_train), dtype=np.float32)])

            # Shuffle training data
            X_train, y_train = shuffle(X_train, y_train, random_state=42)

            print(f"Train: {len(y_train)} samples (positive={np.sum(y_train == 1)}, negative={np.sum(y_train == 0)})")
            print(f"Val: {len(y_val)} samples (positive={np.sum(y_val == 1)}, negative={np.sum(y_val == 0)})")

            # Create model
            model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))

            # Construct model out path base on split type
            if split_method == "intra":
                model_path = os.path.join(out_path, f"{user}_session_{session}_best.keras")
            else:
                model_path = os.path.join(out_path, f"{user}_best_val_auc.keras")

            checkpoint = ModelCheckpoint(
                model_path,
                monitor="val_auc",
                mode="max",
                save_best_only=True,
                verbose=1,
            )

            early_stop = EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=5,
                restore_best_weights=True
            )

            model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=64,
                callbacks=[checkpoint, early_stop],
            )


if __name__ == "__main__":
    for modality in ["mouse", "keystroke", "scroll"]:
        for split in ["intra", "inter"]:
            train_lstm(
                modalities=[modality],
                out_path=f"{modality}_model_{split}",
                split_method=split,
            )