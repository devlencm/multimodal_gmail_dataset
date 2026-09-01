import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import utils.scroll_features as scroll_features

SCROLL_ROOT = "../Gmail Formatted/scroll"

def main():

    all_traj_features = []
    y = []
    all_sessions = []
    traj_edge_times = []

    for user_id in sorted(os.listdir(SCROLL_ROOT)):

        user_dir = os.path.join(SCROLL_ROOT, user_id)

        if not os.path.isdir(user_dir):
            continue

        user_id = int(user_id)

        for session_file in sorted(os.listdir(user_dir)):

            if not session_file.endswith(".csv"):
                continue

            session_path = os.path.join(user_dir, session_file)

            user_data = pd.read_csv(session_path)

            user_data = user_data.drop_duplicates(subset="Timestamp")
            user_data = user_data.dropna(subset=["Timestamp", "Delta Y"])

            if user_data.empty:
                continue

            session = int(
                session_file.replace("session_", "").replace(".csv", "")
            )

            # Check Timestamp values that require coercion
            bad_timestamp = pd.to_numeric(
                user_data["Timestamp"],
                errors="coerce"
            ).isna() & user_data["Timestamp"].notna()

            print("Invalid Timestamp values:")
            print(user_data.loc[bad_timestamp, "Timestamp"].unique())

            # Check Delta Y values that require coercion
            bad_delta_y = pd.to_numeric(
                user_data["Delta Y"],
                errors="coerce"
            ).isna() & user_data["Delta Y"].notna()

            print("\nInvalid Delta Y values:")
            print(user_data.loc[bad_delta_y, "Delta Y"].unique())

            # Ensure numeric columns
            user_data["Timestamp"] = pd.to_numeric(
                user_data["Timestamp"],
                errors="coerce"
            )

            user_data["Delta Y"] = pd.to_numeric(
                user_data["Delta Y"],
                errors="coerce"
            )

            # Sort by timestamp
            user_data = user_data.sort_values(by=["Timestamp"])

            user = user_id
            sess = session

            y_list = user_data["Delta Y"].tolist()

            scaler = MinMaxScaler(feature_range=(0, 1))
            y_list = scaler.fit_transform(
                np.array(y_list).reshape(-1, 1)
            ).flatten()

            t_list = user_data["Timestamp"].tolist()

            traj_ys, traj_ts = scroll_features.trajectories(
                y_list, t_list
            )

            for traj_y, traj_t in zip(traj_ys, traj_ts):
                dy = np.diff(traj_y)
                dt = np.diff(traj_t)

                features = np.column_stack(
                    [dy, dt]
                ).astype(np.float32)

                all_traj_features.append(features)
                y.append(user)
                all_sessions.append(sess)

                # Start/end timestamp for this trajectory
                traj_edge_times.append([
                    traj_t[0],
                    traj_t[-1]
                ])

    # Find the maximum trajectory length
    max_len = max(
        traj.shape[0]
        for traj in all_traj_features
    )

    # Allocate padded array
    X = np.zeros(
        (len(all_traj_features), max_len, 2),
        dtype=np.float32
    )

    # Copy each trajectory into the padded array
    for i, traj in enumerate(all_traj_features):
        X[i, :traj.shape[0], :] = traj

    y = np.asarray(y)

    print(len(X), len(y), len(all_sessions))
    assert len(X) == len(y) == len(all_sessions)

    np.save("../../features/lstm/scroll_X.npy", X)
    np.save("../../features/lstm/scroll_y.npy", y)
    np.save("../../features/lstm/scroll_sessions.npy", all_sessions)
    np.save("../../features/lstm/scroll_trajectory_times.npy", np.asarray(traj_edge_times, dtype=np.int64))

    print(f"Saved {len(all_traj_features)} scroll trajectories.")


if __name__ == "__main__":
    main()