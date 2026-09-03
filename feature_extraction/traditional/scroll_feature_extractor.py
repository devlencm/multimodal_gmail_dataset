import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import utils.scroll_features as scroll_features

SCROLL_ROOT = "../Gmail Formatted/scroll"


def main():

    big_dataframe = pd.DataFrame()
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

            # Store start/end times
            for l in traj_ts:
                traj_edge_times.append((l[0], l[-1]))

            df_features = pd.DataFrame()

            df_features["User_ID"] = [user] * len(traj_ys)
            df_features["session"] = [sess] * len(traj_ys)
            df_features["start"] = [l[0] for l in traj_ts]
            df_features["end"] = [l[-1] for l in traj_ts]
            df_features["Duration"] = scroll_features.duration(traj_ts)
            df_features["Distance"] = scroll_features.traveled_distance(traj_ys)
            df_features["Velocity"] = scroll_features.velocity(
                traj_ys, traj_ts
            )
            df_features["Acceleration"] = scroll_features.v_acceleration(
                traj_ys, traj_ts
            )

            big_dataframe = pd.concat(
                [big_dataframe, df_features],
                ignore_index=True
            )

    print(len(big_dataframe))

    big_dataframe.to_csv("../../features/traditional/scroll_features.csv", index=False)


if __name__ == "__main__":
    main()