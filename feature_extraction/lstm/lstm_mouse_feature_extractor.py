import csv
import os
import pandas as pd
import numpy as np

def trajectories(x_list, y_list, t_list, sess_list, dx_list, dy_list, dt_list):
    base_index = 0
    trajs_x, trajs_y, trajs_t, trajs_s = [], [], [], []
    trajs_dx, trajs_dy, trajs_dt = [], [], []

    for i in range(len(t_list) - 1):
        # split trajectory when gap >= 100
        if (t_list[i + 1] - t_list[i]) >= 100:
            traj_x = x_list[base_index:i + 1]
            traj_y = y_list[base_index:i + 1]
            traj_t = t_list[base_index:i + 1]
            traj_s = sess_list[base_index:i + 1]
            traj_dx = dx_list[base_index:i + 1]
            traj_dy = dy_list[base_index:i + 1]
            traj_dt = dt_list[base_index:i + 1]

            if len(traj_t) >= 3:
                trajs_x.append(traj_x)
                trajs_y.append(traj_y)
                trajs_t.append(traj_t)
                trajs_s.append(traj_s)
                trajs_dx.append(traj_dx)
                trajs_dy.append(traj_dy)
                trajs_dt.append(traj_dt)

            base_index = i + 1

    # handle the last segment after the final gap
    if base_index < len(t_list):
        traj_x = x_list[base_index:]
        traj_y = y_list[base_index:]
        traj_t = t_list[base_index:]
        traj_s = sess_list[base_index:]
        traj_dx = dx_list[base_index:]
        traj_dy = dy_list[base_index:]
        traj_dt = dt_list[base_index:]

        if len(traj_t) >= 3:
            trajs_x.append(traj_x)
            trajs_y.append(traj_y)
            trajs_t.append(traj_t)
            trajs_s.append(traj_s)
            trajs_dx.append(traj_dx)
            trajs_dy.append(traj_dy)
            trajs_dt.append(traj_dt)

    return (
        trajs_x,
        trajs_y,
        trajs_t,
        trajs_s,
        trajs_dx,
        trajs_dy,
        trajs_dt
    )


if __name__ == "__main__":

    MOUSE_ROOT = "../Gmail Formatted/mouse"

    all_traj_features = []
    y = []
    all_sessions = []
    all_traj_times = []

    # Loop over users
    for user_id in sorted(os.listdir(MOUSE_ROOT)):
        user_dir = os.path.join(MOUSE_ROOT, user_id)

        if not os.path.isdir(user_dir):
            continue

        user_id = int(user_id)

        # Loop over sessions per user
        for session_file in sorted(os.listdir(user_dir)):

            if not session_file.endswith(".csv"):
                continue

            session_path = os.path.join(user_dir, session_file)

            # Read session data drop duplicates
            data_mouse = pd.read_csv(session_path).drop_duplicates(
                subset="Timestamp"
            )

            if data_mouse.empty:
                continue

            session = int(
                session_file.replace("session_", "").replace(".csv", "")
            )

            # Raw mouse data
            x_raw = data_mouse["Mouse X"].astype(float).tolist()
            y_raw = data_mouse["Mouse Y"].astype(float).tolist()

            if len(x_raw) < 3:
                continue

            # Normalize coordinates
            screen_width = data_mouse["Mouse X"].max()
            screen_height = data_mouse["Mouse Y"].max()

            x_list = [
                x / screen_width if screen_width > 0 else 0
                for x in x_raw
            ]

            y_list = [
                y / screen_height if screen_height > 0 else 0
                for y in y_raw
            ]

            t_list = data_mouse["Timestamp"].astype(np.int64).tolist()

            sess_list = [session] * len(t_list)

            # Per-event features
            dx_list = [
                np.nan
            ] + [
                (x_list[i] - x_list[i - 1]) * 100
                for i in range(1, len(x_list))
            ]

            dy_list = [
                np.nan
            ] + [
                (y_list[i] - y_list[i - 1]) * 100
                for i in range(1, len(y_list))
            ]

            dt_list = [
                np.nan
            ] + [
                t_list[i] - t_list[i - 1]
                for i in range(1, len(t_list))
            ]

            # Split into trajectories
            trajs = trajectories(
                x_list,
                y_list,
                t_list,
                sess_list,
                dx_list,
                dy_list,
                dt_list
            )

            if len(trajs[0]) == 0:
                continue

            for traj_x, traj_y, traj_t, traj_s, traj_dx, traj_dy, traj_dt in zip(
                trajs[0],
                trajs[1],
                trajs[2],
                trajs[3],
                trajs[4],
                trajs[5],
                trajs[6]
            ):
                traj_df = pd.DataFrame({
                    "Mouse X": traj_x,
                    "Mouse Y": traj_y,
                    "Timestamp": traj_t,
                    "session": traj_s,
                    "dx": traj_dx,
                    "dy": traj_dy,
                    "dt": traj_dt
                })

                # Remove only the first session-level timestep
                traj_df = traj_df.dropna(
                    subset=["dx", "dy", "dt"]
                )

                # Features: (trajectory_length, 3)
                features = traj_df[
                    ["dx", "dy", "dt"]
                ].to_numpy(dtype=np.float32)

                all_traj_features.append(features)
                y.append(user_id)
                all_sessions.append(traj_s[0])
                all_traj_times.append([traj_t[0], traj_t[-1]])

    # Find the maximum trajectory length
    max_len = max(
        traj.shape[0]
        for traj in all_traj_features
    )

    # Allocate padded array
    X = np.zeros(
        (len(all_traj_features), max_len, 3),
        dtype=np.float32
    )

    # Copy each trajectory into the padded array
    for i, traj in enumerate(all_traj_features):
        X[i, :traj.shape[0], :] = traj

    y = np.asarray(y)

    print(len(X), len(y), len(all_sessions))
    assert len(X) == len(y) == len(all_sessions)

    np.save("../../features/lstm/mouse_X.npy", X)
    np.save("../../features/lstm/mouse_y.npy", y)
    np.save("../../features/lstm/mouse_sessions.npy", all_sessions)
    np.save("../../features/lstm/mouse_trajectory_times.npy", np.asarray(all_traj_times, dtype=np.int64))

    print(
        f"Saved {len(all_traj_features)} trajectories."
    )