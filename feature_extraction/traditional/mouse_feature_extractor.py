import os
import pandas as pd
import numpy as np

from utils.mouse_features import *

MOUSE_ROOT = "../Gmail Formatted/mouse"


def get_user_screen_bounds(user_dir):
    max_x = 0
    max_y = 0

    for session_file in os.listdir(user_dir):
        if not session_file.endswith(".csv"):
            continue

        session_path = os.path.join(user_dir, session_file)

        data_mouse = pd.read_csv(session_path).drop_duplicates(
            subset="Timestamp"
        )

        if data_mouse.empty:
            continue

        max_x = max(max_x, data_mouse["Mouse X"].max())
        max_y = max(max_y, data_mouse["Mouse Y"].max())

    return max_x, max_y


if __name__ == "__main__":

    traj_edge_times_all = []
    feature_dfs = []

    for user_id in sorted(os.listdir(MOUSE_ROOT)):

        user_dir = os.path.join(MOUSE_ROOT, user_id)

        if not os.path.isdir(user_dir):
            continue

        user_id = int(user_id)

        screen_width, screen_height = get_user_screen_bounds(user_dir)

        for session_file in sorted(os.listdir(user_dir)):

            if not session_file.endswith(".csv"):
                continue

            session_path = os.path.join(user_dir, session_file)

            data_mouse = pd.read_csv(session_path).drop_duplicates(subset="Timestamp")

            if data_mouse.empty:
                continue

            session = int(session_file.replace("session_", "").replace(".csv", ""))

            # Raw mouse data
            x_raw = data_mouse["Mouse X"].astype(float).tolist()
            y_raw = data_mouse["Mouse Y"].astype(float).tolist()

            if len(x_raw) < 3:
                continue

            # Normalize coordinates
            x_list = [x / screen_width if screen_width > 0 else 0 for x in x_raw]

            y_list = [y / screen_height if screen_height > 0 else 0 for y in y_raw]

            t_list = data_mouse["Timestamp"].astype(np.int64).tolist()

            sess_list = [session] * len(t_list)

            # Segment into trajectories
            trajs = trajectories(x_list, y_list, t_list, sess_list)

            if len(trajs[0]) == 0:
                continue

            # Trajectory start/end information
            traj_edge_times = []

            for times, sessions in zip(trajs[2], trajs[3]):

                traj_edge_times.append([times[0], times[-1], sessions[0], user_id])

            traj_edge_times_all.extend(traj_edge_times)

            # Compute features for all trajectories
            dur_list = duration(trajs[2])
            dk_list_2pts, dk_list_traj = traveled_distance(trajs[0], trajs[1])
            sm_list = curve_len(trajs[0], trajs[1])
            v_list_traj = velocity_traj(sm_list, dur_list)
            v_list_pt = velocity_pt(dk_list_2pts, trajs[2])
            hv_list_traj, hv_list_pt, vv_list_traj, vv_list_pt = hv_velocity(
                trajs[0], trajs[1], trajs[2]
            )
            ha_list_traj, ha_list_pt, va_list_traj, va_list_pt = hv_acceleration(
                trajs[0], trajs[1], trajs[2]
            )
            am_list_traj, am_list_2pts = angle_movement(trajs[0], trajs[1])

            # Summary statistics
            dkmins = minimums(dk_list_2pts)
            dkmaxs = maximums(dk_list_2pts)
            dkmeans = means(dk_list_2pts)

            vmins = minimums(v_list_pt)
            vmaxs = maximums(v_list_pt)
            vmeans = means(v_list_pt)

            hvmins = minimums(hv_list_pt)
            hvmaxs = maximums(hv_list_pt)
            hvmeans = means(hv_list_pt)

            vvmins = minimums(vv_list_pt)
            vvmaxs = maximums(vv_list_pt)
            vvmeans = means(vv_list_pt)

            hamins = minimums(ha_list_pt)
            hamaxs = maximums(ha_list_pt)
            hameans = means(ha_list_pt)

            vamins = minimums(va_list_pt)
            vamaxs = maximums(va_list_pt)
            vameans = means(va_list_pt)

            ammins = minimums(am_list_2pts)
            ammaxs = maximums(am_list_2pts)
            ammeans = means(am_list_2pts)

            session_features = pd.DataFrame(
                {
                    "User_ID": user_id,
                    "session": session,
                    "start": [x[0] for x in traj_edge_times],
                    "end": [x[1] for x in traj_edge_times],
                    "duration": dur_list,
                    "straight_distance": dk_list_traj,
                    "curve_length": sm_list,
                    "velocity": v_list_traj,
                    "horiz_velocity": hv_list_traj,
                    "vert_velocity": vv_list_traj,
                    "horiz_acc": ha_list_traj,
                    "vert_acc": va_list_traj,
                    "angle": am_list_traj,
                    "min_str_dist": dkmins,
                    "max_str_dist": dkmaxs,
                    "mean_str_dist": dkmeans,
                    "min_velocity": vmins,
                    "max_velocity": vmaxs,
                    "mean_velocity": vmeans,
                    "min_horiz_vel": hvmins,
                    "max_horiz_vel": hvmaxs,
                    "mean_horiz_vel": hvmeans,
                    "min_vert_vel": vvmins,
                    "max_vert_vel": vvmaxs,
                    "mean_vert_vel": vvmeans,
                    "min_horiz_acc": hamins,
                    "max_horiz_acc": hamaxs,
                    "mean_horiz_acc": hameans,
                    "min_vert_acc": vamins,
                    "max_vert_acc": vamaxs,
                    "mean_vert_acc": vameans,
                    "min_angle": ammins,
                    "max_angle": ammaxs,
                    "mean_angle": ammeans,
                }
            )

            feature_dfs.append(session_features)

    # Save the time boundaries for each sequence for use in widget segmentation
    traj_edge_times_df = pd.DataFrame(
        traj_edge_times_all, columns=["start", "end", "session", "user"]
    )

    traj_edge_times_df.to_csv("../../features/traditional/traj_edge_times.csv", index=False)

    mouse_features_df = pd.concat(feature_dfs, ignore_index=True)
    mouse_features_df.to_csv("../../features/traditional/mouse_features.csv", index=False)

    print(f"Saved {len(mouse_features_df)} trajectory feature rows.")
