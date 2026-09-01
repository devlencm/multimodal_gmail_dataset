import os
import pandas as pd
import numpy as np


def remove_keys(keys):
    if str(keys).startswith('key'):
        keys = keys[3:]
    elif str(keys).startswith('digit'):
        keys = keys[5:]
    return keys


def get_sess_index(idx, sess_inds):
    i = 1
    for index in sess_inds:
        if int(idx) < index:
            break
        i += 1
    return i


def pair_key_events(keys, times, events):
    active_keys = {}
    key_presses = []

    # Loop over rows
    for key, time, event in zip(keys, times, events):

        # If event is down, a key is active (waiting for up)
        if event == 'down':
            active_keys.setdefault(key, []).append(time)

        # If event is up and the key is active, pair it with the
        # earliest unmatched key-down event
        elif event == 'up' and key in active_keys and len(active_keys[key]) > 0:
            press = {
                'key': key,
                'down': active_keys[key].pop(0),
                'up': time
            }

            key_presses.append(press)

    return key_presses


if __name__ == "__main__":

    KEYSTROKE_ROOT = "../Gmail Formatted/keystroke"

    all_window_features = []
    y = []
    all_sessions = []
    all_window_times = []

    # Loop over users
    for user_id in sorted(os.listdir(KEYSTROKE_ROOT)):

        user_dir = os.path.join(KEYSTROKE_ROOT, user_id)

        if not os.path.isdir(user_dir):
            continue

        user_id = int(user_id)

        # Loop over sessions
        for session_file in sorted(os.listdir(user_dir)):

            if not session_file.endswith(".csv"):
                continue

            session_path = os.path.join(user_dir, session_file)

            # Read session data and drop duplicates
            data = pd.read_csv(session_path)
            data = data.drop_duplicates()

            if data.empty:
                continue

            sess = int(
                session_file.replace("session_", "").replace(".csv", "")
            )

            sess_data = data[data['Key'].notnull()]

            # Get timestamps, key IDs, and events
            t_list = sess_data['Timestamp'].astype(np.int64).tolist()

            keys = sess_data['Key'].tolist()
            keys = list(map(remove_keys, keys))

            event = sess_data['Up/Down'].tolist()

            # Pair key down/up events
            key_presses = pair_key_events(
                keys,
                t_list,
                event
            )

            # Split into windows of 20 keystrokes
            window_size = 20

            for start_idx in range(
                0,
                len(key_presses) - window_size + 1,
                window_size
            ):

                end_idx = start_idx + window_size

                window_presses = key_presses[start_idx:end_idx]

                window_features = []

                # Feature computation
                for i in range(len(window_presses) - 1):

                    current_key = window_presses[i]
                    next_key = window_presses[i + 1]

                    dwell_time = (
                        current_key['up'] - current_key['down']
                    )

                    flight_time_dd = (
                        next_key['down'] - current_key['down']
                    )

                    flight_time_du = (
                        next_key['up'] - current_key['down']
                    )

                    flight_time_ud = (
                        next_key['down'] - current_key['up']
                    )

                    flight_time_uu = (
                        next_key['up'] - current_key['up']
                    )

                    ascii_id = (
                        ord(current_key['key'])
                        if len(current_key['key']) == 1
                        else -1
                    )

                    arr = [
                        ascii_id,
                        dwell_time,
                        flight_time_dd,
                        flight_time_du,
                        flight_time_ud,
                        flight_time_uu
                    ]

                    window_features.append(arr)

                # Only retain complete 20-keystroke windows
                if len(window_features) == window_size - 1:

                    features = np.asarray(
                        window_features,
                        dtype=np.float32
                    )

                    # Start/end timestamps for this window
                    window_start_time = window_presses[0]['down']
                    window_end_time = window_presses[-1]['up']

                    all_window_features.append(features)
                    y.append(user_id)
                    all_sessions.append(sess)

                    # Store [start, end] for this window
                    all_window_times.append([
                        window_start_time,
                        window_end_time
                    ])

    # Since windows are fixed length, no padding is needed
    X = np.asarray(all_window_features, dtype=np.float32)
    y = np.asarray(y)

    window_times = np.asarray(all_window_times, dtype=np.int64)

    print(len(X), len(y), len(all_sessions), len(window_times))
    assert (len(X) == len(y) == len(all_sessions) == len(window_times))

    np.save("../../features/lstm/keystroke_X.npy", X)
    np.save("../../features/lstm/keystroke_y.npy", y)
    np.save("../../features/lstm/keystroke_sessions.npy", all_sessions)
    np.save("keystroke_window_times.npy", window_times)

    print(f"Saved {len(all_window_features)} keystroke windows.")