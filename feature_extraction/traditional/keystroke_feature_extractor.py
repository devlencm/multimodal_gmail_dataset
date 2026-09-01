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

        # if event is up and the key is active, create press dict for the key with its press and release time
        elif event == 'up' and key in active_keys and len(active_keys[key]) > 0:
            press = {
                'key': key,
                'down': active_keys[key].pop(0),
                'up': time
            }

            key_presses.append(press)

    return key_presses


# The following features are sorted in order of frequency across dataset
alpha = "eatoinsrlh"
specials = ['shiftleft', 'backspace', 'controlleft', 'enter', 'period',
            'arrowdown', 'arrowup', 'arrowright', 'arrowleft',
            'controlright', 'shiftright']
common_digraphs = ['t_h', 'i_n', 'a_n', 'h_e',
                   'a_r', 'r_e', 'c_l', 'o_u', 'l_a', 'n_g', 'h_a']

# Only use the top few most frequent features to avoid the model learning unique keys instead of behavior
alpha = alpha[:10]
specials = specials[:0]
common_digraphs = common_digraphs[:5]

keys1 = list(alpha) + specials

# Define feature columns
features_per_key = ['dwell', 'flight_dd', 'flight_du', 'flight_ud', 'flight_uu']

static_features = []
for key in keys1 + common_digraphs:
    for key_feature in features_per_key:
        static_features.append(key + '_' + key_feature)

if __name__ == "__main__":

    KEYSTROKE_ROOT = "../Gmail Formatted/keystroke"
    buffer = None

    # Loop over users
    for user_id in sorted(os.listdir(KEYSTROKE_ROOT)):
        user_dir = os.path.join(KEYSTROKE_ROOT, user_id)

        if not os.path.isdir(user_dir):
            continue

        user_id = int(user_id)

        # Loop over sessions per user
        for session_file in sorted(os.listdir(user_dir)):

            if not session_file.endswith(".csv"):
                continue

            session_path = os.path.join(user_dir, session_file)

            # Read session data drop duplicates
            data = pd.read_csv(session_path)
            data = data.drop_duplicates()

            if data.empty:
                continue

            sess = int(session_file.replace("session_", "").replace(".csv", ""))
            sess_data = data[data['Key'].notnull()]

            # Get timestamps, key IDs, events, remove "key" and "digit" prefixes
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

            # Print sessions that produce 0 windows
            if len(key_presses) < 20:
                print(f"0 windows: User {user_id}, {session_file}")

            # Split into windows of 20 keystrokes
            window_size = 20

            for start_idx in range(0, len(key_presses) - window_size + 1, window_size):
                end_idx = start_idx + window_size

                window_presses = key_presses[start_idx:end_idx]

                key_features = {}

                # Feature computation
                for i in range(len(window_presses) - 1):

                    current_key = window_presses[i]
                    next_key = window_presses[i + 1]

                    dwell_time = current_key['up'] - current_key['down']

                    flight_time_dd = next_key['down'] - current_key['down']
                    flight_time_du = next_key['up'] - current_key['down']
                    flight_time_ud = next_key['down'] - current_key['up']
                    flight_time_uu = next_key['up'] - current_key['up']

                    arr = [
                        dwell_time,
                        flight_time_dd,
                        flight_time_du,
                        flight_time_ud,
                        flight_time_uu
                    ]

                    if current_key['key'] not in key_features:
                        key_features[current_key['key']] = []

                    key_features[current_key['key']].append(arr)
                    digraph_key = current_key['key'] + "_" + next_key['key']

                    if digraph_key in common_digraphs:

                        if digraph_key not in key_features:
                            key_features[digraph_key] = []

                        key_features[digraph_key].append(arr)

                # Aggregate features for this window
                feature_vector = []

                for key in keys1 + common_digraphs:
                    if key in key_features and len(key_features[key]) > 0:
                        avg = np.average(key_features[key], axis=0).tolist()
                        feature_vector.extend(avg)
                    else:
                        feature_vector.extend([0, 0, 0, 0, 0])

                # Add User_ID, session, start_time, end_time
                window_start_time = window_presses[0]['down']
                window_end_time = window_presses[-1]['up']

                cols = ['User_ID', 'session', 'start', 'end'] + static_features

                feature_vector = [
                                     user_id,
                                     sess,
                                     window_start_time,
                                     window_end_time
                                 ] + feature_vector

                feature_df = pd.DataFrame(np.array(feature_vector).reshape(1, len(cols)), columns=cols)

                if buffer is None:
                    buffer = feature_df
                else:
                    buffer = pd.concat(
                        [buffer, feature_df],
                        ignore_index=True
                    )

    # buffer.to_csv("keystroke_features.csv", index=False)
    print(len(buffer))