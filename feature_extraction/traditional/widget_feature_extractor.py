import os
import numpy as np
import pandas as pd
import time

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

t0 = time.time()

WIDGET_ROOT = "../Gmail Formatted/widget"


def hot_encode_with_durations(traj_t, widget_root):
    users = np.unique(traj_t['user'])

    # Get all widget IDs from all user/session files
    all_widgets_set = set()

    for user_id in sorted(os.listdir(widget_root)):

        user_dir = os.path.join(widget_root, user_id)

        if not os.path.isdir(user_dir):
            continue

        for session_file in sorted(os.listdir(user_dir)):

            if not session_file.endswith(".csv"):
                continue

            session_path = os.path.join(user_dir, session_file)

            session_widget_data = pd.read_csv(session_path)

            if session_widget_data.empty:
                continue

            all_widgets_set.update(
                session_widget_data['WidgetID'].dropna().unique()
            )

    all_widgets = np.array(sorted(all_widgets_set))

    encode_map = {widget: idx + 4 for idx, widget in enumerate(all_widgets)}  # start at index 4

    total_rows = []

    # Loop over users
    for user in users:
        user_data = traj_t[traj_t['user'] == user]

        user_dir = os.path.join(widget_root, str(int(user)))

        if not os.path.isdir(user_dir):
            continue

        sessions = np.unique(user_data['session'])

        # Loop over sessions
        for sess in sessions:
            sess_data = user_data[user_data['session'] == sess]

            # Read the corresponding widget session file
            session_file = f"session_{int(sess)}.csv"
            session_path = os.path.join(user_dir, session_file)

            if os.path.isfile(session_path):
                sess_widget_data = pd.read_csv(session_path)
                sess_widget_data = sess_widget_data.drop_duplicates(
                    subset="Timestamp"
                )
            else:
                sess_widget_data = pd.DataFrame(
                    columns=['Timestamp', 'WidgetID', 'Event']
                )

            start_times = sess_data['start'].values
            end_times = sess_data['end'].values

            # Loop over trajectory start times
            for i in range(len(start_times)):
                open_widgets = dict()
                w_traj_durs = np.zeros(len(all_widgets))

                times = sess_widget_data['Timestamp'].values
                widgets = sess_widget_data['WidgetID'].values
                events = sess_widget_data['Event'].values

                # cur_time is row timestamp, wid is widgetID, event is enter or leave
                for cur_time, wid, event in zip(times, widgets, events):
                    # If the mouse entered the widget, add the time entered to open widgets
                    if event == 'Mouse Enter':
                        open_widgets[wid] = cur_time
                    # If the mouse left the widget, and the widget was in open widgets, calculate duration
                    elif event == 'Mouse Leave' and wid in open_widgets:
                        enter_time = open_widgets[wid]
                        leave_time = cur_time
                        # If enter time is <= than the mouse trajectory end time, and leave time is >= the trajectory start time, keep duration
                        if enter_time <= end_times[i] and leave_time >= start_times[i]:
                            duration = min(leave_time, end_times[i]) - max(start_times[i], enter_time)
                            w_traj_durs[encode_map[wid]-4] += duration
                        # Remove the widget from open_widgets
                        del open_widgets[wid]
                        
                # Cleanup any still open widgets, using the end time of the trajectory as the end time for that widget, append duration 
                for wid, enter_time in open_widgets.items():
                    if end_times[i] >= enter_time:
                        duration = end_times[i] - max(start_times[i], enter_time)
                        w_traj_durs[encode_map[wid]-4] += duration

                # Combine presence and durations in one row
                row = [user, sess, start_times[i], end_times[i]]
                row += (w_traj_durs > 0).astype(int).tolist()  # presence
                row += w_traj_durs.tolist()                     # durations

                total_rows.append(row)

    return total_rows, all_widgets


traj_t = pd.read_csv('../../features/traditional/traj_edge_times.csv')

total_rows, all_widgets = hot_encode_with_durations(
    traj_t,
    WIDGET_ROOT
)

columns = ['User_ID', 'session', 'start', 'end'] + list(all_widgets) + [str(w)+'_dur' for w in all_widgets]

out = pd.DataFrame(total_rows, columns=columns)
out.to_csv('widget_features.csv', index=False)

t1 = time.time()
print(t1 - t0)
