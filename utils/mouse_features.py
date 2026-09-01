import warnings
from math import sqrt, atan, pi
import numpy as np


def get_sess_index(idx, sess_inds):
    i = 1
    for index in sess_inds:
        if int(idx) < index:
            break
        i += 1
    return i


def trajectories(x_list, y_list, t_list, sess_list):
    base_index = 0
    trajs_x, trajs_y, trajs_t, trajs_s = [], [], [], []

    for i in range(len(t_list) - 1):
        # split trajectory when gap >= 100
        if (t_list[i + 1] - t_list[i]) >= 100:
            traj_x = x_list[base_index:i + 1]
            traj_y = y_list[base_index:i + 1]
            traj_t = t_list[base_index:i + 1]
            traj_s = sess_list[base_index:i + 1]

            if len(traj_t) >= 3:
                trajs_x.append(traj_x)
                trajs_y.append(traj_y)
                trajs_t.append(traj_t)
                trajs_s.append(traj_s)

            base_index = i + 1

    # handle the last segment after the final gap
    if base_index < len(t_list):
        traj_x = x_list[base_index:]
        traj_y = y_list[base_index:]
        traj_t = t_list[base_index:]
        traj_s = sess_list[base_index:]

        if len(traj_t) >= 3:
            trajs_x.append(traj_x)
            trajs_y.append(traj_y)
            trajs_t.append(traj_t)
            trajs_s.append(traj_s)

    return trajs_x, trajs_y, trajs_t, trajs_s

def traveled_distance(traj_x, traj_y):
    dk_list_2pts = []
    dk_list_traj = []
    for i in range(len(traj_x)):
        dk_traj = sqrt((traj_y[i][-1] - traj_y[i][0]) ** 2 + (traj_x[i][-1] - traj_x[i][0]) ** 2)
        dk_list_traj.append(dk_traj)
        traj_dk_2pts = []
        for j in range(len(traj_x[i]) - 1):
            dk_2pts = sqrt((traj_y[i][j + 1] - traj_y[i][j]) ** 2 + (traj_x[i][j + 1] - traj_x[i][j]) ** 2)
            traj_dk_2pts.append(dk_2pts)
        dk_list_2pts.append(traj_dk_2pts)
    return dk_list_2pts, dk_list_traj


def duration(traj_t):
    dur_list = []
    for traj in traj_t:
        dur = traj[-1] - traj[0]
        dur_list.append(dur)
    return dur_list


def curve_len(traj_x, traj_y):
    sm_list = []
    for i in range(len(traj_x)):
        sm = 0
        for j in range(len(traj_x[i]) - 1):
            sm += sqrt((traj_y[i][j + 1] - traj_y[i][j]) ** 2 + (traj_x[i][j + 1] - traj_x[i][j]) ** 2)
        sm_list.append(sm)
    return sm_list


def angle_movement(traj_x, traj_y):
    am_list_2pts = []
    am_list_traj = []
    for i in range(len(traj_x)):
        traj_am_2pts = []
        if traj_x[i][-1] != traj_x[i][0]:
            am_traj = atan((traj_y[i][-1] - traj_y[i][0]) / (traj_x[i][-1] - traj_x[i][0]))
            am_list_traj.append(am_traj)
        elif traj_x[i][-1] == traj_x[i][0] and traj_y[i][-1] > traj_y[i][0]:
            am_traj = (3 * pi) / 2
            am_list_traj.append(am_traj)
        elif traj_x[i][-1] == traj_x[i][0] and traj_y[i][-1] < traj_y[i][0]:
            am_traj = pi / 2
            am_list_traj.append(am_traj)
        elif traj_y[i][-1] == traj_y[i][0] and traj_x[i][-1] < traj_x[i][0]:
            am_list_traj.append(pi)
        else:
            am_list_traj.append(2 * pi)
        for j in range(len(traj_x[i]) - 1):
            if traj_x[i][j + 1] != traj_x[i][j]:
                am_2pts = atan((traj_y[i][j + 1] - traj_y[i][j]) / (traj_x[i][j + 1] - traj_x[i][j]))
                traj_am_2pts.append(am_2pts)
            elif traj_x[i][j + 1] == traj_x[i][j] and traj_y[i][j + 1] > traj_y[i][j]:
                am_2pts = (3 * pi) / 2
                traj_am_2pts.append(am_2pts)
            elif traj_x[i][j + 1] == traj_x[i][j] and traj_y[i][j + 1] < traj_y[i][j]:
                am_2pts = pi / 2
                traj_am_2pts.append(am_2pts)
            elif traj_y[i][j + 1] == traj_y[i][j] and traj_x[i][j + 1] < traj_x[i][j]:
                traj_am_2pts.append(pi)
            else:
                traj_am_2pts.append(2 * pi)
        am_list_2pts.append(traj_am_2pts)
    return am_list_traj, am_list_2pts


def velocity_traj(sm_list, durs):
    v_list = []
    for i in range(len(sm_list)):
        if durs[i] > 0:
            if len(sm_list) == 0:
                print("sm_list length is zero")
            if durs[i] == 0:
                print("duration is zero")
            v = (sm_list[i] / durs[i])
            v_list.append(v)
    return v_list


def velocity_pt(dk_list, traj_t):
    v_list = []
    for i in range(len(traj_t)):
        traj_v_pt = []
        for j in range(len(traj_t[i]) - 1):
            v = dk_list[i][j] / (traj_t[i][j + 1] - traj_t[i][j])
            traj_v_pt.append(v)
        v_list.append(traj_v_pt)
    return v_list


def hv_velocity(traj_x, traj_y, traj_t):
    hv_list_pt = []
    hv_list_traj = []
    vv_list_pt = []
    vv_list_traj = []
    for i in range(len(traj_x)):
        hv_traj_pt = []
        vv_traj_pt = []

        hv_traj = (traj_x[i][-1] - traj_x[i][0]) / (traj_t[i][-1] - traj_t[i][0])
        hv_list_traj.append(hv_traj)

        vv_traj = (traj_y[i][-1] - traj_y[i][0]) / (traj_t[i][-1] - traj_t[i][0])
        vv_list_traj.append(vv_traj)

        for j in range(len(traj_x[i]) - 1):
            try:
                hv_pt = (traj_x[i][j + 1] - traj_x[i][j]) / (traj_t[i][j + 1] - traj_t[i][j])
                hv_traj_pt.append(hv_pt)
            except:
                print(traj_t[i][j + 1], traj_t[i][j])
            vv_pt = (traj_y[i][j + 1] - traj_y[i][j]) / (traj_t[i][j + 1] - traj_t[i][j])
            vv_traj_pt.append(vv_pt)
        hv_list_pt.append(hv_traj_pt)
        vv_list_pt.append(vv_traj_pt)
    return hv_list_traj, hv_list_pt, vv_list_traj, vv_list_pt


def hv_acceleration(traj_x, traj_y, traj_t):
    ha_list_pt = []
    ha_list_traj = []
    va_list_pt = []
    va_list_traj = []
    for i in range(len(traj_x)):
        ha_traj_pt = []
        va_traj_pt = []
        mid = (len(traj_x[i]) // 2)

        ha_traj = (traj_x[i][-1] - (2 * traj_x[i][mid]) + traj_x[i][0]) / ((traj_t[i][-1] - traj_t[i][mid]) ** 2)
        ha_list_traj.append(ha_traj)

        va_traj = (traj_y[i][-1] - (2 * traj_y[i][mid]) + traj_y[i][0]) / ((traj_t[i][-1] - traj_t[i][mid]) ** 2)
        va_list_traj.append(va_traj)
        for j in range(len(traj_x[i]) - 2):
            ha_pt = (traj_x[i][j + 2] - (2 * traj_x[i][j + 1]) + traj_x[i][j]) / (
                    (traj_t[i][j + 2] - traj_t[i][j + 1]) ** 2)
            ha_traj_pt.append(ha_pt)
            va_pt = (traj_y[i][j + 2] - (2 * traj_y[i][j + 1]) + traj_y[i][j]) / (
                    (traj_t[i][j + 2] - traj_t[i][j + 1]) ** 2)
            va_traj_pt.append(va_pt)

        ha_list_pt.append(ha_traj_pt)
        va_list_pt.append(va_traj_pt)
    return ha_list_traj, ha_list_pt, va_list_traj, va_list_pt


def minimums(x):
    return [np.min(sublist) for sublist in x]


def maximums(x):
    return [np.max(sublist) for sublist in x]


def means(x):
    return [np.mean(sublist) for sublist in x]
