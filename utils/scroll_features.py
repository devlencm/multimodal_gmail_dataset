import warnings
from math import sqrt, atan, pi
from statistics import mean
import numpy as np


def get_sess_index(idx, sess_inds):
    i = 1
    for index in sess_inds:
        if int(idx) < index:
            break
        i += 1
    return i


def trajectories(y_list, t_list):
    base_index = 0
    trajs_y = []
    trajs_t = []

    for i in range(len(t_list) - 1):

        if (t_list[i + 1] - t_list[i]) >= 100:

            traj_y = y_list[base_index:i+1]
            traj_t = t_list[base_index:i+1]

            if len(traj_t) >= 3:
                trajs_y.append(traj_y)
                trajs_t.append(traj_t)

            base_index = i + 1

    # Add final trajectory
    traj_y = y_list[base_index:]
    traj_t = t_list[base_index:]

    if len(traj_t) >= 3:
        trajs_y.append(traj_y)
        trajs_t.append(traj_t)

    return trajs_y, trajs_t


def traveled_distance(traj_y):
    traj_dists = [sum(traj) for traj in traj_y]
    return traj_dists


def duration(traj_t):
    dur_list = []
    for traj in traj_t:
        dur = traj[-1] - traj[0]
        dur_list.append(dur)
    return dur_list


def curve_len(traj_x, traj_y):
    sm_list = []
    sm = 0
    for i in range(len(traj_x)):
        for j in range(len(traj_x[i]) - 1):
            sm += sqrt((traj_y[i][j + 1] - traj_y[i][j]) ** 2 + (traj_x[i][j + 1] - traj_x[i][j]) ** 2)
        sm_list.append(sm)
    return sm_list


def velocity_traj(sm_list, durs):
    v_list = []
    for i in range(len(sm_list)):
        if durs[i] > 0:
            v = (sm_list[i] / durs[i]) / len(sm_list)
            v_list.append(v)
    return v_list


def velocity(traj_ys, traj_ts):
    traj_ds = [traj_y[-1] - traj_y[0] for traj_y in traj_ys]
    traj_dts = [traj_t[-1] - traj_t[0] for traj_t in traj_ts]
    v_list = [traj_ds[i] / traj_dts[i] for i in range(len(traj_ds))]
    return v_list


# Calculates acceleration based on the first half of the trajectory vs the second half.
def acceleration(traj_ys, traj_ts):
    traj_delta_start_ys = [traj_y[len(traj_y)//2] - traj_y[0] for traj_y in traj_ys]
    traj_delta_end_ys = [traj_y[-1] - traj_y[len(traj_y) // 2] for traj_y in traj_ys]
    traj_delta_start_times = [traj_t[len(traj_t)//2] - traj_t[0] for traj_t in traj_ts]
    traj_delta_end_times = [traj_t[-1] - traj_t[len(traj_t) // 2] for traj_t in traj_ts]
    v_start_list = [traj_delta_start_ys[i] / traj_delta_start_times[i] for i in range(len(traj_delta_start_times))]
    v_end_list = [traj_delta_end_ys[i] / traj_delta_end_times[i] for i in range(len(traj_delta_end_times))]
    a_list = [v_end_list[i] - v_start_list[i] / traj_delta_end_times[i] for i in range(len(traj_delta_end_times))]
    return a_list


def durations(traj_ts):
    return [traj_t[-1] - traj_t[0] for traj_t in traj_ts]


def v_acceleration(traj_ys, traj_ts):
    traj_accs = [(traj_ys[i][-1] - (2 * traj_ys[i][len(traj_ys[i])//2]) + traj_ys[i][0]) / ((traj_ts[i][-1] - traj_ts[i][len(traj_ts[i])//2]) ** 2) for i in range(len(traj_ts))]
    return traj_accs


def skewness(x):
    warnings.filterwarnings('ignore')
    sk_list = []
    # warnings.filterwarnings('ignore')
    for i in range(len(x)):
        sk = skew(x[i])
        sk_list.append(sk)
    return sk_list


def kurt(x):
    warnings.filterwarnings('ignore')
    k_list = []
    for i in range(len(x)):
        k = kurtosis(x[i])
        k_list.append(k)
    return k_list


def minimums(x):
    m_list = []
    for i in range(len(x)):
        m = min(x[i])
        m_list.append(m)
    return m_list


def maximums(x):
    m_list = []
    for i in range(len(x)):
        m = max(x[i])
        m_list.append(m)
    return m_list


def means(x):
    m_list = []
    for i in range(len(x)):
        m = mean(x[i])
        m_list.append(m)
    return m_list

