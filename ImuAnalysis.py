#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy import signal
from scipy import integrate
from scipy.stats import variation
import statsmodels.tsa.api as smt
import glob

result_log = 'result.log'


def freq():
    return 100


def read_data_oiseau(datasheet):
    """读取监测系统导出的文件"""
    gait = pd.read_csv(datasheet)
    t = gait[['timeCounter']].to_numpy()[:].reshape(-1)
    t = t / freq()

    euler = gait[['EulerX', 'EulerY', 'EulerZ']].to_numpy()[:]
    acc = gait[['AccX', 'AccY', 'AccZ']].to_numpy()[:]
    gyro = gait[['GyroX', 'GyroY', 'GyroZ']].to_numpy()[:]

    gait_data = {'t': t, 'euler': euler, 'acc': acc, 'gyro': gyro}
    return gait_data


def acc_rot_euler(euler: np.ndarray, acc: np.ndarray):
    """用欧拉角将加速度转化为世界坐标系（经纬地坐标系）"""
    n = euler.shape[0]
    new_acc = np.empty((n, 3))

    for i in range(0, n):
        r = R.from_euler('xyz', euler[i, :], degrees=True)
        rot_mat = r.as_matrix()
        new_acc[i, :] = rot_mat @ acc[i, :]
        # new_acc[i, :] = acc[i, :] @ rot_mat
    return new_acc


def acc_rot_euler_body_ref(euler: np.ndarray, body: np.ndarray, acc: np.ndarray):
    """暂时不用"""
    n = euler.shape[0]
    new_acc = np.empty((n, 3))
    for i in range(0, n):
        r = R.from_euler('xyz', euler[i, :], degrees=True)
        rot_mat = r.as_matrix()
        # new_acc[i, :] = rot_mat @ acc[i, :]
        tmp = rot_mat @ acc[i, :]
        tmp = tmp - np.array([0, 0, 9.9])
        r2 = R.from_euler('zyx', body, degrees=True)
        rot_mat2 = r2.as_matrix()
        new_acc[i, :] = rot_mat2 @ tmp
        # new_acc[i, :] = acc[i, :] @ rot_mat
    return new_acc


def find_period(gyro: np.ndarray):
    """自相关方法，计算大概的周期"""
    fs = freq()
    N = gyro.shape[0]
    acf = smt.stattools.acf(gyro, nlags=N - 1, adjusted=True)

    pks, _ = signal.find_peaks(acf)

    # 短周期
    short = np.diff(pks).mean()
    pkl, _ = signal.find_peaks(acf, height=0.3, distance=short)  # 0.3

    long = pkl[0]
    return long / fs
    # return 1.5


def find_to_hs(gyro: np.ndarray, t, period):
    """截取每一步。to: toe_off, hs: heel_strike, pk_sw: peak of swing phase"""
    gyro = gyro.reshape(-1)

    ang_v_max = gyro.max()
    pk_sw, _ = signal.find_peaks(gyro, 80)  # height=0.4 * ang_v_max)  # 0.4
    n_sw = pk_sw.shape[0]

    # 去除相邻峰值  此方法没有考虑多个（两个以上）峰值扎堆的情况，但是滤波后的数据通常以两个峰为主
    del_pk = np.ones((n_sw,))
    # period = 1.3
    for i in range(0, n_sw - 1):
        if t[pk_sw[i + 1]] - t[pk_sw[i]] < 0.5 * period:
            if gyro[pk_sw[i + 1]] > gyro[pk_sw[i]]:
                del_pk[i] = 0
            else:
                del_pk[i + 1] = 0
    pk_sw = pk_sw[del_pk > 0]
    n_sw = pk_sw.shape[0]

    plt.figure()
    plt.plot(t, gyro)
    # plt.plot(t[:-1], np.diff(gyro)*5)
    plt.plot(t[pk_sw], gyro[pk_sw], 'o')

    to_locs = np.zeros((n_sw,)).astype(int)
    hs_locs = np.zeros((n_sw,)).astype(int)
    to_idx = 1
    hs_idx = 0
    to, _ = signal.find_peaks(-gyro[0:pk_sw[0]], height=0)
    if to.size != 0:
        to_locs[0] = to[-1]
        flag_first = 0
    else:
        print('first to not found')
        flag_first = 1

    s_flag = 1
    for i in range(0, n_sw - 1):
        start_point = pk_sw[i]
        end_point = pk_sw[i + 1]

        tmp, _ = signal.find_peaks(-gyro[start_point:end_point], height=20)
        if s_flag:
            if tmp.size != 0:
                hs_tmp = tmp[0]
                to_tmp = tmp[-1]
                if tmp.size > 2 and gyro[start_point + tmp[0]] - gyro[start_point + tmp[1]] > 50 > tmp[1] - tmp[0]:
                    hs_tmp = tmp[1]
                if hs_tmp == to_tmp:
                    if hs_tmp < (end_point - start_point) / 2:
                        s_flag = 0
                        hs_locs[hs_idx] = hs_tmp + start_point
                        to_locs[to_idx] = 0
                    else:
                        hs_locs[hs_idx] = 0
                        to_locs[hs_idx] = 0
                        to_locs[to_idx] = to_tmp + start_point
                else:
                    hs_locs[hs_idx] = hs_tmp + start_point
                    to_locs[to_idx] = to_tmp + start_point

                hs_idx = hs_idx + 1
                to_idx = to_idx + 1
            else:
                print('tmp not found')
        else:
            if tmp.size != 0:
                to_tmp = tmp[-1]
                hs_locs[hs_idx] = 0
                to_locs[to_idx] = to_tmp + start_point
                hs_idx = hs_idx + 1
                to_idx = to_idx + 1
                s_flag = 1
            else:
                print('tmp not found')
    hs, _ = signal.find_peaks(-gyro[pk_sw[-1]:], height=0)
    if hs.size != 0:
        hs_locs[hs_idx] = hs[0] + pk_sw[-1]
    else:
        print('last hs not found')
        to_locs[-1] = 0

    pk_sw = pk_sw[flag_first:]
    to_locs = to_locs[flag_first:]
    hs_locs = hs_locs[flag_first:]
    # to_locs = to_locs[to_locs > 0]
    # hs_locs = hs_locs[hs_locs > 0]

    n_stride = hs_locs.shape[0]
    hs_locs_real = np.zeros((n_stride, 1)).astype(int)

    for i in range(0, n_stride):
        start_point = pk_sw[i]
        end_point = hs_locs[i]
        tmp = np.where(gyro[start_point:end_point] >= 0)[0] + 1
        hs_locs_real[i] = start_point + tmp[-1] if tmp.shape[0] else 0

    # hs_locs_real = hs_locs_real[hs_locs_real > 0]

    delete_rows = []
    # for i in range(n_stride):
    #     if (gyro[to_locs[i]] > gyro[hs_locs[i]] and gyro[to_locs[i]] > -50) \
    #             or (to_locs[i] <= 0 or hs_locs_real[i] <= 0 or hs_locs[i] <= 0):
    #         delete_rows.append(i)

    to_locs = np.delete(to_locs, delete_rows, axis=0)
    hs_locs = np.delete(hs_locs, delete_rows, axis=0)
    hs_locs_real = np.delete(hs_locs_real, delete_rows, axis=0)

    plt.plot(t[to_locs], gyro[to_locs], 'o')
    plt.plot(t[hs_locs], gyro[hs_locs], 'o')
    plt.plot(t[hs_locs_real], np.zeros(hs_locs_real.shape), 'o')
    plt.minorticks_on()
    plt.grid(which='both')
    # plt.show()

    to_locs = to_locs.reshape((-1, 1))
    hs_locs = hs_locs.reshape((-1, 1))
    hs_locs_real = hs_locs_real.reshape((-1, 1))
    idx_to_hs = np.hstack((to_locs, hs_locs, hs_locs_real))  # test

    return idx_to_hs


def rampp_offset(acc: np.ndarray):
    n = acc.shape[0]
    kn = max(int(0.05 * n), 1)
    ln = max(int(0.05 * n), 1)

    y0 = np.mean(acc[0:kn])
    y1 = np.mean(acc[n - ln:])
    offset1 = np.ones(kn) * y0
    offset3 = np.ones(ln) * y1
    offset2 = np.linspace(y0, y1, n - kn - ln)
    # offset2_sig = 1/(1+np.exp(-offset2))
    offset = np.concatenate((offset1, offset2, offset3))
    return offset


def rampp_offset_n0(acc):
    n = acc.shape[0]
    y0 = acc[0]
    y1 = acc[-1]
    offset = np.linspace(y0, y1, n)
    return offset


def find_ddv_boundary(acc):
    n = acc.shape[0]
    kn = max(0, 1)
    ln = max(0, 1)
    da = np.diff(acc)
    dda = np.diff(da)
    flat_arr = np.where((dda > 0.03) | (dda < -0.03))[0]
    if flat_arr.shape[0] > 10:
        kn = max(flat_arr[0], kn)
        ln = min(flat_arr[-1], ln)
    return kn, n - ln


def ddv_offset(acc: np.ndarray):
    # TODO: 修改两端，使得效果更佳
    n = acc.shape[0]
    kn = max(int(0.05 * n), 1)
    ln = max(int(0.05 * n), 1)
    da = np.diff(acc)
    dda = np.diff(da)
    ddv_threshold = 0.03
    meanN = max(kn, ln)
    flat_arr = np.where((dda > ddv_threshold) | (dda < -ddv_threshold))[0]
    if flat_arr.shape[0] > 10:
        kn = max(flat_arr[0], 0)
        ln = max(n-2-flat_arr[-1]-1, 0)

    # acc_max = np.argmax(acc)
    # acc_min = np.argmin(acc)
    # neg_pks, _ = signal.find_peaks(-acc[:acc_max])
    # kn = neg_pks[-1]
    # pos_pks, _ = signal.find_peaks(acc[acc_min:])
    # ln = pos_pks[0]


    # y0 = np.mean(acc[0:kn])
    # y1 = np.mean(acc[n - ln:])
    # y0 = acc[kn]
    # y1 = acc[n - ln]
    y0 = np.mean(acc[kn:kn+meanN])
    y1 = np.mean(acc[n-ln-meanN:n-ln])
    offset11 = np.ones(meanN) * y0
    offset33 = np.ones(meanN) * y1
    offset1 = acc[0:kn]
    offset3 = acc[n-ln:]
    offset2 = np.linspace(y0, y1, n - kn - ln-2*meanN)
    # sigmoid = np.linspace(-10, 50, n - kn - ln-2*meanN)
    # offset2 = (y1 - y0) / (1 + np.exp(-sigmoid)) + y0
    offset = np.concatenate((offset1, offset11, offset2, offset33, offset3))
    # offset = np.concatenate((offset1, offset2, offset3))
    return offset


def rampp_offset_v(vel: np.ndarray, n_size=5):
    n = vel.shape[0]
    y1 = np.mean(vel[n - n_size:])
    offset1 = np.linspace(0, y1, n - n_size)
    offset2 = np.ones(n_size) * y1
    offset = np.concatenate((offset1, offset2))
    return offset


def rampp_offset_sig(vel: np.ndarray):
    n = vel.shape[0]
    i_max = np.argmax(vel[:int(n * 0.66)])
    i_min_1 = np.argmin(vel[i_max:]) + i_max
    ng_pk, _ = signal.find_peaks(-vel[i_max:])
    dv = np.diff(vel)
    ddv = np.diff(dv)

    if ng_pk.size != 0:
        i_min_2 = ng_pk[0] + i_max
    else:
        i_min_2 = n - 1
    i_min = min(i_min_1, i_min_2)

    if i_min >= n - 3:  # ddv--->n-3 dv->n-2 v->n-1
        idx_end_ddv = i_min
    else:
        ddv_min = np.argmin(ddv[i_min:]) + i_min
        idx_tmp = np.where(ddv[ddv_min:] > 0)[0]
        if idx_tmp.size != 0:
            idx_end_ddv = int(idx_tmp[0] / 2) + ddv_min  # tmp/2
        else:
            idx_end_ddv = min(5 + ddv_min, n - 1)  # 5

    idx_start, _ = signal.find_peaks(-dv[0:i_max])
    # idx_start, _ = signal.find_peaks(-vel[0:i_max])
    dv_max, _ = signal.find_peaks(dv[0:i_max])

    idx_end, _ = signal.find_peaks(vel[i_min:])
    if idx_start.size != 0:
        start_point = idx_start[-1]
    elif dv_max.size != 0:
        start_point = np.argmin(dv[0:dv_max[0]])
    else:
        start_point = 0
    if idx_end.size != 0:
        # end_point = idx_end[-1] + i_min
        end_point = idx_end_ddv
    else:
        end_point = idx_end_ddv

    sigmoid = np.linspace(-15, 10, end_point - start_point)
    offset_sig_val = (vel[end_point - 1] - vel[start_point]) / (1 + np.exp(-sigmoid)) + vel[start_point]

    offset_1 = np.linspace(0, vel[start_point - 1], start_point)
    offset_2 = np.linspace(vel[start_point], vel[end_point - 1], end_point - start_point)
    offset_3 = np.linspace(vel[end_point], vel[-1], n - end_point)
    offset = np.concatenate((offset_1, offset_sig_val, offset_3))
    return offset


def offset_sig(vec: np.ndarray):
    n = vec.shape[0]
    window_size = 5
    start_point = 0
    start_vec = vec[:window_size]
    start_val = np.mean(start_vec)
    end_point = n
    end_vec = vec[n - 1 - window_size:]
    end_val = np.mean(end_vec)
    sigmoid = np.linspace(-10, 10, end_point - start_point)
    offset = (end_val - start_val) / (1 + np.exp(-sigmoid)) + start_val

    return offset


def y_offset_sig(acc: np.ndarray):
    n = acc.shape[0]

    mid = int(n / 2)
    da = np.diff(acc)
    dda = np.diff(da)
    sig = (0.01 - 0.005)
    threshold = max(0.003, 0.001 + 0.01 * (dda.max() - dda.min()))
    # threshold = 0.014
    print(threshold, dda.max() - dda.min(), n)
    start_point = 0
    end_point = n
    window_size = 10
    for i in range(mid, n - window_size - 2):
        if np.all(dda[i:i + window_size] < threshold) and np.all(dda[i:i + window_size] > -threshold):
            end_point = i
            break

    for i in range(mid, window_size, -1):
        if np.all(dda[i - window_size:i] < threshold) and np.all(dda[i - window_size:i] > -threshold):
            start_point = i
            break

    sigmoid = np.linspace(-10, 10, end_point - start_point)
    offset = (acc[end_point - 1] - acc[start_point]) / (1 + np.exp(-sigmoid)) + acc[start_point]

    return np.concatenate((acc[:start_point], offset, acc[end_point:]))


def rampp_offset_s(s: np.ndarray):
    n = s.shape[0]
    ln = max(int(0.02 * n), 1)
    y1 = np.mean(s[n - ln:])
    offset1 = np.linspace(0, y1, n - ln)
    offset2 = np.ones(ln) * y1

    offset = np.concatenate((offset1, offset2))
    return offset


def find_ms(gyro: np.ndarray):
    n_win = 20
    n = gyro.shape[0]
    if n_win > n:
        idx = int(n / 2)
    else:
        energy = np.zeros(n - n_win + 1)
        for i in range(0, n - n_win + 1):
            energy[i] = np.linalg.norm(gyro[i:i + n_win])
        idx = np.argmin(energy) + int(n_win / 2)

    return idx


def find_ms_acc(acc_x: np.ndarray, acc_y: np.ndarray):
    n_win = 15
    n = acc_x.shape[0]

    acc = np.concatenate((acc_x, acc_y), axis=0)
    acc = acc.reshape((2, -1))
    if n_win > n:
        idx = int(n / 2)
    else:
        energy = np.zeros(n - n_win + 1)
        for i in range(0, n - n_win + 1):
            acc_i = acc[:, i:i + n_win] - np.mean(acc[:, i:i + n_win], axis=1).reshape(2, 1)
            energy[i] = np.linalg.norm(acc_i)
        idx = np.argmin(energy) + int(n_win / 2)

    return idx


def find_ms_dacc(acc_x: np.ndarray, acc_y: np.ndarray):
    n_win = max(int(0.15 * acc_x.shape[0]), 15)
    n = acc_x.shape[0] - 1
    acc = np.concatenate((np.diff(acc_x), np.diff(acc_y)), axis=0)
    acc = acc.reshape((2, -1))
    if n_win > n:
        idx = int(n / 2)
    else:
        energy = np.zeros(n - n_win + 1)
        for i in range(0, n - n_win + 1):
            acc_i = acc[:, i:i + n_win] - np.mean(acc[:, i:i + n_win], axis=1).reshape(2, 1)
            energy[i] = np.linalg.norm(acc_i)
        idx = np.argmin(energy) + int(n_win / 2)

    return idx


def find_ms_aw(acc_x: np.ndarray, acc_y: np.ndarray, gyro: np.ndarray):
    n_win = 15
    n = acc_x.shape[0]
    acc_x = acc_x - acc_x.mean()
    acc_y = acc_y - acc_y.mean()
    weight = 0.1
    if n_win > n:
        idx = int(n / 2)
    else:
        energy = np.zeros(n - n_win + 1)
        for i in range(0, n - n_win + 1):
            tmp1 = np.linalg.norm(acc_x[i:i + n_win]) ** 2 + np.linalg.norm(acc_y[i:i + n_win]) ** 2
            tmp2 = np.linalg.norm(gyro[i:i + n_win]) ** 2
            energy[i] = weight * tmp1 + (1 - weight) * tmp2
        idx = np.argmin(energy) + int(n_win / 2)

    return idx


def idx_weight(weight, idx_for_g, idx_for_a):
    idx_for = int(weight * idx_for_g + (1 - weight) * idx_for_a)
    return idx_for


def de_glitch(acc):
    acc_med = signal.medfilt(acc, 5)
    # da = np.diff(acc)
    # glitch_loc = np.where((da > 10) | (da < -10))[0] + 1
    # glitch_intervals = np.split(glitch_loc, np.where(np.diff(glitch_loc) > 3)[0] + 1)
    # glitch_intervals = [interval for interval in glitch_intervals if len(interval) > 0]
    # for interval in glitch_intervals:
    #     start_point = interval[0]
    #     end_point = interval[-1]
    #     n_interval = end_point - start_point + 1
    #     acc[start_point:end_point+1] = acc_med[start_point:end_point+1]
    return acc_med


def gait_param(acc, gyro_xyz, t, euler):
    """计算所需要的一条腿上的步态参数，可进行补充
        stride, speed, cycle_time, spt_ratio,
        phase_time, clearance, idx_to_hs/freq(),
        period, y_Ang
    """
    acc_x = acc[:, 0]
    acc_y = acc[:, 1]
    acc_z = acc[:, 2]
    acc_norm = np.linalg.norm(acc, axis=1)
    gyro = gyro_xyz[:, 1]
    gyro = gyro.reshape(-1)
    gyro_norm = np.linalg.norm(gyro_xyz, ord=2, axis=1)
    gyro_world = acc_rot_euler(euler, gyro_xyz)
    gyro_z_world = gyro_world[:, 2].reshape(-1)
    fs = freq()

    # 滤波
    wc = 2 * 6 / fs  # 截至频率6hz
    b, a = signal.butter(4, wc, 'low')

    z_filtered = signal.filtfilt(b, a, acc_z)
    # x_filtered = signal.filtfilt(b, a, acc_x)
    # y_filtered = signal.filtfilt(b, a, acc_y)
    # x_filtered = de_glitch(acc_x)
    # y_filtered = de_glitch(acc_y)
    x_filtered = signal.filtfilt(b, a, de_glitch(acc_x))
    y_filtered = signal.filtfilt(b, a, de_glitch(acc_y))
    acc_norm_filtered = signal.filtfilt(b, a, acc_norm)
    period = find_period(gyro)

    gyro_filtered = signal.filtfilt(b, a, gyro)
    gyro_norm_filtered = signal.filtfilt(b, a, gyro_norm)
    gyro_z_world_filtered = signal.filtfilt(b, a, gyro_z_world)
    idx_to_hs = find_to_hs(gyro_filtered, t, period)
    n_idx = idx_to_hs.shape[0]
    stride = np.zeros((n_idx, 1))
    stride_chara = np.zeros((n_idx, 2))
    '''mid_sw_time, is_turned'''
    speed = np.zeros((n_idx, 1))
    phase_time = np.zeros((n_idx, 3))
    clearance = np.zeros((n_idx, 1))
    y_Ang = np.zeros((n_idx, 4))
    '''max, min, toAng, hsAng'''

    s_idx = 0
    p_idx = 0

    fig, ax = plt.subplots(4, 1)
    fig.tight_layout(pad=2)
    ax_x = ax[0]
    ax_y = ax[1]
    ax_z = ax[2]
    ax_all = ax[3]
    ax_x.plot(t, x_filtered)
    ax_x.grid(which='both', axis='both')
    ax_x.set_title('X')
    ax_x.minorticks_on()
    ax_y.plot(t, y_filtered)
    ax_y.grid(which='both', axis='both')
    ax_y.set_title('Y')
    ax_y.minorticks_on()
    ax_z.plot(t, z_filtered)
    ax_z.grid(which='both', axis='both')
    ax_z.set_title('Z')
    ax_z.minorticks_on()
    ax_all.plot(t, acc_norm_filtered)
    ax_all.grid(which='both', axis='both')
    ax_all.set_title('All')
    ax_all.minorticks_on()

    end_point_1 = -1
    end_point_2 = -1
    for i in range(0, n_idx):
        if i == 0:
            start_point_tmp = 0
        else:
            start_point_tmp = idx_to_hs[i - 1, 1]

        if i == n_idx - 1:
            end_point_tmp = gyro.shape[0] - 1
        else:
            end_point_tmp = idx_to_hs[i + 1, 0]

        to_point = idx_to_hs[i, 0]
        hs_point = idx_to_hs[i, 1]
        is_dacc = False
        if not is_dacc:
            if i == 0 or hs_point - start_point_tmp > 1.5 * period * freq():
                period_tmp = period * freq()
                # if hs_point - to_point > 0.5 * period:
                #     period_tmp = 2 * (hs_point - to_point)
                back_offset = int(0.8 * period_tmp)  # 0.8 1.2
                idx_back_g = find_ms(gyro_norm_filtered[hs_point - back_offset:hs_point])
                idx_back_a = find_ms_acc(x_filtered[hs_point - back_offset:hs_point],
                                         y_filtered[hs_point - back_offset:hs_point])
                idx_back_1 = idx_weight(0.85, idx_back_g, idx_back_a)  # 1
                start_point_1 = idx_back_1 + hs_point - back_offset

                ax_x.plot(t[start_point_1], x_filtered[start_point_1], 'o', color='b')
                ax_y.plot(t[start_point_1], y_filtered[start_point_1], 'o', color='b')
            else:
                start_point_1 = end_point_1
                start_point_2 = end_point_2  # 2
            if end_point_tmp - to_point > 1.5 * period * freq() or i == n_idx - 1:
                end_point_tmp = to_point + int(0.8 * period * freq())  # 0.8

            idx_for_g = find_ms(gyro_norm_filtered[hs_point:end_point_tmp])
            idx_for_a = find_ms_acc(x_filtered[hs_point:end_point_tmp], y_filtered[hs_point:end_point_tmp])
            idx_for_1 = idx_weight(1, idx_for_g, idx_for_a)  # 1
            end_point_1 = idx_for_1 + hs_point
            idx_for_2 = idx_weight(0.85, idx_for_g, idx_for_a)
            end_point_2 = idx_for_2 + hs_point
            test1 = x_filtered[start_point_1] ** 2 + y_filtered[start_point_1] ** 2
            test2 = x_filtered[end_point_1] ** 2 + y_filtered[end_point_1] ** 2
            test = (test1 - test2) / max(test1, test2)
            sq_diff = (x_filtered[start_point_1] - x_filtered[end_point_1]) ** 2 + \
                      (y_filtered[start_point_1] - y_filtered[end_point_1]) ** 2

            # if abs(x_filtered[end_point_1] - x_filtered[start_point_1]) > 0.45 \
            #         or abs(y_filtered[end_point_1] - y_filtered[start_point_1]) > 0.45 \
            #         or abs(test) > 0.95 \
            #         or sq_diff > 1:
            #     # or abs(variation(x_filtered[end_point_2 - 10:end_point_2 + 10])) > 2 \
            #     # or abs(variation(y_filtered[end_point_2 - 10:end_point_2 + 10])) > 2:
            #     if abs(variation(x_filtered[end_point_2 - 10:end_point_2 + 10])) > 10 \
            #             or abs(variation(y_filtered[end_point_2 - 10:end_point_2 + 10])) > 10:
            #         # or np.std(x_filtered[end_point_2 - 10:end_point_2 + 10]) > 0.7 \
            #         # or np.std(y_filtered[end_point_2 - 10:end_point_2 + 10]) > 0.7:
            #         idx_for_2 = idx_weight(0.25, idx_for_g, idx_for_a)
            #         end_point_2 = idx_for_2 + hs_point
            #         print(str(i) + ' end2 10')
            #     elif abs(variation(x_filtered[end_point_2 - 10:end_point_2 + 10])) > 5 \
            #             or abs(variation(y_filtered[end_point_2 - 10:end_point_2 + 10])) > 5:
            #         # or np.std(x_filtered[end_point_2 - 10:end_point_2 + 10]) > 0.5 \
            #         # or np.std(y_filtered[end_point_2 - 10:end_point_2 + 10]) > 0.5:
            #         idx_for_2 = idx_weight(0.5, idx_for_g, idx_for_a)
            #         end_point_2 = idx_for_2 + hs_point
            #         print(str(i) + ' end2 5')
            #     end_point = end_point_2
            #     start_point = start_point_2  # point2
            #     print('!!!' + str(i))
            # else:
            #     end_point = end_point_1
            #     start_point = start_point_1
            end_point = end_point_1
            start_point = start_point_1
        else:
            if i == 0 or hs_point - start_point_tmp > 1.5 * period * freq():
                period_tmp = period * freq()
                if hs_point - to_point > 0.5 * period:
                    period_tmp = 2 * (hs_point - to_point)
                back_offset = int(1.2 * period_tmp)  # 0.8 1.2
                idx_back_g = find_ms(gyro_norm_filtered[hs_point - back_offset:hs_point])
                idx_back_a = find_ms_acc(x_filtered[to_point - back_offset:hs_point],
                                         y_filtered[to_point - back_offset:hs_point])

                idx_back_1 = idx_weight(0.85, idx_back_g, idx_back_a)
                start_point_1 = idx_back_1 + hs_point - back_offset

                ax_x.plot(t[start_point_1], x_filtered[start_point_1], 'o', color='b')
                ax_y.plot(t[start_point_1], y_filtered[start_point_1], 'o', color='b')
                ax_z.plot(t[start_point_1], z_filtered[start_point_1], 'o', color='b')
                ax_all.plot(t[start_point_1], acc_norm_filtered[start_point_1], 'o', color='b')
            else:
                start_point_1 = end_point_1
            if end_point_tmp - to_point > 1.5 * period * freq() or i == n_idx - 1:
                end_point_tmp = to_point + int(0.9 * period * freq())  # 0.8

            idx_for_g = find_ms(gyro_norm_filtered[hs_point:end_point_tmp])
            idx_for_a = find_ms_dacc(x_filtered[hs_point:end_point_tmp], y_filtered[hs_point:end_point_tmp])
            idx_for_3 = find_ms_aw(x_filtered[hs_point:end_point_tmp],
                                   y_filtered[hs_point:end_point_tmp],
                                   gyro_norm_filtered[hs_point:end_point_tmp])
            idx_for_1 = idx_weight(0.85, idx_for_g, idx_for_a)  # 1
            end_point_1 = idx_for_1 + hs_point
            end_point = end_point_1
            start_point = start_point_1
        if t[hs_point] - t[to_point] < period:
            # start_point, end_point = find_ddv_boundary(x_filtered[start_point:end_point]) + start_point
            # x_offset = rampp_offset_n0(x_filtered[start_point:end_point])
            # y_offset = rampp_offset_n0(y_filtered[start_point:end_point])
            x_offset = ddv_offset(x_filtered[start_point:end_point])
            y_offset = ddv_offset(y_filtered[start_point:end_point])
            # x_offset = rampp_offset(x_filtered[start_point:end_point])
            # y_offset = rampp_offset(y_filtered[start_point:end_point])
            z_offset = rampp_offset(z_filtered[start_point:end_point])
            g_offset = rampp_offset(-gyro_filtered[start_point:end_point])
            gyro_z_offset = rampp_offset(gyro_z_world_filtered[start_point:end_point])

            ax_x.plot(t[start_point:end_point], x_filtered[start_point:end_point], 'r', zorder=2)
            ax_x.plot(t[end_point], x_filtered[end_point], 'o', zorder=4, markersize='5', alpha=0.5)
            ax_y.plot(t[start_point:end_point], y_filtered[start_point:end_point], 'r', zorder=2)
            ax_y.plot(t[end_point], y_filtered[end_point], 'o', zorder=4, markersize='5', alpha=0.5)
            ax_z.plot(t[start_point:end_point], z_filtered[start_point:end_point], 'r', zorder=2)
            ax_z.plot(t[end_point], z_filtered[end_point], 'o', zorder=4, markersize='5', alpha=0.5)
            ax_all.plot(t[start_point:end_point], acc_norm_filtered[start_point:end_point], 'r', zorder=2)
            ax_all.plot(t[end_point], acc_norm_filtered[end_point], 'o', zorder=4, markersize='5', alpha=0.5)

            s_vel_x = integrate.cumulative_trapezoid(x_filtered[start_point:end_point] - x_offset,
                                                     t[start_point:end_point], initial=0)
            s_vel_y = integrate.cumulative_trapezoid(y_filtered[start_point:end_point] - y_offset,
                                                     t[start_point:end_point], initial=0)
            s_vel_z = integrate.cumulative_trapezoid(z_filtered[start_point:end_point] - z_offset,
                                                     t[start_point:end_point], initial=0)
            x_offset_v = offset_sig(s_vel_x)
            y_offset_v = offset_sig(s_vel_y)
            # y_offset_v = s_vel_y
            # x_offset_v = rampp_offset(s_vel_x)
            # y_offset_v = rampp_offset(s_vel_y)
            z_offset_v = rampp_offset_v(s_vel_z)
            s_len_x = integrate.cumulative_trapezoid(s_vel_x - x_offset_v, t[start_point:end_point], initial=0)
            s_len_y = integrate.cumulative_trapezoid(s_vel_y - y_offset_v, t[start_point:end_point], initial=0)
            s_len_z = integrate.cumulative_trapezoid(s_vel_z - z_offset_v, t[start_point:end_point], initial=0)
            # plt.figure()
            # plt.plot(t[start_point:end_point], x_filtered[start_point:end_point])
            # plt.plot(t[start_point:end_point], x_filtered[start_point:end_point]-x_offset)
            # plt.show()
            # theta = np.arctan((s_vel_y-y_offset_v)/(s_vel_x-x_offset_v))*180/np.pi
            # theta2 = euler_reset(theta, 120)
            theta_y = integrate.cumulative_trapezoid(-gyro_filtered[start_point:end_point] - g_offset,
                                                     t[start_point:end_point], initial=0)
            ty_offset = rampp_offset_sig(theta_y)

            turning_ang = integrate.cumulative_trapezoid(gyro_z_world_filtered[start_point:end_point] - gyro_z_offset,
                                                         t[start_point:end_point], initial=0)

            y_Ang[s_idx, 0] = np.max(theta_y - ty_offset)
            y_Ang[s_idx, 1] = np.min(theta_y - ty_offset)
            # y_Ang[s_idx, 2] = (theta_y-ty_offset)[idx_to_hs[i, 0]-start_point]
            # y_Ang[s_idx, 3] = (theta_y - ty_offset)[idx_to_hs[i, 2] - start_point]
            stride[s_idx] = (s_len_x[-1] ** 2 + s_len_y[-1] ** 2) ** 0.5
            # stride[s_idx] = abs(s_len_x[-1])
            # if i == 0:
            #     speed[s_idx] = stride[s_idx] / (t[end_point] - t[start_point])
            # else:
            if i and t[idx_to_hs[i, 2]] - t[idx_to_hs[i - 1, 2]] < 1.5 * period:
                speed[s_idx] = stride[s_idx] / (t[idx_to_hs[i, 2]] - t[idx_to_hs[i - 1, 2]])
            else:
                speed[s_idx] = stride[s_idx] / (t[end_point] - t[start_point])
            stride_chara[s_idx, 0] = t[to_point]
            stride_chara[s_idx, 1] = abs(turning_ang[-1] - turning_ang[0]) < 20
            s_idx = s_idx + 1
        else:
            print('notice: abnormal gait period')

        if i < n_idx - 1 and t[idx_to_hs[i + 1, 2]] - t[idx_to_hs[i, 2]] < 1.5 * period:
            phase_time[p_idx, 0] = t[idx_to_hs[i + 1, 2]] - t[idx_to_hs[i, 2]]  # hs->hs 这样实际上计算的是下一步的时间参数
            phase_time[p_idx, 1] = t[idx_to_hs[i + 1, 0]] - t[idx_to_hs[i, 2]]  # stance
            phase_time[p_idx, 2] = t[idx_to_hs[i + 1, 2]] - t[idx_to_hs[i + 1, 0]]  # swing
            p_idx = p_idx + 1
    # plt.show()  # plt.show()导致每一步显示一次图像
    # plt.close(fig)
    stride = stride[stride > 0]
    n_stride = stride.shape[0]
    stride_chara = stride_chara[0:n_stride, :]  # not used
    speed = speed[speed > 0]
    clearance = clearance.reshape(-1, )
    y_Ang = y_Ang.reshape((-1, 4))
    phase_time = phase_time[phase_time > 0].reshape((-1, 3))
    cycle_time = phase_time[:, 0]
    spt_ratio = phase_time[:, 1] / phase_time[:, 0]

    return stride, speed, cycle_time, spt_ratio, phase_time, clearance, idx_to_hs / freq(), \
        period, y_Ang, stride_chara


def si_param(param_l, param_r):
    return abs(param_l - param_r) * 2 / (param_l + param_r)


def cv_param(param: np.ndarray):
    """传入的参数统一计算CV, 返回绝对值"""
    param = param.reshape(-1)
    return abs(variation(param))


def cv_param_list(param_list: list):
    cv_p = []
    for param in param_list:
        cv_p.append(cv_param(param))
    return np.array(cv_p)


def step_param(to_hs, period, stride):
    to_hs_l = to_hs[0]
    to_hs_r = to_hs[1]
    n_l = to_hs_l.shape[0]
    n_r = to_hs_l.shape[0]
    stride_l = stride[0]
    stride_r = stride[2]
    stride_to_l = stride[1][:, 0]
    stride_to_r = stride[3][:, 0]
    ds_duration = []
    step_time = []
    swing_time = []
    stride_length = []
    si_swing_time = []
    lr_flag = to_hs_l[0, 0] > to_hs_r[0, 0]
    period = np.array(period)
    p_mean = period.mean()
    for i in range(0, n_l - 1):
        hs_l = to_hs_l[i, 2]
        to_l = to_hs_l[i + 1, 0]
        bool_1 = to_hs_r[:, 0] > hs_l
        bool_2 = to_hs_r[:, 0] < to_l
        bool_to = bool_1 & bool_2
        to_ds_idx = np.where(bool_to)[0]
        to_ds = to_hs_r[to_ds_idx, 0]
        if to_ds_idx.size == 1 and to_ds - hs_l < 0.8 * p_mean:
            ds_duration.append(to_ds - hs_l)
            if to_ds_idx != 0 and hs_l - to_hs_r[to_ds_idx - 1, 2] < 0.8 * p_mean:
                step_time.append(hs_l - to_hs_r[to_ds_idx - 1, 2])  # l-r
                if lr_flag:
                    tmp = to_hs_r[to_ds_idx - 1, 2] - to_hs_r[to_ds_idx - 1, 0]
                    swing_time.append([hs_l - to_hs_l[i, 0], tmp[0]])
                    idx_s_l = np.where(stride_to_l == to_hs_l[i, 0])[0]
                    idx_s_r = np.where(stride_to_r == to_hs_r[to_ds_idx - 1, 0])[0]
                    if idx_s_l.size == 1 and idx_s_r.size == 1:
                        stride_length.append([stride_l[idx_s_l][0], stride_r[idx_s_r][0]])
        bool_3 = to_hs_r[:, 2] > hs_l
        bool_4 = to_hs_r[:, 2] < to_l
        bool_hs = bool_3 & bool_4
        hs_ds_idx = np.where(bool_hs)[0]
        hs_ds = to_hs_r[hs_ds_idx, 2]
        if hs_ds_idx.size == 1 and to_l - hs_ds < 0.8 * p_mean:
            ds_duration.append(to_l - hs_ds)
            step_time.append(hs_ds - hs_l)  # r-l
            if not lr_flag and hs_ds_idx != n_r - 1:
                tmp = to_hs_l[hs_ds_idx, 2] - to_hs_r[hs_ds_idx, 0]
                swing_time.append([hs_l - to_hs_l[i, 0], tmp[0]])
                idx_s_l = np.where(stride_to_l == to_hs_l[i, 0])[0]
                idx_s_r = np.where(stride_to_r == to_hs_r[hs_ds_idx, 0])[0]
                if idx_s_l.size == 1 and idx_s_r.size == 1:
                    stride_length.append([stride_l[idx_s_l], stride_r[idx_s_r]])

    ds_duration = np.array(ds_duration)
    step_time = np.array(step_time)
    stride_length = np.array(stride_length)
    swing_time = np.array(swing_time)
    return ds_duration, step_time, stride_length, swing_time


def initResultFile(text=''):
    with open(result_log, 'w') as f:
        print('-' * 50, file=f)
        print(text, file=f)
        print('-' * 50, file=f)


def printArrResult(name, arr=np.array([])):
    with open(result_log, 'a') as f:
        if arr.size != 0:
            print('Para:', name, file=f)
            # arr_str = np.array2string(arr, separator='\n')
            if type(arr) == np.ndarray:
                for row in arr:
                    print(row, file=f)
            else:
                print(arr, file=f)

        else:
            print(name, file=f)


def imu_analysis(csvPath):
    # path1 = r'C:\\Users\\13372\\Documents\\课程\\csvData\*'
    print(csvPath)
    print('now Tw Ta')

    global result_log
    result_log = csvPath + '\\result.log'

    searchDir = csvPath + '\\*'
    file_l = glob.glob(searchDir + 'left' + '*.csv')
    file_r = glob.glob(searchDir + 'right' + '*.csv')

    time_l = file_l[0].split('\\')[-1].split('_left_')[-1].split('_192')[0]
    time_r = file_r[0].split('\\')[-1].split('_right_')[-1].split('_192')[0]

    name_l = file_l[0].split('\\')[-1].split('_')[0]
    name_r = file_r[0].split('\\')[-1].split('_')[0]
    # if time_l != time_r or name_l != name_r:
    #     print("not matched!")
    #     return

    file = file_l + file_r
    lor = ['left', 'right']

    to_hs = []
    period = []
    stride = []
    """stride_l, character_l, stride_r, character_r"""
    duration = []
    initResultFile(name_l + ' ' + time_l)
    for i in range(0, len(file)):
        printArrResult('\n' + '-' * 22 + lor[i] + '-' * 22 + '\n', )
        datasheet = file[i]
        gait_data = read_data_oiseau(datasheet)
        euler = gait_data.get('euler')
        acc = acc_rot_euler(gait_data.get('euler'), gait_data.get('acc'))
        t = gait_data.get('t')
        gyro_xyz = gait_data.get('gyro')
        if i == 0 or i == 1 or i == 2:
            gyro_xyz[:, 1] = -gyro_xyz[:, 1]

        gait_result = gait_param(acc[:], gyro_xyz[:], t[:], euler)
        to_hs.append(gait_result[6])
        period.append(gait_result[7])

        stride_length = gait_result[0]
        stride.append(stride_length)
        printArrResult('stride length', stride_length)
        printArrResult('')
        stride_chara = gait_result[9]
        stride.append(stride_chara)

        phase_time = gait_result[4]
        duration.append(phase_time)
        printArrResult('phase time', phase_time)

        gait_event = gait_result[6]
        printArrResult('gait_event', gait_event)
        # print('clearance:')
        # clearance = gait_result[5]

        speed = gait_result[1]
        printArrResult('speed', speed)

        angle = gait_result[8][:, 0:2]
        printArrResult('angle', angle)

        printArrResult('-' * 50)
    plt.show()
    ds_duration, step_time, stride_length_paired, swing_time_paired = step_param(to_hs, period, stride)

    cadence = 60 / step_time.mean()
    printArrResult('cadence', cadence)

    si_sw = si_param(swing_time_paired[:, 0], swing_time_paired[:, 1]).mean()
    si_sl = si_param(stride[0].mean(), stride[2].mean())
    printArrResult('si_sw', si_sw)
    printArrResult('si_sl', si_sl)

    cv_ds = cv_param(ds_duration)
    sw_all = [duration[0][:, 2], duration[1][:, 2]]
    cv_sw = cv_param_list(sw_all)
    stride_length_all = [stride[0], stride[2]]
    cv_sl = cv_param_list(stride_length_all)
    cd_all = [duration[0][:, 0], duration[1][:, 0]]
    cv_cd = cv_param_list(cd_all)
    printArrResult('cv_ds', cv_ds)
    printArrResult('cv_sw', cv_sw)
    printArrResult('cv_sl ', cv_sl)
    printArrResult('cv_cd: ', cv_cd)
