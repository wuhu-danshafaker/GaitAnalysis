import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy import signal
from scipy import integrate
from scipy.stats import variation
import statsmodels.tsa.api as smt
import glob
import timeit


def freq():
    return 120


def read_data_xsens(datasheet, skip_rows):
    """读取xsens导出的文件，skip_rows根据实际情况有所不同"""
    gait = pd.read_csv(datasheet, skiprows=skip_rows)
    t = gait[['PacketCounter']].to_numpy()
    t = t / freq()

    euler = gait[['Euler_X', 'Euler_Y', 'Euler_Z']].to_numpy()
    if skip_rows == 6:
        acc = gait[['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
    else:
        acc = gait[['FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z']].to_numpy()
    gyro = gait[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
    gait_data = {'t': t, 'euler': euler, 'acc': acc, 'gyro': gyro}
    return gait_data


def acc_rot(quat: np.ndarray, acc: np.ndarray):
    n = quat.shape[0]
    new_acc = np.empty((n, 3))
    for i in range(0, n):
        r = R.from_quat(quat[i, :])  # x y z w
        rot_mat = r.as_matrix()
        new_acc[i, :] = rot_mat @ acc[i, :]
        # new_acc[i, :] = acc[i, :] @ rot_mat
    return new_acc * 9.8


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


def double_integral_line(x0, y0, x1, y1):
    k = (y1 - y0) / (x1 - x0)
    b = (x1 * y0 - y1 * x0) / (x1 - x0)
    c = -(0.5 * k * x0 ** 2 + b * x0)
    return 1 / 6 * k * (x1 ** 3 - x0 ** 3) + 1 / 2 * b * (x1 ** 2 - x0 ** 2) + c * (x1 - x0)


def find_period(gyro: np.ndarray):
    """自相关方法，计算大概的周期"""
    fs = freq()
    N = gyro.shape[0]
    acf = smt.stattools.acf(gyro, nlags=N - 1, adjusted=True)
    # plt.plot(acf)
    # plt.show()

    pks, _ = signal.find_peaks(acf)
    # plt.plot(pks, acf[pks], 'o')

    # 短周期
    short = np.diff(pks).mean()
    # print(short)
    pkl, _ = signal.find_peaks(acf, height=0.3, distance=short)  # 0.3
    # plt.plot(pkl, acf[pkl], 'o')
    # plt.plot(acf)
    # plt.show()
    if pkl.size > 1:
        long = np.diff(pkl).mean()
    # else:
    long = pkl[0]
    # print(long)
    return long / fs


def find_to_hs(gyro: np.ndarray, t, period):
    """截取每一步。to: toe_off, hs: heel_strike, pk_sw: peak of swing phase"""
    gyro = gyro.reshape(-1)

    ang_v_max = gyro.max()
    pk_sw, _ = signal.find_peaks(gyro, height=0.4 * ang_v_max)  # 0.4
    n_sw = pk_sw.shape[0]

    # 去除相邻峰值  此方法没有考虑多个（两个以上）峰值扎堆的情况，但是滤波后的数据通常以两个峰为主
    del_pk = np.ones((n_sw,))
    period = 1.3
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

    to_locs = np.zeros((n_sw, 1)).astype(int)
    hs_locs = np.zeros((n_sw, 1)).astype(int)
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
        if i==8:
            print('1')
        tmp, _ = signal.find_peaks(-gyro[start_point:end_point], height=20)
        if s_flag:
            if tmp.size != 0:
                hs_tmp = tmp[0]
                to_tmp = tmp[-1]
                # if tmp.size > 2 and (gyro[start_point+tmp[0]]-gyro[start_point+tmp[1]] > 50 or gyro[start_point+tmp[1]]/gyro[start_point+tmp[0]]>0.7) and tmp[1]-tmp[0]<50:
                if tmp.size > 2 and gyro[start_point + tmp[0]] - gyro[start_point + tmp[1]] > 50 and tmp[1] - tmp[0] < 50:
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
    to_locs = to_locs[to_locs > 0]
    hs_locs = hs_locs[hs_locs > 0]

    n_stride = hs_locs.shape[0]
    hs_locs_real = np.zeros((n_stride, 1)).astype(int)

    for i in range(0, n_stride):
        start_point = pk_sw[i]
        end_point = hs_locs[i]
        tmp = np.where(gyro[start_point:end_point] >= 0)[0] + 1
        hs_locs_real[i] = start_point + tmp[-1]

    hs_locs_real = hs_locs_real[hs_locs_real > 0]

    plt.plot(t[to_locs], gyro[to_locs], 'o')
    plt.plot(t[hs_locs], gyro[hs_locs], 'o')
    plt.plot(t[hs_locs_real], np.zeros(hs_locs_real.shape), 'o')
    plt.minorticks_on()
    plt.grid(which='both')
    plt.show()

    # print(to_locs/120)
    # print(hs_locs/120)
    # print(hs_locs_real/120)

    to_locs = to_locs.reshape((-1, 1))
    hs_locs = hs_locs.reshape((-1, 1))
    hs_locs_real = hs_locs_real.reshape((-1, 1))
    idx_to_hs = np.hstack((to_locs, hs_locs, hs_locs_real))  # test

    return idx_to_hs


def find_offset2zero(acc: np.ndarray):
    bin_a = acc

    hist, bin_edges = np.histogram(bin_a, range=(-1, 1), bins='auto')
    idx_hist = np.argmax(hist)
    # idx_major = np.where(hist > hist.sum() * 0.3)[0]
    # # if idx_major.size == 0 or idx_major.max()-idx_major.min() >= idx_major.shape[0]:
    # if idx_major.size == 0:
    #     idx_major = np.array([np.argmax(hist)])

    tmp = bin_a
    tmp = tmp[tmp > bin_edges[idx_hist]]
    tmp = tmp[tmp < bin_edges[idx_hist + 1]]
    vec_major = tmp

    var = variation(vec_major)
    # print(var)
    if abs(var) < 0.5 and vec_major.size != 0:
        offset = vec_major.mean()
        # fig, ax = plt.subplots()
        # ax.hist(bin_edges[:-1], bin_edges, weights=hist, density=True)
        # fig.show()
    else:
        offset = 0
    # fig, ax = plt.subplots()
    # ax.hist(bin_edges[:-1], bin_edges, weights=hist, density=True)
    # fig.show()
    # print(offset)

    return offset


def find_offset2zero_longSQ(acc: np.ndarray):
    bin_a = acc
    hist, bin_edges = np.histogram(bin_a, bins='auto')
    idx_hist = np.argmax(hist)
    idx_major = np.where(hist > hist.sum() * 0.25)[0]
    if idx_major.size == 0:
        idx_major = np.array([idx_hist])
    tmp = bin_a
    tmp = tmp[tmp > bin_edges[idx_major[0]]]
    tmp = tmp[tmp < bin_edges[idx_major[-1] + 1]]
    vec_major = tmp

    offset = vec_major.mean()
    # fig, ax = plt.subplots()
    # ax.hist(bin_edges[:-1], bin_edges, weights=hist, density=True)
    # fig.show()
    return offset


def rampp_offset(acc: np.ndarray):
    n = acc.shape[0]
    kn = max(int(0.04 * n), 1)
    ln = max(int(0.02 * n), 1)

    y0 = np.mean(acc[0:kn])
    y1 = np.mean(acc[n - ln:])
    offset1 = np.ones(kn) * y0
    offset3 = np.ones(ln) * y1
    offset2 = np.linspace(y0, y1, n - kn - ln)
    # offset2_sig = 1/(1+np.exp(-offset2))
    offset = np.concatenate((offset1, offset2, offset3))
    return offset


def rampp_offset_v(vel: np.ndarray):
    n = vel.shape[0]
    y1 = np.mean(vel[n - 5:])
    offset1 = np.linspace(0, y1, n - 5)
    offset2 = np.ones(5) * y1
    offset = np.concatenate((offset1, offset2))
    return offset


def rampp_offset_sig(vel: np.ndarray):
    n = vel.shape[0]
    i_max = np.argmax(vel[:int(n*0.66)])
    i_min_1 = np.argmin(vel[i_max:]) + i_max
    ng_pk, _ = signal.find_peaks(-vel[i_max:])
    dv = np.diff(vel)
    ddv = np.diff(dv)
    # fig = plt.figure()
    # plt.plot(np.linspace(0, n, n), vel / 10)
    # plt.plot(np.linspace(0, n - 1, n - 1), dv)
    # plt.plot(np.linspace(0, n - 2, n - 2), ddv * 10)
    # plt.minorticks_on()
    # plt.grid(which='both')

    if ng_pk.size != 0:
        i_min_2 = ng_pk[0] + i_max
    else:
        i_min_2 = n-1
    i_min = min(i_min_1, i_min_2)

    if i_min >= n-3:  # ddv--->n-3 dv->n-2 v->n-1
        idx_end_ddv = i_min
    else:
        ddv_min = np.argmin(ddv[i_min:]) + i_min
        idx_tmp = np.where(ddv[ddv_min:] > 0)[0]
        if idx_tmp.size != 0:
            idx_end_ddv = int(idx_tmp[0]/2) + ddv_min  # tmp/2
        else:
            idx_end_ddv = min(5 + ddv_min, n-1)  # 5

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
    # plt.plot([start_point, end_point], [vel[start_point]/10, vel[end_point]/10], 'o')
    # plt.show()
    sigmoid = np.linspace(-15, 10, end_point-start_point)
    offset_sig = (vel[end_point-1] - vel[start_point]) / (1 + np.exp(-sigmoid)) + vel[start_point]

    offset_1 = np.linspace(0, vel[start_point-1], start_point)
    offset_2 = np.linspace(vel[start_point], vel[end_point-1], end_point-start_point)
    offset_3 = np.linspace(vel[end_point], vel[-1], n-end_point)
    offset = np.concatenate((offset_1, offset_sig, offset_3))
    return offset


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
    # acc_norm = np.linalg.norm(acc, axis=0)
    # acc_x1 = acc_x - acc_x.mean()
    # acc_y1 = acc_y - acc_y.mean()
    if n_win > n:
        idx = int(n / 2)
    else:
        energy = np.zeros(n - n_win + 1)
        for i in range(0, n - n_win + 1):
            # energy[i] = (np.linalg.norm(acc_x1[i:i + n_win]) ** 2 + np.linalg.norm(acc_y1[i:i + n_win]) ** 2) ** 0.5
            acc_i = acc[:, i:i + n_win] - np.mean(acc[:, i:i + n_win], axis=1).reshape(2, 1)
            # acc_norm_i = np.linalg.norm(acc_i, axis=0)
            # tt = np.linalg.norm(acc_norm_i)
            # ttt = np.linalg.norm(acc_i)
            # energy[i] = np.linalg.norm(acc_norm[i:i + n_win])
            energy[i] = np.linalg.norm(acc_i)
        idx = np.argmin(energy) + int(n_win / 2)

    return idx


def find_ms_aw(acc_x: np.ndarray, acc_y: np.ndarray, gyro: np.ndarray):
    n_win = 15
    n = acc_x.shape[0]
    acc_x = acc_x - acc_x.mean()
    acc_y = acc_y - acc_y.mean()
    weight = 0.1
    # acc = np.concatenate((acc_x, acc_y), axis=0)
    # acc_norm = np.linalg.norm(acc, axis=0)
    if n_win > n:
        idx = int(n / 2)
    else:
        energy = np.zeros(n - n_win + 1)
        for i in range(0, n - n_win + 1):
            tmp1 = np.linalg.norm(acc_x[i:i + n_win]) ** 2 + np.linalg.norm(acc_y[i:i + n_win]) ** 2
            tmp2 = np.linalg.norm(gyro[i:i + n_win]) ** 2
            energy[i] = weight*tmp1 + (1-weight)*tmp2
        idx = np.argmin(energy) + int(n_win / 2)

    return idx


def find_ms_acc_3d(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray):
    n_win = 15
    n = acc_x.shape[0]
    acc_x = acc_x - acc_x.mean()
    acc_y = acc_y - acc_y.mean()
    acc_z = acc_z - acc_z.mean()
    # acc = np.concatenate((acc_x, acc_y), axis=0)
    # acc_norm = np.linalg.norm(acc, axis=0)
    if n_win > n:
        idx = int(n / 2)
    else:
        energy = np.zeros(n - n_win + 1)
        for i in range(0, n - n_win + 1):
            energy[i] = (np.linalg.norm(acc_x[i:i + n_win]) ** 2 + np.linalg.norm(acc_y[i:i + n_win]) ** 2 +
                         np.linalg.norm(acc_z[i:i + n_win]) ** 2) ** 0.5
        idx = np.argmin(energy) + int(n_win / 2)

    return idx


def idx_weight(weight, idx_for_g, idx_for_a):
    idx_for = int(weight * idx_for_g + (1 - weight) * idx_for_a)
    return idx_for


def euler_reset(euler: np.ndarray, ang=300):
    euler = euler.reshape(-1,)
    euler_diff = np.diff(euler)
    down_flag = np.where(euler_diff < -ang)[0]
    up_flag = np.where(euler_diff > ang)[0]
    flag = np.concatenate((down_flag, up_flag))
    n_down = down_flag.shape[0]
    n_up = up_flag.shape[0]
    idx = np.argsort(flag)
    flag_sorted = np.sort(flag)
    flag_sort = idx < n_down
    i = 0
    if ang > 200:
        ran = 360
    else:
        ran = 180
    while i < n_down+n_up:
        if i < n_down+n_up-1:
            if flag_sort[i] != flag_sort[i+1]:
                if flag_sort[i]:
                    euler[flag_sorted[i]+1:flag_sorted[i+1]+1] = euler[flag_sorted[i]+1:flag_sorted[i+1]+1] + 360
                else:
                    euler[flag_sorted[i]+1:flag_sorted[i+1]+1] = euler[flag_sorted[i]+1:flag_sorted[i+1]+1] - 360
                i = i + 2
            else:
                if flag_sort[i]:
                    euler[flag_sorted[i]+1:] = euler[flag_sorted[i]+1:] + ran
                else:
                    euler[flag_sorted[i]+1:] = euler[flag_sorted[i]+1:] - ran
                i = i + 1
        else:
            if flag_sort[i]:
                euler[flag_sorted[i]+1:] = euler[flag_sorted[i]+1:] + ran
            else:
                euler[flag_sorted[i]+1:] = euler[flag_sorted[i]+1:] - ran
            i = i + 1
    return euler


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
    # euler[:, 2] = euler_reset(euler[:, 2], 201)
    # 滤波
    wc = 2 * 6 / fs  # 截至频率6hz
    b, a = signal.butter(4, wc, 'low')
    x_filtered = signal.filtfilt(b, a, acc_x)
    y_filtered = signal.filtfilt(b, a, acc_y)
    z_filtered = signal.filtfilt(b, a, acc_z)
    acc_norm_filtered = signal.filtfilt(b, a, acc_norm)
    period = find_period(gyro)
    period = 1.4
    gyro_filtered = signal.filtfilt(b, a, gyro)
    gyro_norm_filtered = signal.filtfilt(b, a, gyro_norm)
    gyro_z_world_filtered = signal.filtfilt(b, a, gyro_z_world)
    idx_to_hs = find_to_hs(gyro_filtered, t, period)
    n_idx = idx_to_hs.shape[0]
    # fig_g = plt.figure()
    # plt.plot(t, gyro_filtered)
    # plt.plot(t, 10*z_filtered)
    # plt.grid(which='both', axis='both')
    # plt.minorticks_on()
    # plt.show()
    stride = np.zeros((n_idx, 1))
    stride_chara = np.zeros((n_idx, 2))
    '''mid_sw_time, is_turned'''
    speed = np.zeros((n_idx, 1))
    phase_time = np.zeros((n_idx, 3))
    '''total, stance, swing'''
    phase_time2 = np.zeros((n_idx, 3))
    clearance = np.zeros((n_idx, 1))
    y_Ang = np.zeros((n_idx, 4))
    '''max, min, toAng, hsAng'''

    s_idx = 0
    p_idx = 0

    fig, ax = plt.subplots(2, 1)
    fig.tight_layout(pad=2)
    ax_x = ax[0]
    ax_y = ax[1]
    # ax_z = ax[2]
    ax_x.plot(t, x_filtered)
    ax_x.grid(which='both', axis='both')
    ax_x.set_title('X')
    ax_x.minorticks_on()
    ax_y.plot(t, y_filtered)
    ax_y.grid(which='both', axis='both')
    ax_y.set_title('Y')
    ax_y.minorticks_on()
    # ax_z.plot(t, z_filtered)
    # ax_z.grid(which='both', axis='both')
    # ax_z.set_title('Z')
    # ax_z.minorticks_on()

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
            next_to = idx_to_hs[i + 1, 0]

        to_point = idx_to_hs[i, 0]
        hs_point = idx_to_hs[i, 1]
        if i == 0 or hs_point - start_point_tmp > 1.5 * period * freq():
            # idx_back_arr, _ = signal.find_peaks(gyro_filtered[start_point_tmp:to_point])
            # idx_back = idx_back_arr[-1]
            period_tmp = period
            if hs_point - to_point > 0.5*period:
                period_tmp = 2*(hs_point-to_point)/freq()
            back_offset = int(0.8 * period_tmp * freq())  # 0.8 1.2
            idx_back_g = find_ms(gyro_norm_filtered[hs_point - back_offset:hs_point])
            idx_back_a = find_ms_acc(x_filtered[hs_point - back_offset:hs_point],
                                     y_filtered[hs_point - back_offset:hs_point])

            idx_back_3 = find_ms_aw(x_filtered[hs_point-back_offset:hs_point],
                                   y_filtered[hs_point-back_offset:hs_point],
                                   gyro_norm_filtered[hs_point-back_offset:hs_point])
            idx_back_1 = idx_weight(1, idx_back_g, idx_back_a)
            # idx_back_1 = idx_back_3
            # idx_back_a = idx_back_g
            start_point_1 = idx_back_1 + hs_point - back_offset
            idx_back_2 = idx_weight(0.85, idx_back_g, idx_back_a)
            # idx_back_2 = idx_back_1
            start_point_2 = idx_back_2 + hs_point - back_offset
            if np.std(x_filtered[start_point_2 - 10:start_point_2 + 10]) > 0.4 \
                    or np.std(y_filtered[start_point_2 - 10:start_point_2 + 10]) > 0.4:
                idx_back_2 = idx_weight(0.5, idx_back_g, idx_back_a)
                start_point_2 = idx_back_2 + hs_point - back_offset
            #     print(str(i) + ' start2')
            # print('s ' + str(np.std(x_filtered[start_point_2 - 10:start_point_2 + 10])) + ' ' + str(
            #     np.std(y_filtered[start_point_2 - 10:start_point_2 + 10])))
            # start_point_1 = idx_back + start_point_tmp
            # start_point_2 = start_point_1
            ax_x.plot(t[start_point_1], x_filtered[start_point_1], 'o', color='b')
            ax_y.plot(t[start_point_1], y_filtered[start_point_1], 'o', color='b')
            # ax_z.plot(t[start_point_1], z_filtered[start_point_1], 'o', color='b')
        else:
            start_point_1 = end_point_1
            start_point_2 = end_point_2  # 2
        if end_point_tmp - to_point > 1.5 * period * freq() or i == n_idx - 1:
            end_point_tmp = to_point + int(0.9 * period * freq())  # 0.8

            # idx_for_t_arr, _ = signal.find_peaks(gyro_filtered[start_point_tmp:to_point])
            # idx_for_t = idx_for_t_arr[0]
            # print('tmp '+str(i))

        idx_for_g = find_ms(gyro_norm_filtered[hs_point:end_point_tmp])
        idx_for_a = find_ms_acc(x_filtered[hs_point:end_point_tmp], y_filtered[hs_point:end_point_tmp])
        idx_for_3 = find_ms_aw(x_filtered[hs_point:end_point_tmp],
                               y_filtered[hs_point:end_point_tmp],
                               gyro_norm_filtered[hs_point:end_point_tmp])
        # idx_for_a = find_ms_acc_3d(x_filtered[hs_point:end_point_tmp], y_filtered[hs_point:end_point_tmp], z_filtered[hs_point:end_point_tmp])
        idx_for_1 = idx_weight(1, idx_for_g, idx_for_a)  # 1
        # idx_for_a = idx_for_g
        # idx_for_1 = idx_for_3
        end_point_1 = idx_for_1 + hs_point
        idx_for_2 = idx_weight(0.85, idx_for_g, idx_for_a)
        # idx_for_2 = idx_for_1
        end_point_2 = idx_for_2 + hs_point
        test1 = x_filtered[start_point_1] ** 2 + y_filtered[start_point_1] ** 2
        test2 = x_filtered[end_point_1] ** 2 + y_filtered[end_point_1] ** 2
        test = (test1 - test2) / max(test1, test2)
        sq_diff = (x_filtered[start_point_1]-x_filtered[end_point_1])**2 +\
                  (y_filtered[start_point_1]-y_filtered[end_point_1])**2
        # print('  ' + str(i) + ' ' + str(np.std(x_filtered[end_point_2 - 10:end_point_2 + 10])) + ' '
        #       + str(np.std(y_filtered[end_point_2 - 10:end_point_2 + 10])) + ' '
        #       + str(variation(x_filtered[end_point_2 - 10:end_point_2 + 10])) + ' '
        #       + str(variation(y_filtered[end_point_2 - 10:end_point_2 + 10])) + ' '
        #       + str(sq_diff)+' '+str(x_filtered[end_point_1] - x_filtered[start_point_1])+' '+str(y_filtered[end_point_1] - y_filtered[start_point_1]))

        if abs(x_filtered[end_point_1] - x_filtered[start_point_1]) > 0.45\
                or abs(y_filtered[end_point_1] - y_filtered[start_point_1]) > 0.45 \
                or abs(test) > 0.95 \
                or sq_diff > 1:
                # or abs(variation(x_filtered[end_point_2 - 10:end_point_2 + 10])) > 2 \
                # or abs(variation(y_filtered[end_point_2 - 10:end_point_2 + 10])) > 2:
            if abs(variation(x_filtered[end_point_2 - 10:end_point_2 + 10])) > 10 \
                    or abs(variation(y_filtered[end_point_2 - 10:end_point_2 + 10])) > 10:
                    # or np.std(x_filtered[end_point_2 - 10:end_point_2 + 10]) > 0.7 \
                    # or np.std(y_filtered[end_point_2 - 10:end_point_2 + 10]) > 0.7:
                idx_for_2 = idx_weight(0.25, idx_for_g, idx_for_a)
                end_point_2 = idx_for_2 + hs_point
                print(str(i) + ' end2 10')
            elif abs(variation(x_filtered[end_point_2 - 10:end_point_2 + 10])) > 5 \
                    or abs(variation(y_filtered[end_point_2 - 10:end_point_2 + 10])) > 5:
                    # or np.std(x_filtered[end_point_2 - 10:end_point_2 + 10]) > 0.5 \
                    # or np.std(y_filtered[end_point_2 - 10:end_point_2 + 10]) > 0.5:
                idx_for_2 = idx_weight(0.5, idx_for_g, idx_for_a)
                end_point_2 = idx_for_2 + hs_point
                print(str(i) + ' end2 5')
            end_point = end_point_2
            start_point = start_point_2  # point2
            print('!!!' + str(i))
            # print('  ' + str(np.std(x_filtered[end_point - 10:end_point + 10])) + ' '
            #       + str(np.std(y_filtered[end_point - 10:end_point + 10])) + ' '
            #       + str(variation(x_filtered[end_point_2 - 10:end_point_2 + 10])) + ' '
            #       + str(variation(y_filtered[end_point_2 - 10:end_point_2 + 10])))
        else:
            end_point = end_point_1
            start_point = start_point_1
            # print('  '+str(np.std(x_filtered[end_point - 10:end_point + 10])) + ' ' + str(
            #     np.std(y_filtered[end_point - 10:end_point + 10])))
        # print(str(i) + ' ' + str(x_filtered[end_point_1]) + ' ' + str(y_filtered[end_point_1]) + ' ' + str(
        #     x_filtered[end_point] ** 2 + y_filtered[end_point] ** 2) + ' ' + str(test))
        # print('  ' + str(x_filtered[start_point_1]) + ' ' + str(y_filtered[start_point_1]) + ' ' + str(
        #     x_filtered[start_point] ** 2 + y_filtered[end_point] ** 2)+'\n')
        # end_point = end_point_1
        x_offset = rampp_offset(x_filtered[start_point:end_point])
        y_offset = rampp_offset(x_filtered[start_point:end_point])
        z_offset = rampp_offset(z_filtered[start_point:end_point])
        g_offset = rampp_offset(-gyro_filtered[start_point_1:end_point_1])
        gyro_z_offset = rampp_offset(gyro_z_world_filtered[start_point:end_point])
        # te = t[end_point_tmp]

        if t[hs_point] - t[to_point] < period:
            ax_x.plot(t[start_point:end_point], x_filtered[start_point:end_point], 'r', zorder=2)
            # ax_x.plot(t[hs_point:end_point_tmp], x_filtered[hs_point:end_point_tmp], 'b', zorder=3)
            ax_x.plot(t[end_point], x_filtered[end_point], 'o', zorder=4, markersize='5', alpha=0.5)
            ax_y.plot(t[start_point:end_point], y_filtered[start_point:end_point], 'r', zorder=2)
            # ax_y.plot(t[hs_point:end_point_tmp], y_filtered[hs_point:end_point_tmp], 'b', zorder=3)
            ax_y.plot(t[end_point], y_filtered[end_point], 'o', zorder=4, markersize='5', alpha=0.5)
            # ax_z.plot(t[start_point:end_point], z_filtered[start_point:end_point], 'r', zorder=2)
            # ax_z.plot(t[hs_point:end_point_tmp], z_filtered[hs_point:end_point_tmp], 'b', zorder=3)
            # ax_z.plot(t[end_point], z_filtered[end_point], 'o', zorder=4, markersize='5', alpha=0.5)

            s_vel_x = integrate.cumtrapz(x_filtered[start_point:end_point] - x_offset, dx=1 / fs)
            s_vel_y = integrate.cumtrapz(y_filtered[start_point:end_point] - y_offset, dx=1 / fs)
            s_vel_z = integrate.cumtrapz(z_filtered[start_point:end_point] - z_offset, dx=1 / fs)
            x_offset_v = rampp_offset_v(s_vel_x)
            y_offset_v = rampp_offset_v(s_vel_y)
            z_offset_v = rampp_offset_v(s_vel_z)
            s_len_x = integrate.cumtrapz(s_vel_x - x_offset_v, dx=1 / fs)
            s_len_y = integrate.cumtrapz(s_vel_y - y_offset_v, dx=1 / fs)
            s_len_z = integrate.trapz(s_vel_z - z_offset_v, dx=1 / fs)
            s_z = integrate.cumtrapz(s_vel_z - z_offset_v, dx=1 / fs)
            z_offset_s = rampp_offset_s(s_z)
            # theta = np.arctan((s_vel_y-y_offset_v)/(s_vel_x-x_offset_v))*180/np.pi
            # theta2 = euler_reset(theta, 120)
            theta_y = integrate.cumtrapz(-gyro_filtered[start_point_1:end_point_1] - g_offset, dx=1 / fs)
            ty_offset = rampp_offset_sig(theta_y)

            turning_ang = integrate.cumtrapz(gyro_z_world_filtered[start_point:end_point] - gyro_z_offset, dx=1/fs)
            # turning_ang = euler[start_point:end_point]
            # fig = plt.figure(5)
            # plt.plot(-gyro_filtered[start_point_1:end_point_1] - g_offset)
            # plt.plot(-gyro_filtered[start_point_1:end_point_1] + g_offset)
            # plt.plot(t[start_point_1:end_point_1-1], theta_y)
            # plt.plot(t[start_point_1:end_point_1-1], theta_y-ty_offset)
            # plt.plot(t[to_point:next_to], euler[to_point:next_to, 2])
            # plt.plot(t[start_point:end_point - 1], theta)
            # plt.grid()
            # plt.show()
            # plt.plot(z_filtered[start_point:end_point])
            # plt.plot(z_filtered[start_point:end_point]-z_offset)

            y_Ang[s_idx, 0] = np.max(theta_y-ty_offset)
            y_Ang[s_idx, 1] = np.min(theta_y-ty_offset)
            # y_Ang[s_idx, 2] = (theta_y-ty_offset)[idx_to_hs[i, 0]-start_point]
            y_Ang[s_idx, 3] = (theta_y-ty_offset)[idx_to_hs[i, 2]-start_point]
            stride[s_idx] = (s_len_x[-1] ** 2 + s_len_y[-1] ** 2) ** 0.5
            if i == 0:
                speed[s_idx] = stride[s_idx] / (t[end_point] - t[start_point])
            else:
                speed[s_idx] = stride[s_idx] / (t[idx_to_hs[i, 2]] - t[idx_to_hs[i-1, 2]])
            # clearance[s_idx] = np.max(s_z-z_offset_s + 0.05*(1-np.cos((theta_y-ty_offset)*np.pi/180))[:-1])
            stride_chara[s_idx, 0] = t[to_point]
            stride_chara[s_idx, 1] = abs(turning_ang[-1] - turning_ang[0]) < 20
            s_idx = s_idx + 1
        else:
            print('notice: abnormal gait period')

        if i < n_idx - 1 and t[idx_to_hs[i + 1, 2]] - t[idx_to_hs[i, 2]] < 1.5 * period:
            phase_time[p_idx, 0] = t[idx_to_hs[i+1, 2]] - t[idx_to_hs[i, 2]]  # hs->hs 这样实际上计算的是下一步的时间参数
            phase_time[p_idx, 1] = t[idx_to_hs[i+1, 0]] - t[idx_to_hs[i, 2]]  # stance
            phase_time[p_idx, 2] = t[idx_to_hs[i+1, 2]] - t[idx_to_hs[i+1, 0]]  # swing
            phase_time2[p_idx, 0] = t[idx_to_hs[i + 1, 1]] - t[idx_to_hs[i, 1]]  # hs->hs 这样实际上计算的是下一步的时间参数
            phase_time2[p_idx, 1] = t[idx_to_hs[i + 1, 0]] - t[idx_to_hs[i, 1]]  # stance
            phase_time2[p_idx, 2] = t[idx_to_hs[i + 1, 1]] - t[idx_to_hs[i + 1, 0]]  # swing
            p_idx = p_idx + 1
    plt.show()
    plt.close(fig)
    stride = stride[stride > 0]
    n_stride = stride.shape[0]
    stride_chara = stride_chara[0:n_stride, :]
    speed = speed[speed > 0]
    clearance = clearance.reshape(-1,)
    y_Ang = y_Ang.reshape((-1, 4))
    phase_time = phase_time[phase_time > 0].reshape((-1, 3))
    phase_time2 = phase_time2[phase_time2 > 0].reshape((-1, 3))
    cycle_time = phase_time[:, 0]
    spt_ratio = phase_time[:, 1] / phase_time[:, 0]
    print(spt_ratio)

    return stride, speed, cycle_time, spt_ratio, phase_time, clearance, idx_to_hs/freq(), \
        period, y_Ang, stride_chara, phase_time2


def si_param(param_l, param_r):
    return abs(param_l-param_r)*2/(param_l+param_r)


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
    for i in range(0, n_l-1):
        hs_l = to_hs_l[i, 2]
        to_l = to_hs_l[i+1, 0]
        bool_1 = to_hs_r[:, 0] > hs_l
        bool_2 = to_hs_r[:, 0] < to_l
        bool_to = bool_1 & bool_2
        to_ds_idx = np.where(bool_to)[0]
        to_ds = to_hs_r[to_ds_idx, 0]
        if to_ds_idx.size == 1 and to_ds-hs_l < 0.8*p_mean:
            ds_duration.append(to_ds-hs_l)
            if to_ds_idx != 0 and hs_l - to_hs_r[to_ds_idx-1, 2] < 0.8*p_mean:
                step_time.append(hs_l - to_hs_r[to_ds_idx-1, 2])  # l-r
                if lr_flag:
                    tmp = to_hs_r[to_ds_idx - 1, 2] - to_hs_r[to_ds_idx - 1, 0]
                    swing_time.append([hs_l-to_hs_l[i, 0], tmp[0]])
                    idx_s_l = np.where(stride_to_l == to_hs_l[i, 0])[0]
                    idx_s_r = np.where(stride_to_r == to_hs_r[to_ds_idx-1, 0])[0]
                    if idx_s_l.size == 1 and idx_s_r.size == 1:
                        stride_length.append([stride_l[idx_s_l][0], stride_r[idx_s_r][0]])
        bool_3 = to_hs_r[:, 2] > hs_l
        bool_4 = to_hs_r[:, 2] < to_l
        bool_hs = bool_3 & bool_4
        hs_ds_idx = np.where(bool_hs)[0]
        hs_ds = to_hs_r[hs_ds_idx, 2]
        if hs_ds_idx.size == 1 and to_l-hs_ds < 0.8 * p_mean:
            ds_duration.append(to_l-hs_ds)
            step_time.append(hs_ds - hs_l)  # r-l
            if not lr_flag and hs_ds_idx != n_r-1:
                tmp = to_hs_l[hs_ds_idx, 2]-to_hs_r[hs_ds_idx, 0]
                swing_time.append([hs_l-to_hs_l[i, 0], tmp[0]])
                idx_s_l = np.where(stride_to_l == to_hs_l[i, 0])[0]
                idx_s_r = np.where(stride_to_r == to_hs_r[hs_ds_idx, 0])[0]
                if idx_s_l.size == 1 and idx_s_r.size == 1:
                    stride_length.append([stride_l[idx_s_l], stride_r[idx_s_r]])

    ds_duration = np.array(ds_duration)
    step_time = np.array(step_time)

    # # print(np.array(swing_time))
    stride_length = np.array(stride_length)
    swing_time = np.array(swing_time)
    # print(stride_length)
    # print(si_param(stride_length[:, 0], stride_length[:, 1]))
    # print(si_param(stride_length[:, 0], stride_length[:, 1]).mean())
    # print(si_param(stride_l.mean(), stride_r.mean()))
    # print(swing_time)
    # print(si_param(swing_time[:, 0], swing_time[:, 1]))
    # print(si_param(swing_time[:, 0], swing_time[:, 1]).mean())
    return ds_duration, step_time, stride_length, swing_time


def main():
    # datasheet = "C:/Users/13372/Documents/MATLAB/Proj/gait/datasheet/步态样例/16/20230402-161422_IMU.csv"
    start = timeit.default_timer()
    path1 = r'C:\Users\13372\Documents\exp_data\gait\0928_1\daily'
    print(path1)
    print('now Tw Ta')
    l1 = '01'
    l2 = '08'
    l3 = '09'
    r1 = '03'
    r2 = '06'
    r3 = '07'
    file_l1 = glob.glob(path1 + '/DOT ' + l1 + '*.csv')
    file_l2 = glob.glob(path1 + '/DOT ' + l2 + '*.csv')
    file_l3 = glob.glob(path1 + '/DOT ' + l3 + '*.csv')
    file_r1 = glob.glob(path1 + '/DOT ' + r1 + '*.csv')
    file_r2 = glob.glob(path1 + '/DOT ' + r2 + '*.csv')
    file_r3 = glob.glob(path1 + '/DOT ' + r3 + '*.csv')

    file_l = file_l1 + file_l2 + file_l3
    file_r = file_r1 + file_r2 + file_r3
    file = file_l3 + file_r3
    to_hs = []
    period = []
    stride = []
    """stride_l, character_l, stride_r, character_r"""
    duration = []
    for i in range(0, len(file)):
        datasheet = file[i]
        skip_rows = 6
        # gait_data = read_data_xsens(datasheet, skip_rows)
        # body = np.mean(euler[500:1000], axis=0)
        # acc = acc_rot_euler_body_ref(gait_data.get('euler'), body, gait_data.get('acc'))
        gait_data = read_data_xsens(datasheet, skip_rows)
        euler = gait_data.get('euler')
        acc = acc_rot_euler(gait_data.get('euler'), gait_data.get('acc'))
        t = gait_data.get('t')
        gyro_xyz = gait_data.get('gyro')
        if i == 0 or i == 1 or i == 2:
            gyro_xyz[:, 1] = -gyro_xyz[:, 1]

        gait_result = gait_param(acc[:], gyro_xyz[:], t[:], euler)
        to_hs.append(gait_result[6])
        period.append(gait_result[7])
        print('stride length:')
        lens = gait_result[0]
        stride.append(lens)
        lens_str = np.array2string(lens, separator='\n')
        stride_chara = gait_result[9]
        # print(stride_chara)
        stride.append(stride_chara)
        print(lens_str[1:-1] + '\n')
        print('phase time:')
        lens = gait_result[4]
        duration.append(lens)
        print(lens)
        lens_str = np.array2string(lens[:, 0], separator='\n')
        print(lens_str[1:-1]+'\n')
        # lens = gait_result[-1]
        # print(lens)
        lens_str = np.array2string(lens[:, 1], separator='\n')
        print(lens_str[1:-1]+'\n')
        lens_str = np.array2string(lens[:, 2], separator='\n')
        print(lens_str[1:-1]+'\n')
        # print('clearance:')
        # lens = gait_result[5]
        # lens_str = np.array2string(lens, separator='\n')
        # print(lens_str[1:-1]+'\n')
        print('speed:')
        lens = gait_result[1]
        lens_str = np.array2string(lens, separator='\n')
        print(lens_str[1:-1] + '\n')
        print('-'*50)
        print('angle:')
        lens = gait_result[8][:, 0:2]
        lens_str = np.array2string(lens[:, 0], separator='\n')
        lens_str2 = np.array2string(lens[:, 1], separator='\n')
        print(lens_str[1:-1] + '\n' + '\n' + lens_str2[1:-1] + '\n')
        print('-' * 50)
    # print(to_hs)
    ds_duration, step_time, stride_length_paired, swing_time_paired = step_param(to_hs, period, stride)

    cadence = 60/step_time.mean()
    print('cadence: ', cadence)
    si_sw = si_param(swing_time_paired[:, 0], swing_time_paired[:, 1]).mean()
    si_sl = si_param(stride[0].mean(), stride[2].mean())
    print('si_sw: ', si_sw, ' si_sl: ', si_sl)
    cv_ds = cv_param(ds_duration)
    sw_all = [duration[0][:, 2], duration[1][:, 2]]
    cv_sw = cv_param_list(sw_all)
    stride_length_all = [stride[0], stride[2]]
    cv_sl = cv_param_list(stride_length_all)
    cd_all = [duration[0][:, 0], duration[1][:, 0]]
    cv_cd = cv_param_list(cd_all)

    print('cv_ds: ', cv_ds, ' cv_sw: ', cv_sw, ' cv_sl: ', cv_sl, ' cv_cd: ', cv_cd)
    end = timeit.default_timer()
    print('\n'+str(end - start))
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
