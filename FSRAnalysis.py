#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import glob
import XsensorAnalysis as x


def freq():
    return 100


def read_data_oiseau(datasheet):
    """读取监测系统导出的文件"""
    gait = pd.read_csv(datasheet)
    t = gait[['timeCounter']].to_numpy()[:].reshape(-1)
    t = t / freq()
    fsr = gait[['FSR0', 'FSR1', 'FSR2', 'FSR3', 'FSR4', 'FSR5', 'FSR6', 'FSR7']].to_numpy()
    adc = gait[['ADC0', 'ADC1', 'ADC2', 'ADC3', 'ADC4', 'ADC5', 'ADC6', 'ADC7', 'ADC8',
                'ADC9', 'ADC10', 'ADC11', 'ADC12']].to_numpy()
    ntc = gait[['NTC0', 'NTC1', 'NTC2', 'NTC3']]
    gait_data = {'t': t, 'fsr': fsr, 'adc': adc, 'ntc': ntc}
    return gait_data


def median_filter(mat: np.ndarray):
    return scipy.signal.medfilt(mat, (3, 1))


def get_step(fsr: np.ndarray, zeros=30):
    total_force = fsr.sum(axis=1)
    zero_indices = np.where(total_force == 0)[0]
    zero_intervals = np.split(zero_indices, np.where(np.diff(zero_indices) != 1)[0] + 1)  # 噪点的预防
    zero_intervals = [interval for interval in zero_intervals if len(interval) >= zeros]

    step = np.zeros((len(zero_intervals) - 1, 2)).astype(int)
    for i in range(0, step.shape[0]):
        step[i, 0] = zero_intervals[i][-1]
        step[i, 1] = zero_intervals[i + 1][0]
        if step[i, 1] - step[i, 0] > 150:  # 实时修改？
            step[i, 0] = 0
            step[i, 1] = 0
    return step


def get_step_para(fsr: np.ndarray, steps: np.ndarray):
    steps = steps[~np.any(steps == 0, axis=1)]
    step_count = steps.shape[0]
    max_pressure = np.zeros((step_count, 8))
    duration = np.zeros((step_count, 8))
    for i in range(0, step_count):
        if np.any(steps[i, :] == 0):
            continue

        start_point = steps[i, 0]
        end_point = steps[i, 1]
        max_pressure[i] = np.max(fsr[start_point:end_point, :], axis=0)
        duration[i] = np.count_nonzero(fsr[start_point:end_point, :], axis=0)

    return {'max_pressure': max_pressure, 'duration': duration, 'step_duration': steps[:, 1] - steps[:, 0]}


def get_filtered_mean(arr, method='IQR'):
    filtered_data = np.array([])
    if not arr.shape:
        pass
    if method == 'IQR':
        Q1 = np.percentile(arr, 25)
        Q3 = np.percentile(arr, 75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        filtered_data = arr[(arr >= lower) & (arr <= upper)]
        # filtered_data = arr
    if method == 'STD':
        mean = np.mean(arr)
        std_dev = np.std(arr)
        threshold = 3 * std_dev
        filtered_data = arr[abs(arr - mean) < threshold]
    if method == 'No':
        filtered_data = arr
    if method == 'NoM':
        pass
    return np.mean(filtered_data)


def get_paras_from_dir(searchDir):
    file_l = glob.glob(searchDir + 'left' + '*.csv')
    file_r = glob.glob(searchDir + 'right' + '*.csv')
    time_l = file_l[0].split('\\')[-1].split('_left_')[-1].split('_192')[0]
    time_r = file_r[0].split('\\')[-1].split('_right_')[-1].split('_192')[0]
    name_l = file_l[0].split('\\')[-1].split('_')[0]
    name_r = file_r[0].split('\\')[-1].split('_')[0]
    if time_l != time_r or name_l != name_r:
        print("not matched!")
        return
    files = file_l + file_r
    lor = ['left', 'right']
    dic = {}
    for i in range(len(files)):
        file = files[i]
        data = read_data_oiseau(file)
        fsr_data = median_filter(data.get('fsr'))
        steps = get_step(fsr_data)
        paras = get_step_para(fsr_data, steps)
        max_pressure_mean = np.apply_along_axis(get_filtered_mean, axis=0, arr=paras.get('max_pressure'), method='No') / (
                    np.pi * 0.0045 * 0.0045) / 1000
        duration_mean = np.apply_along_axis(get_filtered_mean, axis=0, arr=paras.get('duration'), method='No') / freq()
        step_mean = get_filtered_mean(paras.get('step_duration'), method='No') / freq()
        tmp = {'mean_pressure': max_pressure_mean, 'mean_duration': duration_mean, 'mean_step': step_mean}
        dic[lor[i]] = tmp

        # plt.figure()
        # for i in range(8):
        #     plt.plot(fsr_data[:, i])
        # plt.show()

    return dic


def printArrResult(name, arr=np.array([])):
    with open('fsr_result.log', 'a') as fr:
        if arr.size != 0:
            print('Para:', name, file=fr)
            if type(arr) == np.ndarray:
                # arr_str = np.array2string(arr, separator=' ')
                # print(arr_str, file=f)
                for row in arr:
                    print(row, file=fr, end=',')
                print('', file=fr)
            else:
                print(arr, file=fr)
        else:
            print(name, file=fr)


def arr2str(arr):
    return np.array2string(arr, separator=',').strip('[').strip(']')


def get_df(gait_para):
    df = pd.DataFrame({'mean_pressure_l': [arr2str(gait_para.get('left').get('mean_pressure'))],
                       ' mean_duration_l': [arr2str(gait_para.get('left').get('mean_duration'))],
                       'mean_step_l': [arr2str(gait_para.get('left').get('mean_step'))],
                       'mean_pressure_r': [arr2str(gait_para.get('right').get('mean_pressure'))],
                       'mean_duration_r': [arr2str(gait_para.get('right').get('mean_duration'))],
                       'mean_step_r': [arr2str(gait_para.get('right').get('mean_step'))]
                       })
    return df


def main(path_list):
    df = pd.DataFrame(columns=
                      ['mean_pressure_l', ' mean_duration_l', 'mean_step_l', 'mean_pressure_r', 'mean_duration_r',
                       'mean_step_r'])

    step_datas = []
    for searchDir in path_list:
        searchDir = searchDir + '\\*'
        step_para = get_paras_from_dir(searchDir)
        step_datas.append(step_para)
        df = pd.concat([df, get_df(step_para)], ignore_index=True)

    tmp_pl = []
    tmp_dl = []
    tmp_sl = []
    tmp_pr = []
    tmp_dr = []
    tmp_sr = []
    data_mean = []
    for step_data in step_datas:
        tmp_pl.append(step_data.get('left').get('mean_pressure'))
        tmp_dl.append(step_data.get('left').get('mean_duration'))
        tmp_sl.append(step_data.get('left').get('mean_step'))
        tmp_pr.append(step_data.get('right').get('mean_pressure'))
        tmp_dr.append(step_data.get('right').get('mean_duration'))
        tmp_sr.append(step_data.get('right').get('mean_step'))
    for single_data in [tmp_pl, tmp_dl, tmp_sl, tmp_pr, tmp_dr, tmp_sr]:
        single_data = np.array(single_data)
        if not single_data.shape:
            print(1)
        data_mean.append(np.apply_along_axis(np.mean, axis=0, arr=single_data))

    mean_row = [','.join(str(data) for data in data_arr.tolist()) if data_arr.shape
                else str(data_arr)
                for data_arr in data_mean]
    empty_row = pd.DataFrame([mean_row], columns=df.columns)
    df = pd.concat([df, empty_row], ignore_index=True)

    x_res = x.main(r'C:\Users\13372\Documents\课程\监测系统\实验数据\10MWT\Xsensors\宋若存_new.csv')
    df = pd.concat([df, get_df(x_res)], ignore_index=True)
    df.to_csv('fsr_result.csv', index=False)


if __name__ == '__main__':
    with open('fsr_result.log', 'w') as f:
        f.write('---\n')
    path = r'C:\Users\13372\Documents\课程\监测系统\实验数据\宋若存\*\*'
    paths = glob.glob(path)
    main(paths)
