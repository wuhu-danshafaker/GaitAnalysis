#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import FSRAnalysis as fa


def freq():
    return 60


def read_data_oiseau(datasheet):
    """读取监测系统导出的文件"""
    gait = pd.read_csv(datasheet, skiprows=1)
    t = gait[['Frame']].to_numpy()[:].reshape(-1)
    t = t / freq()
    fsr_l = gait[['Average Pressure', 'Average Pressure.1', 'Average Pressure.2', 'Average Pressure.3', 'Average Pressure.4', 'Average Pressure.5', 'Average Pressure.6', 'Average Pressure.7']].to_numpy()
    fsr_r = gait[
        ['Average Pressure.8', 'Average Pressure.9',
         'Average Pressure.10', 'Average Pressure.11', 'Average Pressure.12', 'Average Pressure.13',
         'Average Pressure.14', 'Average Pressure.15']].to_numpy()
    return fsr_l, fsr_r


def get_step(fsr: np.ndarray, zeros=30):
    # fsr[:, 3] = 0
    total_force = fsr.sum(axis=1)  # - fsr[:, 3]
    zero_indices = np.where(total_force == 0)[0]
    zero_intervals = np.split(zero_indices, np.where(np.diff(zero_indices) >= 10)[0] + 1)  # 噪点的预防
    zero_intervals = [interval for interval in zero_intervals if len(interval) >= zeros]

    step = np.zeros((len(zero_intervals) - 1, 2)).astype(int)
    for i in range(0, step.shape[0]):
        step[i, 0] = zero_intervals[i][-1]
        step[i, 1] = zero_intervals[i + 1][0]
        if step[i, 1] - step[i, 0] > 60:  # 实时修改？
            step[i, 0] = 0
            step[i, 1] = 0
    return step


def main(csvData):
    fsrs = read_data_oiseau(csvData)
    res = {}
    lor = ['left', 'right']
    for i, fsr in enumerate(fsrs):
        steps = get_step(fsr, 15)
        step_para = fa.get_step_para(fsr, steps)
        max_pressure_mean = np.apply_along_axis(fa.get_filtered_mean, axis=0, arr=step_para.get('max_pressure'), method='No')
        duration_mean = np.apply_along_axis(fa.get_filtered_mean, axis=0, arr=step_para.get('duration'), method='No')/freq()
        step_mean = np.mean(step_para.get('step_duration'))/freq()
        tmp = {'mean_pressure': max_pressure_mean, 'mean_duration': duration_mean, 'mean_step': step_mean}
        # tmp = max_pressure_mean.tolist() + duration_mean.tolist()
        # tmp.append(step_mean)
        # fa.printArrResult('Xsensor', np.array(tmp))
        res[lor[i]] = tmp
    return res


if __name__ == '__main__':
    with open('fsr_result.log', 'w') as f:
        f.write('---\n')
    # path = r'C:\Users\13372\Documents\课程\监测系统\实验数据\罗柏清\1107\*'
    main(r'C:\Users\13372\Documents\课程\监测系统\实验数据\10MWT\Xsensors\luo2.csv')