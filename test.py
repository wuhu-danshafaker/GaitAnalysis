import numpy as np
import pandas as pd


def read_data_oiseau(datasheet):
    data = pd.read_csv(datasheet, skiprows=0)
    data = data.values

    # for row in data:
    #     row[0:8] = row[0:8]/np.sum(row[0:8])
    #     row[8:16] = row[8:16]/row[16]
    #     row[17:25] = row[17:25]/np.sum(row[17:25])
    #     row[25:33] = row[25:33]/row[33]

    even_rows = data[::2]
    # even_rows[:, 0] = even_rows[:, 0]*3.55/4.6
    # even_rows[:, 7] = even_rows[:, 7]*3.75/4.6
    # even_rows[:, 17] = even_rows[:, 17]*3.55/4.6
    # even_rows[:, 24] = even_rows[:, 24]*3.75/4.6
    odd_rows = data[1::2]

    res = np.empty((even_rows.shape[0], even_rows.shape[1]*2))
    res[:, 0::2] = even_rows
    res[:, 1::2] = odd_rows

    header = 'mean_pressure_l.1,mean_pressure_l.2,mean_pressure_l.3,mean_pressure_l.4,mean_pressure_l.5,mean_pressure_l.6,mean_pressure_l.7,mean_pressure_l.8, mean_duration_l.1, mean_duration_l.2, mean_duration_l.3, mean_duration_l.4, mean_duration_l.5, mean_duration_l.6, mean_duration_l.7, mean_duration_l.8,mean_step_l,mean_pressure_r.1,mean_pressure_r.2,mean_pressure_r.3,mean_pressure_r.4,mean_pressure_r.5,mean_pressure_r.6,mean_pressure_r.7,mean_pressure_r.8,mean_duration_r.1,mean_duration_r.2,mean_duration_r.3,mean_duration_r.4,mean_duration_r.5,mean_duration_r.6,mean_duration_r.7,mean_duration_r.8,mean_step_r'

    header_list = header.split(',')

    new_header = []
    for h in header_list:
        new_header.append(h +' OM')
        new_header.append(h + ' X')
    header = ','.join(new_header)
    np.savetxt(r'C:\Users\13372\Documents\MATLAB\Proj\校正fsr\res.csv', res, delimiter=',', fmt='%f', header=header)

    np.savetxt('xsensor.csv', odd_rows, delimiter=',', fmt='%f')
    np.savetxt('oiseauMonitor.csv', even_rows, delimiter=',', fmt='%f')
    return


read_data_oiseau('test.csv')
