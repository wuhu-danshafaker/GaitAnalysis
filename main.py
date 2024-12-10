import argparse
import ImuAnalysis

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser example')
    parser.add_argument('--todo', type=str, default='gait')
    parser.add_argument('--csvPath', type=str)

    args = parser.parse_args()

    csvPath = args.csvPath if args.csvPath else r'C:\Users\13372\Documents\课程\csvData\1106'

    if args.todo == 'gait':
        ImuAnalysis.imu_analysis(r'C:\Users\13372\Documents\课程\监测系统\健康医学院数据\csv\1130\孙佳伟\OMC_3')
