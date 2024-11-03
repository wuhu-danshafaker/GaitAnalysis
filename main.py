import argparse
import ImuAnalysis

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser example')
    parser.add_argument('--todo', type=str, default='gait')
    parser.add_argument('--csvPath', type=str)

    args = parser.parse_args()

    csvPath = args.csvPath if args.csvPath else r'C:\Users\13372\Documents\课程\csvData\黎\OMC_1'

    if args.todo == 'gait':
        ImuAnalysis.imu_analysis(csvPath)
