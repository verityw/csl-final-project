import argparse
import csv
import os

TIME_LOOKAHEAD = 2.0
time_0 = 0

def avg(nums):
    return sum(nums)/len(nums)

def str_to_time(timestamp):
    time_split = timestamp.split("_")
    return (float(time_split[-1]) + 1000*float(time_split[-2]) + 60*1000*float(time_split[-3]) + 60*60*1000*float(time_split[-4]))/1000

def get_time_and_steer(data):
    data_new = []
    time_0 = str_to_time(data[0][0])

    for row in data:
        data_new.append([str_to_time(row[0]) - time_0, float(row[1])])
    return data_new

def get_smoothness(data):
    smoothness = []
    for index, point in enumerate(data):
        compare = 1
        smoothness.append([])
        while compare < len(data) - index and data[index + compare][0] - point[0] < TIME_LOOKAHEAD:
            smoothness[-1].append((abs(point[1] - data[index + compare][1])/(data[index + compare][0] - point[0])))
            compare += 1
    return smoothness

def get_metrics(data):
    maxs = []
    means = []
    for point in data:
        if len(point) > 0:
            maxs.append(max(point))
            means.append(avg(point))
    return maxs, means

def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Assess Smoothness')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='data')
    args = parser.parse_args()

    data_path = args.data_dir
    f = open(data_path)
    csv_f = csv.reader(f)
    data_strings = []
    for index, element in enumerate(csv_f):
        if index % 2 == 0:
            data_strings.append(element)
    data = get_time_and_steer(data_strings)
    smoothness = get_smoothness(data)
    maxs, means = get_metrics(smoothness)
    print("MAX OF TIME MAXS: ", max(maxs))
    print("MAX OF TIME MEANS: ", max(means))
    print("MEAN OF TIME MAXS: ", avg(maxs))
    print("MEAN OF TIME MEANS", avg(means))

if __name__ == '__main__':
    main()
