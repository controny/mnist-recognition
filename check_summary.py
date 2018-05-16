# coding=utf-8
import matplotlib.pyplot as plt
import json
import getopt
import sys
import os
from train import log_dir


def load_summary(summary_path):
    with open(summary_path, 'r') as f:
        summary = json.load(f)['summary']

    return summary


def main():
    model_name = 'model'
    # 设置命令行参数更改model_name
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['model_name='])
    except getopt.GetoptError as err:
        print('ERROR: %s!' % err)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '--model_name':
            model_name = arg

    summary_path = os.path.join(log_dir, model_name+'-summary.json')
    summary = load_summary(summary_path)
    x = [int(list(e.keys())[0]) for e in summary]
    y = [list(e.values())[0] for e in summary]
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()
