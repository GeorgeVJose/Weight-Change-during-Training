import numpy as np
import matplotlib.pyplot as plt
import math

path_dir = "./summary_hpc/"

def read_file(filename):
    return np.load(filename)

def plot_single(data, title=None, xtitle=None, ytitle=None):
    x_axis = [i[0] for i in data]
    y_axis = [i[1] for i in data]
    figure = plt.plot(x_axis, y_axis)
    figure.title(title)
    figure.xlabel(xtitle)
    figure.ylabel(ytitle)
    figure.show()

def plot_multiple(datas, titles=None, xtitles=None, ytitles=None):
    num_plots = len(datas)
    if num_plots%2==0:
        plt_size = (num_plots/2, num_plots/2)
    else:
        plt_size = (num_plots/2, num_plots/2 + 1)

    figure, plots = plt.subplots(plt_size)

    for i, data in enumerate(datas):
        x_axis = [x[0] for x in data]
        y_axis = [x[1] for x in data]




# output_mean = np.load(path_dir+"output_mean.npy")
#
# # X axis
# x = [i[0] for i in output_mean]
#
# # Y axis
# y = [i[1] for i in output_mean]
# y_new = []
#
# y_prev = y[0]
# for item in y:
#     item = np.absolute(item-y_prev)
#     y_new.append(item)
#     y_prev = item
#
# plt.plot(x, y_new[:-1])
# plt.title("Output Mean")
# plt.show()
# # print(output_mean)
