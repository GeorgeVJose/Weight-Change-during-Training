import numpy as np
import pandas as pd
import os

def read_file(file_name):
    x = np.load(file_name)
    x_return = [i[0] for i in x]
    return x_return

dir_list = os.listdir('./')
files_to_read = []
for name in dir_list:
    if name[-3:]=='npy':
        files_to_read.append(name)

data = {}
label_names = []
for name in files_to_read:
    label_names.append(name[:-4])

for name in files_to_read:
    file_data = list(read_file(name))
    data[name[:-4]] = file_data

df = pd.DataFrame(data, columns=label_names)
print(df.head())

df.to_csv("weight_mean.csv", index=True)
df.to_excel("weight_mean.xlsx", sheet_name="Mean_Data")
