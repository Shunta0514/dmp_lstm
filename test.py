import glob
import re
import os
import csv
from operator import itemgetter
import pandas as pd

csvFileName = 'IMUlocation.csv'
file = glob.glob('lstm_imu_raw_data/*/*/*.csv')
newlist = []
newfile = []
for num in range(len(file)):
    newlist.append('.\\'+file[num])
    newfile.append(newlist)
    newlist = []
with open(csvFileName, 'w',newline="") as wf:
    fwriter = csv.writer(wf, lineterminator='\n')
    for num in range(len(newfile)):
        fwriter.writerow(newfile[num])
    
df = pd.read_csv('IMUdata_information.csv')
alive_df = df[df['is_damaged'] == False]
print(alive_df)
dfsub = alive_df.iloc[3]
print(dfsub)
print(dfsub['useful[s]'])