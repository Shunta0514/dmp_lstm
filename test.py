import glob
import re
import os
from operator import itemgetter

file_train = []
file_test = []
file = glob.glob('lstm_imu_raw_data/[a][s][p][h]*/*.csv')
for i in range(len(file)):
    if i % 5 == 0:
        file_test.append(file[i])
    else:
        file_train.append(file[i])
        
print(file_train)
print(file_test)