import pandas as pd
import numpy as np
import os
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import codecs
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.utils import to_categorical

def read_csv(csv_name):
    with codecs.open(csv_name, "r", "Shift-JIS", "ignore") as file:
        df = pd.read_table(file, delimiter=",",header = 1)
    return df


def append_dataframe(df_list, csv_list, csv_dir):
    try:
        for num in range(len(csv_list)):
            df = read_csv(csv_dir + csv_list[num])
            df = df.head(1200)
            df = df.drop(range(0,60))
            df = df.iloc[:,[5,6,7,9,10,19,20,21,23,24,33,34,35,37,38,47,48,49,51,52,61,62,63,65,66]]
            df_list.append(df)
            #print(df_list[num])
    except:
        print('csv reading failed')
        pass 

def plot_histroy_loss(fit):
    axL.plot(fit.history['loss'], label = 'for training', color = 'dodgerblue')
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    
def plot_history_accuracy(fit):
    axR.plot(fit.history['accuracy'], label = 'for training', color = 'dodgerblue')
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    

if __name__ == '__main__':
    
    csv_dir = './lstm_imu_raw_data/'
    csv_list_asphalt = ['asphalt_normal_straight/アスファルト_普通_1.csv',
                        'asphalt_normal_straight/アスファルト_普通_2.csv',
                        'asphalt_normal_straight/アスファルト_普通_3.csv',
                        'asphalt_normal_straight/アスファルト_普通_4.csv',
                        'asphalt_normal_straight/アスファルト_普通_5.csv']
    
    csv_list_sand = ['sand_normal_straight/砂利_普通_1.csv', #x軸方向の線形加速度が不適当なデータ
                     'sand_normal_straight/砂利_普通_2.csv',
                     'sand_normal_straight/砂利_普通_3.csv',
                     'sand_normal_straight/砂利_普通_4.csv',
                     'sand_normal_straight/砂利_普通_5.csv',
                     'sand_normal_straight/砂利_普通_6.csv']
    
    #list is road_walkspeed_rout
    df_list_asphalt = []
    df_list_sand = []
    append_dataframe(df_list_asphalt, csv_list_asphalt, csv_dir)
    append_dataframe(df_list_sand, csv_list_sand, csv_dir)
    
    for num in range(len(csv_list_asphalt)):
        df_list_asphalt[num]['index'] = 0
    for num in range(len(csv_list_sand)):
        df_list_sand[num]['index'] = 1
    
    LABELNUM = 2
    
    
    """データの前処理"""
    df_train =  pd.concat([df_list_asphalt[0],
                           df_list_asphalt[1],
                           df_list_asphalt[2],
                           df_list_asphalt[3],
                           df_list_sand[1],
                           df_list_sand[2],
                           df_list_sand[3],
                           df_list_sand[4]])
    
    df_test = pd.concat([df_list_asphalt[4], df_list_sand[5]])
    
    x_train = df_train.drop('index', axis = 1)
    y_train = df_train['index']
    x_test = df_test.drop('index', axis = 1)
    y_test = df_test['index']
    
    y_test_label = y_test
        
    x_test = np.array(x_test)
    zeros = np.zeros((x_train.shape[0] - x_test.shape[0], x_train.shape[1]))
    x_test = np.append(x_test, zeros, axis = 0)
    x_test = pd.DataFrame(x_test)
    x_train = np.reshape(x_train.values, [1,x_train.shape[0], x_train.shape[1]])
    y_train = np.reshape(np.array(to_categorical(y_train)), [1, y_train.shape[0], LABELNUM])
    x_test = np.reshape(x_test.values, [1,x_test.shape[0], x_test.shape[1]])
    y_test = np.reshape(np.array(to_categorical(y_test)), [1, y_test.shape[0], LABELNUM])
       
    
    """モデル定義"""
    BATCHSIZE = 8
    EPOCHS = 300
    UNITNUM = 20
    optimizer = RMSprop()
    model = Sequential()
    model.add(LSTM(UNITNUM, input_shape = (x_train.shape[1], x_train.shape[2]), return_sequences = True))
    model.add(Dropout(0.2))
    model.add(Dense(LABELNUM))
    model.add(Activation('softmax'))
    model.compile(loss = 'mean_squared_error', optimizer = optimizer, metrics = ['accuracy'])
    model.summary()
    
    history = model.fit(x_train, y_train, batch_size = BATCHSIZE, epochs = EPOCHS)
    modelName = model.__class__.__name__
    
    y_predicted = model.predict(x_test)
    y_predicted = np.reshape(y_predicted, (x_train.shape[1], LABELNUM))
    y_predicted = pd.DataFrame(y_predicted)
    y_predicted = y_predicted.idxmax(axis = 1)
    y_predicted.to_csv('predicted_%s_LSTM.csv' % modelName)
    
    trueLabel = y_test_label.values
    predictedLabel = y_predicted.values
    predictedLabel = predictedLabel[: y_test.shape[1]]
    
    cm = confusion_matrix(trueLabel, predictedLabel)
    sns.heatmap(cm, cbar = True, cmap = 'Greens')
    plt.xlabel('predictedLabel')
    plt.ylabel('trueLabel')
    plt.savefig('CM_%s_LSTM.png'%modelName, format = 'png', dpi =300)
    
    fig, (axL, axR) = plt.subplots(ncols = 2, figsize = (10,5))
    plt.subplots_adjust(wspace = 0.5)
    plot_histroy_loss(history)
    plot_history_accuracy(history)
    fig.savefig('loss_and_accuracy_LSTM.png',format = 'png', dpi = 300)

