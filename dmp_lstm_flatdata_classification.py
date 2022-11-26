import pandas as pd
import numpy as np
import os
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import codecs
from sklearn.metrics import confusion_matrix
import glob

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
    sampling_frequency = 20
    LABELNUM = 5
    #get csv infomation
    csv_information_file = 'IMUdata_information.csv'
    info_df = pd.read_csv(csv_information_file)
    alive_dfs = info_df[info_df['is_damaged'] == False]
    flat_dfs = alive_dfs[alive_dfs['is_slope'] == False]
    #データが不十分なのでカット
    flat_dfs = flat_dfs[flat_dfs['Label_num'] != 1]
    flat_dfs = flat_dfs[flat_dfs['Label_num'] != 5]
    """データの前処理"""
    df_list_train = []
    df_list_test  = []
    for num in range(len(flat_dfs)):
        individual_info = flat_dfs.iloc[num]
        
        Location = individual_info['file_location']
        Useful = individual_info['useful[s]']
        Cut = individual_info['cut_initial[s]']
        Label = individual_info['Label_num']
        if Label == 6:
            Label = 1
        Count = individual_info['data_counts']
                
        csv_df = read_csv(Location)
        csv_df = csv_df.head(Useful*sampling_frequency)
        csv_df = csv_df.drop(range(0, Cut * sampling_frequency))
        csv_df = csv_df.iloc[:,[5,6,7,9,10,19,20,21,23,24,33,34,35,37,38,47,48,49,51,52,61,62,63,65,66]]
        csv_df['index'] = Label
        if(Count % 5 == 0):
            df_list_test.append(csv_df)
        else:
            df_list_train.append(csv_df)
    
    df_train =  pd.concat(df_list_train)
    
    df_test = pd.concat(df_list_test)
    
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
    BATCHSIZE = 16
    EPOCHS = 300
    UNITNUM = 10
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

