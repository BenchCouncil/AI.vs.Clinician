import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf

import gc
from time import time
import os
import math
import pickle
import numpy as np
import pandas as pd
from pad_sequences import PadSequences
from attention_function import attention_3d_block as Attention
from keras import backend as K
# from keras.models import Model, Input, load_model #model_from_json
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Masking, Flatten, Embedding, Dense, LSTM, TimeDistributed
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_curve
from keras import regularizers
from keras import optimizers
# from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization,Bidirectional

import sys




# FILE = "CHARTEVENTS_reduced_24_hour_blocks_plus_admissions_plus_patients_plus_scripts_plus_icds_plus_notes.csv"

######################################
## MAIN ###
######################################

def get_synth_sequence(n_timesteps=14):
    """

    Returns a single synthetic data sequence of dim (bs,ts,feats)

    Args:
    ----
      n_timesteps: int, number of timesteps to build model for

    Returns:
    -------
      X: npa, numpy array of features of shape (1,n_timesteps,2)
      y: npa, numpy array of labels of shape (1,n_timesteps,1)

    """

    X = np.array([[np.random.rand() for _ in range(n_timesteps)], [np.random.rand() for _ in range(n_timesteps)]])
    X = X.reshape(1, n_timesteps, 2)
    y = np.array([0 if x.sum() < 0.5 else 1 for x in X[0]])
    y = y.reshape(1, n_timesteps, 1)
    return X, y


def wbc_crit(x):
    if (x > 12 or x < 4) and x != 0:
        return 1
    else:
        return 0


def temp_crit(x):
    if (x > 100.4 or x < 96.8) and x != 0:
        return 1
    else:
        return 0


def build_model(no_feature_cols=None, time_steps=7, output_summary=False):
       print("time_steps:{0}|no_feature_cols:{1}".format(time_steps, no_feature_cols))
       input_layer = Input(shape=(time_steps, no_feature_cols))
       x = Attention(input_layer, time_steps)
       x = Masking(mask_value=-4.0)(x)
       x = LSTM(256, return_sequences=True)(x)
       x = Bidirectional(LSTM(256, return_sequences=True))(x)

       x = Dropout(dropout)(x)
       x = BatchNormalization()(x)
       x = LSTM(128, return_sequences=True)(x)
       x = Bidirectional(LSTM(128, return_sequences=True))(x)

       x = Dropout(dropout)(x)
       x = BatchNormalization()(x)
       x = LSTM(64, return_sequences=False)(x)
       x = Bidirectional(LSTM(64, return_sequences=False))(x)

       x = Dropout(dropout)(x)
       x = BatchNormalization()(x)
       preds = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(x)
       model = Model(inputs=input_layer, outputs=preds)

       RMS = optimizers.RMSprop(lr=lr_size, rho=0.9, epsilon=1e-08)
       model.compile(optimizer=RMS, loss='binary_crossentropy', metrics=['acc'])

       if output_summary:
           model.summary()
       return model

def train(model_name="kaji_mach_0", predict=True, return_model=True, time_steps=336, epochs=10, data_path='',
          interval='',kind = '',miss_rate = 0):
    df_tr = pd.read_csv(data_path + 'lstm_' + interval + f'h_train_model_input_data_by_{kind}_del_missdata_{miss_rate}.csv')
    df_tr2 = pd.read_csv(data_path + 'lstm_' + interval + f'h_mimiciv_train_model_input_data_by_{kind}.csv')
    # if interval == '0':
    #     df_tr2 = pd.concat([df_tr2,df_tr2,df_tr2])
    df_tr = pd.concat([df_tr,df_tr2])
    train_x, train_y = get_label(df_tr, time_steps)

    tr_permutation = np.random.permutation(train_x.shape[0])
    train_x = train_x[tr_permutation]
    train_y = train_y[tr_permutation]

    df_va = pd.read_csv(data_path + 'lstm_' + interval + f'h_val_model_input_data_by_{kind}_del_missdata_{miss_rate}.csv')
    df_val2 = pd.read_csv(data_path + 'lstm_' + interval + f'h_mimiciv_val_model_input_data_by_{kind}.csv')
    # if interval == '0':
    #     df_val2 = pd.concat([df_val2,df_val2,df_val2])
    df_va = pd.concat([df_va,df_val2])
    val_x, val_y = get_label(df_va, time_steps)
    va_permutation = np.random.permutation(val_x.shape[0])
    val_x = val_x[va_permutation]
    val_y = val_y[va_permutation]

    df_test = pd.read_csv(data_path + 'lstm_' + interval + f'h_test_model_input_data_by_{kind}_del_missdata_{miss_rate}.csv')
    df_test2 = pd.read_csv(data_path + 'lstm_' + interval + f'h_mimiciv_test_model_input_data_by_{kind}.csv')
    # if interval == '0':
    #     df_test2 = pd.concat([df_test2,df_test2,df_test2])
    df_test = pd.concat([df_test, df_test2])
    test_x, test_y = get_label(df_test, time_steps)

    no_feature_cols = train_x.shape[2]

    # build model
    model = build_model(no_feature_cols=no_feature_cols, output_summary=True,
                        time_steps=time_steps)

    # init callbacks
    tb_callback = TensorBoard(log_dir='./logs/{0}_{1}.log'.format(model_name, time),
                              histogram_freq=0,
                              write_grads=False,
                              write_images=True,
                              write_graph=True)

    # early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=3)

    # Make checkpoint dir and init checkpointer
    checkpoint_dir = "./saved_models/{0}".format(model_name)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpointer = ModelCheckpoint(
        filepath=checkpoint_dir + "/model.{epoch:02d}-{val_loss:.2f}.hdf5",
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)


    # fit
    model.fit(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        epochs=epochs,
        # callbacks=[tb_callback], #, checkpointer],
        # callbacks=[tb_callback,  checkpointer],
        callbacks=[tb_callback, early_stop],
        validation_data=(val_x, val_y),
        shuffle=True)

    model.save('./saved_models/{0}.h5'.format(model_name))

    history = model.history.history
    train_loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_plot_{interval}.png')

    if predict:
        print('================TEST===============')
        predict_test_y = model.predict(test_x)
        get_score(test_y, predict_test_y)

        print('===============SAMPLE===============')
        df_sample = pd.read_csv(data_path + 'lstm_' + interval + f'h_mimiciv_sample_model_input_data_by_{kind}.csv')
        sample_test_x, sample_test_y = get_label(df_sample, time_steps)

        predict_sample_test_y = model.predict(sample_test_x)
        get_score(sample_test_y, predict_sample_test_y)

    # if return_model:
    #     return model, test_x, test_y

def get_label(df_sample, time_steps=336):
    df_label = df_sample['ill_label']
    df_label_array = df_label.values
    df_label_array = df_label_array.reshape(int(df_label_array.shape[0] / time_steps), time_steps)
    df_label_array = df_label_array[:, 0]
    array_label = []
    for i in range(len(df_label_array)):
        array_label.append(int(df_label_array[i]))
    if 'ill_label' in list(df_sample.columns):
        del df_sample['ill_label']

    sample_array = df_sample.values
    sample_array = sample_array.reshape(int(sample_array.shape[0] / time_steps), time_steps, sample_array.shape[1])

    sample_test_x = sample_array
    sample_test_y = np.array(array_label)
    return sample_test_x, sample_test_y

def get_score(test_y, predict_test_y):
    print('Confusion Matrix')
    print(confusion_matrix(test_y, np.around(predict_test_y)))
    print('Accuracy')
    print(accuracy_score(test_y, np.around(predict_test_y)))
    print('ROC AUC SCORE')
    print(roc_auc_score(test_y, predict_test_y))
    print('CLASSIFICATION REPORT')
    print(classification_report(test_y, np.around(predict_test_y)))

    count_1 = sum(x == 1 and y == 1 for x, y in zip(np.around(predict_test_y), test_y))
    count_2 = sum(x == 0 and y == 0 for x, y in zip(np.around(predict_test_y), test_y))
    count_3 = sum(x == 1 and y == 0 for x, y in zip(np.around(predict_test_y), test_y))
    count_4 = sum(x == 0 and y == 1 for x, y in zip(np.around(predict_test_y), test_y))
    print(f'real label 1,model predict 1  number {count_1}')
    print(f'real label 0,model predict 0 number {count_2}')
    print(f'real label 1,model predict 0 number {count_4}')
    print(f'real label 0,model predict 1 number {count_3}')
    print(f'sensitivity {count_1 / np.sum(test_y == 1)}')
    print(f'specificity {count_2 / np.sum(test_y == 0)}')

    fpr, tpr, thresholds = roc_curve(test_y, predict_test_y)
    return fpr, tpr, thresholds


def get_0h_3h_score():
    df_3000_sample = pd.read_csv('mimic_database/mapped_elements_3000/mimic_3000_sample.csv',
                                 usecols=['HADM_ID', 'START_ENDTIME', 'ILL_TIME'])
    df_predict_0 = pd.read_csv('mimic_database/mapped_elements_3000/mimic_3000_sample_0h_predict_data.csv')
    df_predict_3 = pd.read_csv('mimic_database/mapped_elements_3000/mimic_3000_sample_3h_predict_data.csv')

    df_3000_sample = pd.merge(df_3000_sample, df_predict_0, on=['HADM_ID', 'START_ENDTIME'], how='inner')
    df_3000_sample = pd.merge(df_3000_sample, df_predict_3, on=['HADM_ID', 'START_ENDTIME'], how='inner')

    real_label_list = []
    predict_list = []
    for index, row in df_3000_sample.iterrows():
        predict_0h = float(row['0h_PREDICT'])
        predict_3h = float(row['3h_PREDICT'])
        ill_time = row['ILL_TIME']
        if str(ill_time) == 'nan':
            real_label = 0
        else:
            real_label = 1
        # 首先判断0h是否发病
        if predict_0h <= 0.5:
            if predict_3h <= 0.5:
                pre = min(predict_0h, predict_0h)
            else:
                pre = predict_3h
        else:
            pre = predict_0h
        predict_list.append(pre)
        real_label_list.append(real_label)

    predict_test_y = np.array(predict_list)
    test_y = np.array(real_label_list)
    print('============= SAMPLE ===============')
    get_score(test_y, predict_test_y)

def sample_predict(data_path,interval):
    model_name = f'/home/mimic-lstm/model_3h/saved_models_3h_75/lstm_1layer_model_3h_epochs2_diff_meanvalue_del_missdata_0.4.h5'
    df_sample = pd.read_csv(data_path + 'lstm_3h_mimiciv_sample_model_input_data_by_diff_meanvalue.csv')
    sample_test_x, sample_test_y = get_label(df_sample, time_steps=336)

    model = load_model(model_name)
    predict_sample_test_y = model.predict(sample_test_x)
    get_score(sample_test_y, predict_sample_test_y)


batch_size = 32
lr_size = 0.0001
dropout = 0.4

if __name__ == "__main__":
    # arguments = sys.argv
    # miss_rate = float(arguments[1])
    miss_rate = 0.4

    kind = 'diff_meanvalue'
    time_steps = 336
    epochs = 2
    interval = '3'
    lstm_lary = 1

    ROOT = f'/home/ddcui/hai-med-database/mimic-lstm/data/'
    data_path = ROOT+f'{kind}_del_missdata_{miss_rate}/'
    model_name = f'lstm_{lstm_lary}layer_model_{interval}h_epochs{epochs}_{kind}_del_missdata_{miss_rate}'
    print('Training')
    K.clear_session()
    train(model_name=model_name, epochs=epochs, predict=True, return_model=True,
         time_steps=time_steps, data_path=data_path, interval=interval,kind=kind,miss_rate = miss_rate)
    print('Predict')
    sample_predict(data_path,interval)
