from MLP_helpers import horz_split, trainModel, fedMLP, acc
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd
import numpy as np
import tensorflow as tf
import gc
gc.enable()

tf.random.set_seed(42)

def one_hot_encode(values, num_classes):
    values = values.values.astype(int)
    one_hot = np.zeros((len(values), num_classes))
    one_hot[np.arange(len(values)), values] = 1
    return one_hot

def Fed_MLP(df_list):
    
    x_train, y_train, x_test, y_test = horz_split(df_list)
        
    models = []
    for x, y in zip(x_train, y_train):
        models.append(trainModel(x, y))
        
    callbacks = [ReduceLROnPlateau(monitor = "val_accuracy",
                                 factor = 0.5,
                                 patience = 3,
                                 verbose = 1,
                                 mode = 'max'),
                 EarlyStopping(monitor='val_accuracy',
                               patience=5,           
                               verbose=1,
                               restore_best_weights=True,
                               mode = 'max')]
      
    try:
        num_classes = y_train[0].shape[1]
    except:
        num_classes = 1
        
    fed1 = fedMLP(models)
    fed1.compile(optimizer=Adam(learning_rate = 0.001), loss='binary_crossentropy' if num_classes==1 else 'categorical_crossentropy', metrics=['accuracy'])
    
    models2 = []
    for x, y in zip(x_train, y_train):
        models2.append(trainModel(x, y, fed1))
        
    fed2 = fedMLP(models2)
    fed2.compile(optimizer=Adam(learning_rate = 0.001), loss='binary_crossentropy' if num_classes==1 else 'categorical_crossentropy', metrics=['accuracy'])
    
    # Aggregate data from all models
    # df = pd.concat(df_list)
    # x_fed = df.iloc[:, :-1]
    # y_fed = df.iloc[:, -1]
    
    # if num_classes > 1:
    #     y_fed = one_hot_encode(y_fed, num_classes)
    
    # num_rows = x_fed.shape[0]
    
    # xfedval = x_fed.iloc[int(num_rows*0.8):]
    # try:
    #     yfedval = y_fed.iloc[int(num_rows*0.8):]
    # except:
    #     yfedval = y_fed[int(num_rows*0.8):, :]
        
    # fed.fit(x_fed, y_fed, callbacks=callbacks, steps_per_epoch=64, epochs=20, validation_data=(xfedval, yfedval))
        
    fed_acc = []
    fed_p = []
    fed_r = []
    fed_f = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        fed_acc.append(acc(fed2, x, y)[0])
        fed_p.append(acc(fed2, x, y)[1])
        fed_r.append(acc(fed2, x, y)[2])
        fed_f.append(acc(fed2, x, y)[3])
    accu = sum(fed_acc)/len(fed_acc)
    prec = sum(fed_p)/len(fed_p)
    rec = sum(fed_r)/len(fed_r)
    f1 = sum(fed_f)/len(fed_f)
    
    # del(models)
    # del(fed)
    # del(df)
    # del(accuracies)
    
    return accu, prec, rec, f1