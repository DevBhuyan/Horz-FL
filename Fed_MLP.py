from MLP_helpers import horz_split, trainModel, fedMLP, acc
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd
import numpy as np

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
        
    fed = fedMLP(models)
    fed.compile(optimizer=Adam(learning_rate = 0.001), loss='binary_crossentropy' if num_classes==1 else 'categorical_crossentropy', metrics=['accuracy'])
    
    # Aggregate data from all models
    df = pd.concat(df_list)
    x_fed = df.iloc[:, :-1]
    y_fed = df.iloc[:, -1]
    
    if num_classes > 1:
        y_fed = one_hot_encode(y_fed, num_classes)
    
    num_rows = x_fed.shape[0]
    
    xfedval = x_fed.iloc[int(num_rows*0.8):]
    try:
        yfedval = y_fed.iloc[int(num_rows*0.8):]
    except:
        yfedval = y_fed[int(num_rows*0.8):, :]
        
    fed.fit(x_fed, y_fed, callbacks=callbacks, steps_per_epoch=64, epochs=20, validation_data=(xfedval, yfedval))
        
    accuracies = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        accuracies.append(acc(fed, x, y))
    accuracy = sum(accuracies)/len(accuracies)
    
    del(models)
    del(fed)
    del(df)
    del(accuracies)
    
    return accuracy