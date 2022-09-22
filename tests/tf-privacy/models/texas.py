from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input
from tensorflow.keras import regularizers

def texas(num_classes=100):
    model = Sequential()
    model.add(Input(shape=(6169,)))        
    model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes,
                kernel_regularizer=regularizers.L1L2(l1=1e-2, l2=1e-2),
                bias_regularizer=regularizers.L2(1e-2),
                activity_regularizer=regularizers.L2(1e-2)
            ))

    return model


def shadow(num_classes=100):
    # model = Sequential()
    # model.add(Input(shape=(6169,))) 
    # model.add(Dense(4096, activation='tanh'))
    # model.add(Dropout(0.5))   
    # model.add(Dense(2048, activation='tanh'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1024, activation='tanh'))
    # model.add(Dropout(0.5))
    # model.add(Dense(512, activation='tanh'))
    # model.add(Dropout(0.5))
    # model.add(Dense(256, activation='tanh'))
    # model.add(Dropout(0.5))
    # model.add(Dense(128, activation='tanh'))
    # model.add(Dense(num_classes))

    model = Sequential()
    model.add(Input(shape=(6169,)))    
    model.add(Dense(1024, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(num_classes))

    return model

