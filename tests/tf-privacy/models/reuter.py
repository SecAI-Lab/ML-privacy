from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.regularizers import l1

def reuter(num_classes=46):
    model = Sequential()
    model.add(Dense(512, input_shape=(10000,)))
    model.add(Activation('relu'))        
    model.add(Dense(num_classes, kernel_regularizer=l1(0.1)) )  
    return model
    