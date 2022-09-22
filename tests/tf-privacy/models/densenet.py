import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import backend as K

def densenet(input_shape, num_classes):
    base_model = DenseNet121(weights='imagenet', include_top=False)    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes)(x) 

    model = Model(inputs=base_model.input, outputs=predictions)   
   
    for layer in base_model.layers:
        layer.trainable = False

    return model
