import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Rescaling, Dense, SpatialDropout2D, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input

tf.random.set_seed(40)

def create_model(img_height, img_width, img_depth=3, rate_dropout=0.2):
    input_layer = Input([img_height, img_width, img_depth])
    x = Rescaling(1./255.)(input_layer)
    
    x = Conv2D(16, (1, 1), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    
    x = Conv2D(32, (1, 1), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(64, (1, 1), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(64, (1, 1), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(128, (1, 1), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(256, (1, 1), activation="relu")(x)
    x = SpatialDropout2D(rate_dropout)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=input_layer, outputs=x)

    return model