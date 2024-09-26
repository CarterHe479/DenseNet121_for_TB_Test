from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, MaxPool2D, AvgPool2D, concatenate
from tensorflow.keras.models import Model

def bn_rl_conv(x, filters, kernel=1, strides=1):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel, strides=strides, padding='same')(x)
    return x

def dense_block(x, repetition, filters=32):
    for _ in range(repetition):
        y = bn_rl_conv(x, 4 * filters)
        y = bn_rl_conv(y, filters, 3)
        x = concatenate([y, x])
    return x

def transition_layer(x):
    x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
    x = AvgPool2D(2, strides=2, padding='same')(x)
    return x

def densenet(input_shape, n_classes, filters=32):
    input_layer = Input(input_shape)
    x = Conv2D(64, 7, strides=2, padding='same')(input_layer)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    
    for repetition in [6, 12, 24, 16]:
        d = dense_block(x, repetition)
        x = transition_layer(d)
    
    x = GlobalAveragePooling2D()(d)
    output = Dense(n_classes, activation='sigmoid')(x)  # For binary classification
    
    model = Model(input_layer, output)
    return model
