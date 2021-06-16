import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Activation, SeparableConv2D, MaxPooling2D, add
from tensorflow.keras.models import Model

#Depth separable convolution layers

def make_model(input_shape):
    inp_tensor = Input(input_shape)
    
    ###Input Block
    x = Conv2D(32, 3, padding = 'same')(inp_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(32, 3, strides = 2, use_bias = False, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, 3, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #### Block 1
    residual = Conv2D(128, 1, strides = 2, use_bias = False, padding = 'same')(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(2, strides = 2, padding = 'same')(x)

    x = add([x, residual])

    #### Block 2
    residual = Conv2D(256, 1, strides = 2, use_bias = False, padding = 'same')(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(2, strides = 2, padding = 'same')(x)

    x = add([x, residual])

    #### Block 3
    residual = Conv2D(256, 1, strides = 2, use_bias = False, padding = 'same')(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(2, strides = 2, padding = 'same')(x)

    x = add([x, residual])

    #### Block 4
    residual = Conv2D(512, 1, strides = 2, use_bias = False, padding = 'same')(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(2, strides = 2, padding = 'same')(x)

    x = add([x, residual])

    

    #### output block
    x = SeparableConv2D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    model = Model(inp_tensor, x)
    # model.summary()
    return model