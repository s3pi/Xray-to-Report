import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, SeparableConv2D, add, Conv3D, Conv2D, MaxPool3D, MaxPooling2D, GlobalAveragePooling3D, GlobalAveragePooling2D, Dropout, Dense, Lambda, TimeDistributed, LSTM, Bidirectional, GlobalAveragePooling1D, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, Activation

def normalisation_layer(x):
    return(tf.nn.l2_normalize(x, 1, 1e-10))

def batchnorm_relu(input_tensor):
    x = BatchNormalization()(input_tensor)
    x = Activation('relu')(x)

    return x

def RedCRFENet(input_shape, drop_probability = 0.0, bottleneck_layer_size=512, weight_decay=1e-5):
    inp_tensor = Input(input_shape)
    
    ###Input Block
    x = Conv2D(32, 3, padding = 'same',kernel_regularizer=l2(weight_decay))(inp_tensor)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(32, 3, strides = 2, padding = 'same',kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, 3, padding = 'same',kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #### Block 1
    residual = Conv2D(128, 1, strides = 1, use_bias = False, padding = 'same',kernel_regularizer=l2(weight_decay))(x)
    # residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, 3, padding='same',kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(128, 3, padding='same',kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = MaxPooling2D(2, strides = 2, padding = 'same')(x)

    x = add([x, residual])

    #### Block 2
    residual = Conv2D(256, 1, strides = 2, use_bias = False, padding = 'same',kernel_regularizer=l2(weight_decay))(x)
    # residual = BatchNormalization()(residual)

    x = SeparableConv2D(256, 3, padding='same',kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(256, 3, padding='same',kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(2, strides = 2, padding = 'same')(x)

    x = add([x, residual])

    #### Block 3
    residual = Conv2D(256, 1, strides = 2, use_bias = False, padding = 'same',kernel_regularizer=l2(weight_decay))(x)
    # residual = BatchNormalization()(residual)

    x = SeparableConv2D(256, 3, padding='same',kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(256, 3, padding='same',kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(2, strides = 2, padding = 'same')(x)

    x = add([x, residual])

    #### Block 4
    residual = Conv2D(512, 1, strides = 2, use_bias = False, padding = 'same',kernel_regularizer=l2(weight_decay))(x)
    # residual = BatchNormalization()(residual)

    x = SeparableConv2D(512, 3, padding='same',kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(512, 3, padding='same',kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(2, strides = 2, padding = 'same')(x)

    x = add([x, residual])


    #### output block
    x = SeparableConv2D(512, 3, padding='same',kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(512, 3, padding='same',kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    base_model = Model(inp_tensor, x)
    base_model.load_weights('base_model205.h5') #classifier_model_weights as mentioned in the ADNet_readme.txt

    x = GlobalAveragePooling2D()(x)
    x = Dense(bottleneck_layer_size)(x)
    # print(x.shape)
    x = Lambda(normalisation_layer)(x)
    # print(x.shape)
    # exit()
    final_model = Model(inp_tensor, x)

    # model.summary()
    return final_model
