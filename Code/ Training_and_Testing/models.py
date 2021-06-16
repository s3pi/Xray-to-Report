from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import sys
sys.path.insert(1, '../Networks')
from CRFEENET import make_model # from CRFENet import make_model (Depth separable convolution layers)

def custom_base_model_training(): #CRFFENet is the base model. This model puts Dense(15) on top of it.
    base_model = make_model((410, 410, 3)) 
    input_tensor = Input((410, 410, 3))
    output_tensor = base_model(input_tensor)
    op = GlobalAveragePooling2D()(output_tensor)
    op = Dense(15, activation = "sigmoid")(op)
    model = Model(input_tensor, op) #model is CRFENet+Dense(15) to classify NIH data
    # base_model.trainable = False
    # model.load_weights('Model_Weights_custom_2/model_89.h5')
    # base_model.save_weights('custom_512.h5')
    adam = Adam(lr = 1e-4)
    model.compile(loss = "binary_crossentropy", optimizer = adam)

    return model, base_model

