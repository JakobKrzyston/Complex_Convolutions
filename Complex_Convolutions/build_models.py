def build(in_shp, classes, dr):
    """
    # Author
    Jakob Krzyston
    
    # Purpose
    Build the mdoels to be used in the experiment
    
    # Inputs
    in_shp  - (array) Input shape
    classes - (str) list of modulation classes
    dr      - (double) Dropout rate
    
    # Outputs
    Models      - Tuple containing the architectures
    model_names - List of model names
    """
    # Import Packages
    import os
    import keras.backend as K
    os.environ["KERAS_BACKEND"] = "tensorflow"
    from keras.utils import np_utils
    import keras.models as models
    from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,Lambda,Permute
    from keras.layers.convolutional import Convolution2D, ZeroPadding2D
    from keras.optimizers import adam
    from collections import namedtuple


    # CNN2
    cnn2 = models.Sequential()
    cnn2.add(Reshape([1]+in_shp, input_shape=in_shp))
    cnn2.add(ZeroPadding2D((0, 2),data_format='channels_first'))
    cnn2.add(Convolution2D(256, (1, 3), padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform', data_format='channels_first'))#ch from 3->4
    cnn2.add(Dropout(dr))
    cnn2.add(Convolution2D(80, (2, 1), padding='valid', activation="relu", name="conv2", kernel_initializer='glorot_uniform', data_format='channels_first'))
    cnn2.add(Dropout(dr))
    cnn2.add(Flatten())
    cnn2.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
    cnn2.add(Dropout(dr))
    cnn2.add(Dense(len(classes), kernel_initializer='he_normal', name="dense2"))
    cnn2.add(Activation('softmax'))
    cnn2.add(Reshape([len(classes)]))
    cnn2.compile(loss='categorical_crossentropy', optimizer='adam')


    # CNN2-260
    cnn2_260 = models.Sequential()
    cnn2_260.add(Reshape([1]+in_shp, input_shape=in_shp))
    cnn2_260.add(ZeroPadding2D((0, 2),data_format='channels_first'))
    cnn2_260.add(Convolution2D(256, (1, 3), padding='valid', activation="relu", kernel_initializer='glorot_uniform', data_format='channels_first'))#ch from 3->4
    cnn2_260.add(Dropout(dr))
    cnn2_260.add(Convolution2D(80, (2, 1), padding='valid', activation="relu", kernel_initializer='glorot_uniform', data_format='channels_first'))
    cnn2_260.add(Dropout(dr))
    cnn2_260.add(Flatten())
    cnn2_260.add(Dense(260, activation='relu', kernel_initializer='he_normal'))
    cnn2_260.add(Dropout(dr))
    cnn2_260.add(Dense(len(classes), kernel_initializer='he_normal'))
    cnn2_260.add(Activation('softmax'))
    cnn2_260.add(Reshape([len(classes)]))
    cnn2_260.compile(loss='categorical_crossentropy', optimizer='adam')


    # Complex
    
    # Define the linear combination
    def LC(x):
        import keras.backend as K
        y = K.constant([0, 1, 0, -1, 0, 1],shape=[2,3])
        return K.dot(x,K.transpose(y))

    complex_CNN = models.Sequential()
    complex_CNN.add(Reshape([1]+in_shp, input_shape=in_shp))
    complex_CNN.add(ZeroPadding2D((1, 2),data_format='channels_first'))
    complex_CNN.add(Convolution2D(256, (2, 3), padding='valid', activation='linear', name="conv1", kernel_initializer='glorot_uniform', data_format='channels_first'))#ch from 3->4
    complex_CNN.add(Permute((1,3,2)))
    complex_CNN.add(Lambda(LC))
    complex_CNN.add(Permute((1,3,2)))
    complex_CNN.add(Activation('relu'))
    complex_CNN.add(Dropout(dr))
    complex_CNN.add(Convolution2D(80, (2, 3), padding='valid', activation="relu", name="conv2", kernel_initializer='glorot_uniform', data_format='channels_first'))
    complex_CNN.add(Dropout(dr))
    complex_CNN.add(Flatten())
    complex_CNN.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
    complex_CNN.add(Dropout(dr))
    complex_CNN.add(Dense( len(classes), kernel_initializer='he_normal', name="dense2" ))
    complex_CNN.add(Activation('softmax'))
    complex_CNN.add(Reshape([len(classes)]))
    complex_CNN.compile(loss='categorical_crossentropy', optimizer='adam')
    
    
    # Create a named tuple to store the architectures in
    Models = namedtuple('Models','CNN2 CNN2_260 Complex')
    model_names = ['CNN2', 'CNN2_260', 'Complex']
    
    return Models(cnn2, cnn2_260, complex_CNN), model_names