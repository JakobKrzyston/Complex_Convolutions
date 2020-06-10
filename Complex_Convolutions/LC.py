def LC(x):
    """
    Author: Jakob Krzyston

	Purpose: Define the linear combination enabling complex convolutions
    """
    import keras.backend as K
    y = K.constant([0, 1, 0, -1, 0, 1],shape=[2,3])
    return K.dot(x,K.transpose(y))