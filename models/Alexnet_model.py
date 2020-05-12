##import packages
import keras
from keras import optimizers
from keras import initializers
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input, concatenate, Dropout, BatchNormalization, add, Layer
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.datasets import cifar10
import time


#Alexnet - including original and mini version of Alexnet
class alexnet():
    def __init__(self, input_shape, y_shape, activation, num_class, *,
                lr = 1e-3, kernel_initializer = None, kernel_regularizer = None,
                epochs = 10, batch_size = 32, shuffle = False, opt = None):
        
        self.input_shape = input_shape
        self.y_shape = y_shape
        self.activation = activation
        self.lr = lr
       
        self.num_class = num_class
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if opt is None:
            self.opt = optimizers.Adam(lr = 1e-3)
        else:
            self.opt = opt
        
        if kernel_initializer is None:
            self.ki = 'glorot_normal'
        else: 
            self.ki = kernel_initializer
            
        self.kr = kernel_regularizer
        self.model = None
        self.datagen = None
        
        self.input_layers = None
        
        
    @staticmethod
    def maxpool(*, prev_layers = None, pool_size = 2, strides = 2, padding = 'valid'):
        #pool size is 2 by default as used in the paper
        if prev_layers is None:
            return MaxPooling2D(pool_size = pool_size, strides = strides)
        else:
            return MaxPooling2D(pool_size = pool_size, strides = strides)(prev_layers)
        
    def conv_def(self, filters, *, prev_layers = None, kernel_size = 3, strides = 1, padding = 'same', input_shape = None, bn = False):
        
        if input_shape is not None:
            input_layers = Input(shape = self.input_shape)
            x = Conv2D(filters = filters, kernel_size = kernel_size, padding = padding, strides = strides,
                       kernel_initializer = self.ki, kernel_regularizer = self.kr)(input_layers)
            
            self.input_layers = input_layers
            
        else:
            x = Conv2D(filters = filters, kernel_size = kernel_size, padding = padding, strides = strides,
                       kernel_initializer = self.ki, kernel_regularizer = self.kr)(prev_layers)
            
            
        if bn:
            x = BatchNormalization()(x)
            
        return Activation(self.activation)(x)
    
    
    def image_aug(self, X_train, **kwargs):
        datagen = ImageDataGenerator(**kwargs)
        datagen.fit(X_train)
        self.datagen = datagen
        
    #"Response-normalization layers follow the first and second convolutional layers."
    
    #mini - for mini version of Alexnet
    def build_model(self, *, mini = False):
        
        if mini:
            x = self.conv_def(filters = 64, strides = 2, input_shape = self.input_shape, bn = True)
            x = self.maxpool(prev_layers = x)
            x = self.conv_def(filters = 192, prev_layers = x, bn = True)
            x = self.maxpool(prev_layers = x)
            x = self.conv_def(filters = 256, prev_layers = x)
            x = self.conv_def(filters = 256, prev_layers = x)
            x = self.maxpool(prev_layers = x)
            
            x = Flatten()(x)
            x = Dense(512, activation = self.activation)(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation = self.activation)(x)
            x = Dropout(0.5)(x)
            x = Dense(self.num_class, activation = 'softmax')(x)
            
        #original alexnet
        else:
            x = self.conv_def(filters = 96, kernel_size = 11, strides = 4, padding = 'valid', input_shape = self.input_shape, bn = True)
            x = self.maxpool(prev_layers = x) #padding = valid
            x = self.conv_def(filters = 256, kernel_size = 5, prev_layers = x, bn = True)
            x = self.maxpool(prev_layers = x)
            x = self.conv_def(filters = 384, prev_layers = x)
            x = self.conv_def(filters = 384, prev_layers = x)
            x = self.conv_def(filters = 256, prev_layers = x)
            x = self.maxpool(prev_layers = x)
            
            x = Flatten()(x)
            x = Dense(4096, activation = self.activation)(x)
            x = Dropout(0.5)(x)
            x = Dense(4096, activation = self.activation)(x)
            x = Dropout(0.5)(x)
            x = Dense(self.num_class, activation = 'softmax')(x)
        
        model = Model(inputs = self.input_layers, outputs = x)
        
        opt = self.opt
        
        if self.y_shape == 1:
            loss = 'sparse_categorical_crossentropy'
        else:
            loss = 'categorical_crossentropy'
            
        model.compile(loss = loss, optimizer=  opt, metrics = ['accuracy'])
        
        self.model = model
        
        return model
    
    def fit(self, X_train, y_train, *, X_valid = None, y_valid = None):
        if X_valid is not None:
            history = self.model.fit(X_train, y_train, epochs = self.epochs, validation_data = (X_valid, y_valid), 
                                     batch_size = self.batch_size, shuffle = self.shuffle)
        else:
            history = self.model.fit(X_train, y_train, epochs = self.epochs,
                                     batch_size = self.batch_size, shuffle = self.shuffle)
            
        return history
    
    def fit_generator(self, X_train, y_train, *, X_valid = None, y_valid = None):
        if X_valid is not None and self.datagen is not None:
            history = self.model.fit_generator(self.datagen.flow(X_train, y_train, batch_size = self.batch_size),
                                               epochs = self.epochs,
                                               steps_per_epoch = len(X_train)/self.batch_size,
                                               validation_data = (X_valid, y_valid))
        else:
            history = self.model.fit_generator(self.datagen.flow(X_train, y_train, batch_size = self.batch_size),
                                              epochs = self.epochs)
            
        return history
    
    
    def evaluate(self, X_test, y_test):
        score = self.model.evaluate(X_test, y_test)
        return score
    
    def predict(self, X_new):
        pred = self.model.predict(X_new)
        return pred
    
    
    
    
#mini Alexnet - can be improved by modifying this model
alx = alexnet(input_shape = (32,32,3), y_shape = 10, activation = 'relu', num_class = 10)
alx.build_model(mini = True)
alx.model.summary()

#original Alexnet
alx_org = alexnet(input_shape = (227, 227, 3), y_shape = 1, activation = 'relu', num_class = 10)
alx_org.build_model()
alx_org.model.summary()

#fitting and predicting
#alx.fit(X_train = X_train, y_train = y_train, X_valid = X_valid, y_valid = y_valid)
#alx.evaluate(X_test = X_valid, y_test = y_valid)
#alx.predict(X_new = X_test)