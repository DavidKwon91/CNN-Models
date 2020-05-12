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


#simple Network In Network with 3 mlpconv layers
#reference for the filters and structures
#https://github.com/BIGBALLON/cifar-10-cnn/blob/master/2_Network_in_Network/Network_in_Network_bn_keras.py
def simple_nin(input_shape, num_class):
    def conv(filters, kernel_size, prev_layers):
        x = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same')(prev_layers)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        return x
    
    input_layers = Input(shape = input_shape)
    
    #first mlpconv layers
    x = conv(filters = 192, kernel_size = 5, prev_layers = input_layers)
    x = conv(filters = 160, kernel_size = 1, prev_layers = x)
    x = conv(filters = 96, kernel_size = 1, prev_layers = x)
    x = MaxPooling2D(pool_size = 3, strides = 2, padding = 'same')(x)
    x = Dropout(0.5)(x)
    
    #second mlpconv layers
    x = conv(filters = 192, kernel_size = 5, prev_layers = x)
    x = conv(filters = 192, kernel_size = 1, prev_layers = x)
    x = conv(filters = 192, kernel_size = 1, prev_layers = x)
    x = MaxPooling2D(pool_size = 3, strides = 2, padding = 'same')(x)
    x = Dropout(0.5)(x)
    
    #third mlpconv layers
    x = conv(filters = 192, kernel_size = 3, prev_layers = x)
    x = conv(filters = 192, kernel_size = 1, prev_layers = x)
    x = conv(filters = num_class, kernel_size = 1, prev_layers = x)
    
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)
    
    model = Model(inputs = input_layers, outputs = x)
    
    return model
        
#example
simple_nin(input_shape = (32,32,3), num_class = 10).summary()


#You can customize the mlpconv layers with this class

#Network In Network
class NIN():
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
        
        self.mlpconv_layer = None
        
        self.input_layers = None
        
    def image_aug(self, X_train, **kwargs):
        datagen = ImageDataGenerator(**kwargs)
        datagen.fit(X_train)
        self.datagen = datagen
        
    def conv_def(self, filters, kernel_size, prev_layers, *, padding = 'same'):
        x = Conv2D(filters = filters, kernel_size = kernel_size, padding = padding,
                  kernel_initializer = self.ki,
                  kernel_regularizer = self.kr)(prev_layers)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        
        return x
    
    def mlpconv(self, filters_structure, first_kernel, *, prev_layers = None, first = False, out = False):
        
        if first:
            input_layers = Input(shape = self.input_shape)
            
            x = self.conv_def(filters = filters_structure[0], kernel_size = first_kernel, prev_layers = input_layers)
            x = self.conv_def(filters = filters_structure[1], kernel_size = 1, prev_layers = x)
            x = self.conv_def(filters = filters_structure[2], kernel_size = 1, prev_layers = x)
            
            self.input_layers = input_layers
        
        if first is False:
            if out:
                x = self.conv_def(filters = filters_structure[0], kernel_size = first_kernel, prev_layers = prev_layers)
                x = self.conv_def(filters = filters_structure[1], kernel_size = 1, prev_layers = x)
                x = self.conv_def(filters = self.num_class, kernel_size = 1, prev_layers = x)
                x = GlobalAveragePooling2D()(x)
                x = Activation('softmax')(x)
                
            else:
                x = self.conv_def(filters = filters_structure[0], kernel_size = first_kernel, prev_layers = prev_layers)
                x = self.conv_def(filters = filters_structure[1], kernel_size = 1, prev_layers = x)
                x = self.conv_def(filters = filters_structure[2], kernel_size = 1, prev_layers = x)
            
        if out is False:
            x = MaxPooling2D(pool_size = 3, strides = 2, padding = 'same')(x)
            x = Dropout(0.5)(x)
    
        self.mlpconv_layer = x
        
        return x
        
    def build_model(self, *, mlpconv = None):
        
        if self.input_layers is None:
            raise ValueError("Please initialize the first mlpconv")
            
        if mlpconv is None:
            model = Model(inputs = self.input_layers, outputs = self.mlpconv_layer)
        if mlpconv:
            model = Model(inputs = self.input_alyers, outputs = mlpconv)
        
        opt = self.opt
        
        if self.y_shape == 1:
            model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
            
        if self.y_shape > 1:
            model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
        
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

#customized version 1 (mini version)
nin = NIN(input_shape = (32,32,3), y_shape = 1, activation = 'relu', num_class = 10)
first_mlpconv = nin.mlpconv(filters_structure = [192, 160, 96], first_kernel = 5, first = True)
second_mlpconv = nin.mlpconv(filters_structure = [192,192,192], first_kernel = 5, prev_layers = first_mlpconv)
out_mlpconv = nin.mlpconv(filters_structure = [192, 192], first_kernel = 3, out = True, prev_layers = second_mlpconv)
nin.build_model()
nin.model.summary()

#customized version 2
nin_big = NIN(input_shape = (227,227,3), y_shape = 1000, activation = 'relu', num_class = 1000)
first_mlpconv = nin_big.mlpconv(filters_structure = [512, 256, 128], first_kernel = 11, first = True)
second_mlpconv = nin_big.mlpconv(filters_structure = [512, 512, 512], first_kernel = 7, prev_layers = first_mlpconv)
third_mlpconv = nin_big.mlpconv(filters_structure = [512,512,512], first_kernel = 7, prev_layers = second_mlpconv)
out_mlpconv = nin_big.mlpconv(filters_structure = [512, 512], first_kernel = 3, out = True, prev_layers = third_mlpconv)
nin_big.build_model()
nin_big.model.summary()


#fitting and predicting
#nin_big.fit(X_train = X_train, y_train = y_train, X_valid = X_valid, y_valid = y_valid)
#nin_big.evaluate(X_test = X_valid, y_test = y_valid)
#nin_big.predict(X_new = X_test)