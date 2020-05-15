##import packages
import keras
from keras import optimizers
from keras import initializers
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input, concatenate, Dropout, BatchNormalization, add, Layer
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

#simple Lenet
def simple_lenet(input_shape, num_class):
    input_layers = Input(shape = input_shape)
    
    x = Conv2D(filters = 6, kernel_size = 5, padding = 'valid', activation = 'tanh')(input_layers)
    x = AveragePooling2D(pool_size = 2, strides = 2)(x)
    x = Conv2D(filters = 16, kernel_size = 5, padding = 'valid', activation = 'tanh')(x)
    x = AveragePooling2D(pool_size = 2, strides = 2)(x)
    x = Flatten()(x)
    x = Dense(120, activation = 'tanh')(x)
    x = Dense(84, activation = 'tanh')(x)
    x = Dense(num_class, activation = 'tanh')(x)
    
    model = Model(inputs = input_layers, outputs = x)
    
    return model
    
simple_lenet(input_shape = (32,32,3), num_class = 10).summary()


#You can customize of the lenet model

#Lenet
class lenet():
    def __init__(self, input_shape, y_shape, activation, num_class, *,
                batch_size = 32, padding = 'valid', kernel_initializer = None, kernel_regularizer = None,
                epochs = 10, shuffle = False, opt = None):
        
        self.input_shape = input_shape
        self.y_shape = y_shape
        self.activation = activation
        self.num_class = num_class
        self.batch_size = batch_size
        self.padding = padding
        
        if opt is None:
            self.opt = optimizers.Adam(lr = 1e-3)
        else:
            self.opt = opt
        
        if kernel_initializer is None:
            self.ki = 'glorot_normal'
        else:
            self.ki = kernel_initializer

        self.kr = kernel_regularizer
        self.epochs = epochs
        self.shuffle = shuffle
        
        self.model = None
        self.datagen = None
        
        
    def image_aug(self,X_train, **kwargs):
        datagen = ImageDataGenerator(**kwargs)
        datagen.fit(X_train)
        self.datagen = datagen
        
    @staticmethod
    def _maxpool(*, x = None, pool_size = 2, strides = 2):
        #pool size is 2 by default as used in the paper
        if x is None:
            return MaxPooling2D(pool_size = pool_size, strides = strides)
        else:
            return MaxPooling2D(pool_size = pool_size, strides = strides)(x)
    
    @staticmethod
    def _avgpool(*, x = None, pool_size = 2, strides = 2):
        #pool size is 2 by default as used in the paper
        if x is None:
            return AveragePooling2D(pool_size = pool_size, strides = strides)
        else:
            return AveragePooling2D(pool_size = pool_size, strides = strides)(x)
                
    
    def _conv_def(self, filters, *,kernel_size = 5, strides = 1, input_shape = None, bn=False):
        #kernel size is 5 by default as used in the paper
        if input_shape is None:
            return Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, 
                          padding = self.padding, kernel_initializer = self.ki, kernel_regularizer = self.kr)
        
        else:
            return Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, 
                          padding = self.padding, kernel_initializer = self.ki, kernel_regularizer = self.kr, input_shape = input_shape)
       
    def _structure_layers(self, structure):
        
        layers = [self._conv_def(filters = structure[0], input_shape = self.input_shape)]
        
        for i in structure[1:]:
            if i == 'M':
                layers += [self._maxpool()]
            elif i == 'A':
                layers += [self._avgpool()]
            else:
                layers += [self._conv_def(filters = i),
                          Activation(self.activation)]
                
        return layers
        
        
    
    def build_model(self, *, structure = (6, 'A', 16, 'A')):
        
        """
        ------------------------------------------------------------------------------
        #Arguments
        
            structure (list or tuple) : list or tuple for the structure of lenet.
                int : filters of each convolution layers
                'A' : Average pooling 2D
                'M' : Max Pooling 2D
                
        #Return
        
            model (Model) : Keras model instance, compiled
        ------------------------------------------------------------------------------
        
        """
        
        model = Sequential(self._structure_layers(structure))
        
        model.add(Flatten())
        model.add(Dense(120, activation = self.activation, kernel_initializer = self.ki, kernel_regularizer = self.kr))
        model.add(Dense(84, activation = self.activation, kernel_initializer = self.ki, kernel_regularizer = self.kr))
        model.add(Dense(10, activation = 'softmax', kernel_initializer = self.ki, kernel_regularizer = self.kr))
        
        opt = self.opt
        
        if self.y_shape == 1:
            model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
            
        else:
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
    


    
#original Lenet model example

ln = lenet(input_shape = (32,32,3), y_shape = 1, activation = 'tanh', num_class = 10)
ln.build_model()
ln.model.summary()

#fitting
#ln.fit(X_train = X_train, y_train = y_train, X_valid = X_valid, y_valid = y_valid)

#evaluating
#ln.evalute(X_test = X_test, y_test = y_test)

#predict
#ln.predict(X_new = X_test)

#customized Lenet model example

ln_custom = lenet(input_shape = (32,32,3), y_shape = 1, activation = 'relu', num_class = 10)
ln_custom.build_model(structure = (6, 'M', 16, 'M'))
ln_custom.model.summary()


weight_decay = 1e-5
ln_custom2 = lenet(input_shape = (32,32,3), y_shape = 1, activation = 'relu', num_class = 10,
                  kernel_initializer = initializers.he_normal(), kernel_regularizer = regularizers.l2(weight_decay),
                  opt = optimizers.SGD(lr = 1e-3, momentum = 0.9, nesterov = True))

ln_custom2.build_model(structure = (6,'M',16,32,'M'))
ln_custom2.model.summary()
