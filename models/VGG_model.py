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



class vgg():
    def __init__(self, input_shape, y_shape, activation, num_class, *,
                lr = 1e-3, kernel_initializer = None, kernel_regularizer = None,
                 padding = 'same', epochs = 10, batch_size = 32, shuffle = False, opt = None,
                output_dense = 4096):
        
        self.input_shape = input_shape
        self.y_shape = y_shape
        self.activation = activation
        self.lr = lr
        self.padding = padding
       
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
            
        self.input_layers = None
        self.kr = kernel_regularizer
        self.model = None
        self.datagen = None
        
        self.vgg_version = set(['11', '13', '16_conv1', '16', '19'])
        self.output_dense = output_dense
        
    def image_aug(self,X_train, **kwargs):
        datagen = ImageDataGenerator(**kwargs)
        datagen.fit(X_train)
        self.datagen = datagen
        
    def structure_layers(self, structure):
        
        layers = [Conv2D(filters = 64, kernel_size = 3, padding = self.padding, input_shape = self.input_shape,
                        kernel_initializer = self.ki, kernel_regularizer = self.kr),
                 BatchNormalization(),
                 Activation('relu')]
        
        for i in structure:
            if i == "M":
                layers += [MaxPooling2D(pool_size = 2, strides = 2)]
                
            else:
                layers += [Conv2D(filters = i, kernel_size = 3, padding = self.padding,
                                 kernel_initializer = self.ki, kernel_regularizer = self.kr),
                          BatchNormalization(),
                          Activation('relu')]
                
        return layers
        
    def build_model(self, vgg_version = '19'):
        if vgg_version not in self.vgg_version:
            raise ValueError(vgg_version + ', this vgg version is not available')

        if vgg_version == '11':
            structure = ['M',128,'M',256,256,'M',512,512,'M',512,512,'M']

            model = Sequential(self.structure_layers(structure))

        if vgg_version == '13':
            structure = [64, 'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M']

            model = Sequential(self.structure_layers(structure))

        if vgg_version == '16':
            structure = [64, 'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']

            model = Sequential(self.structure_layers(structure))

        if vgg_version == '19':
            structure = [64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']

            model = Sequential(self.structure_layers(structure))


        model.add(Flatten())
        model.add(Dense(self.output_dense, kernel_initializer = self.ki, kernel_regularizer = self.kr))
        model.add(BatchNormalization())
        model.add(Activation(self.activation))
        model.add(Dropout(0.5))

        model.add(Dense(self.output_dense, kernel_initializer = self.ki, kernel_regularizer = self.kr))
        model.add(BatchNormalization())
        model.add(Activation(self.activation))
        model.add(Dropout(0.5))

        model.add(Dense(self.num_class))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

        opt = self.opt

        if self.y_shape == 1:
            model.compile(loss = "sparse_categorical_crossentropy", optimizer = opt, metrics = ['accuracy'])
        if self.y_shape > 1:
            model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ['accuracy'])

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
            
        

#vgg11
vgg_model = vgg(input_shape = (32,32,3), y_shape = 1, activation = 'relu', num_class = 10)
vgg_model.build_model(vgg_version = '11')


#vgg19
vgg_model = vgg(input_shape = (32,32,3), y_shape = 1, activation = 'relu', num_class = 10)
vgg_model.build_model(vgg_version = '19')


#fitting and predicting

#vgg_model.fit(X_train = X_train, y_train = y_train, X_valid = X_valid, y_valid = y_valid)
#vgg_model.evaluate(X_test = X_valid, y_test = y_valid)
#vgg_model.predict(X_new = X_test)