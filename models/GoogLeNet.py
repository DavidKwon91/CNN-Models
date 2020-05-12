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



#This GoogLeNet has 2 versions, which is original(Inception) and mini version

#the architecture of mini version - https://arxiv.org/pdf/1611.03530.pdf
class googlenet():
    def __init__(self, input_shape, y_shape, activation, num_class, *,
                batch_size = 32, padding = 'same', kernel_initializer = None, kernel_regularizer = None, 
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
        #keras.regularizers.l2(weight_decay)
        self.epochs = epochs
        self.shuffle = shuffle
        
        self.model = None
        self.datagen = None
        
    def image_aug(self, X_train, **kwargs):
        datagen = ImageDataGenerator(**kwargs)
        datagen.fit(X_train)
        self.datagen = datagen
        
    def conv_def(self, filters, kernel_size, *, strides = 1, prev_layers = None):
        x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = self.padding,
                  kernel_initializer = self.ki, kernel_regularizer = self.kr)(prev_layers)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        
        return x
    
    def Inception(self, prev_layers, conv1_1, conv13_1, conv13_3, conv15_1, conv15_5, convp1_1):
        conv11_l = self.conv_def(filters = conv1_1, kernel_size = 1, prev_layers = prev_layers)
        
        conv13_l = self.conv_def(filters = conv13_1, kernel_size = 1, prev_layers = prev_layers)
        conv13_3_l = self.conv_def(filters = conv13_3, kernel_size = 3, prev_layers = conv13_l)
        
        conv15_l = self.conv_def(filters = conv15_1, kernel_size = 1, prev_layers = prev_layers)
        conv15_5_l = self.conv_def(filters = conv15_5, kernel_size = 5, prev_layers = conv15_l)
        
        convp1_l = MaxPooling2D(pool_size = 3, strides = 1, padding = self.padding)(prev_layers)
        convp1_1_l = self.conv_def(filters = convp1_1, kernel_size = 1, prev_layers = convp1_l)
        
        inception_out = concatenate([conv11_l, conv13_3_l, conv15_5_l, convp1_1_l], axis = -1)
        
        return inception_out
    
    def mini_Inception(self, prev_layers, conv1_1, conv3_1):
        conv11 = self.conv_def(filters = conv1_1, kernel_size = 1, prev_layers = prev_layers)
        conv33 = self.conv_def(filters = conv3_1, kernel_size = 3, prev_layers = prev_layers)
        
        inception_out = concatenate([conv11, conv33], axis = -1)
        
        return inception_out
    
    def downsample(self, prev_layers, filters):
        
        conv33 = self.conv_def(filters = filters, kernel_size = 3, strides = 2, prev_layers = prev_layers)
        pool = MaxPooling2D(pool_size = 3, strides = 2, padding = 'same')(prev_layers)
        
        downsample_out = concatenate([conv33, pool], axis = -1)
        
        return downsample_out
    
    def build_model(self, *, mini = False):
        
        input_layers = Input(shape = self.input_shape)
        
        if mini is False:
            
            conv1 = self.conv_def(filters = 64, kernel_size = 7, strides = 2, prev_layers = input_layers)
            conv1_pool = MaxPooling2D(pool_size = 3, strides = 2, padding = 'same')(conv1)
            
            conv2_1 = self.conv_def(filters = 64, kernel_size = 1, strides = 1, prev_layers = conv1_pool)
            conv2 = self.conv_def(filters = 192, kernel_size = 3, strides = 1, prev_layers = conv2_1)
            conv2_pool = MaxPooling2D(pool_size = 3, strides = 2, padding = 'same')(conv2)
            
            inception_3a = self.Inception(prev_layers = conv2_pool,
                                         conv1_1 = 64, conv13_1 = 96, conv13_3 = 128, conv15_1 = 16, conv15_5 = 32, convp1_1 = 32)
            inception_3b = self.Inception(prev_layers = inception_3a,
                                         conv1_1 = 128, conv13_1 = 128, conv13_3 = 192, conv15_1 = 32, conv15_5 = 96, convp1_1 = 64)
            inception_3_pool = MaxPooling2D(pool_size = 3, strides = 2, padding = 'same')(inception_3b)
            
            inception_4a = self.Inception(inception_3_pool,
                                         192, 96, 208, 16, 48, 64)
            inception_4b = self.Inception(inception_4a,
                                         160, 112, 224, 24, 64, 64)
            inception_4c = self.Inception(inception_4b,
                                         128, 128, 256, 24, 64, 64)
            inception_4d = self.Inception(inception_4c,
                                         112, 144, 144, 32, 64, 64)
            inception_4e = self.Inception(inception_4d,
                                         256, 160, 320, 32, 128, 128)
            inception_4_pool = MaxPooling2D(pool_size = 3, strides = 2, padding = 'same')(inception_4e)
            
            inception_5a = self.Inception(inception_4_pool,
                                         256, 160, 320, 32, 128, 128)
            inception_5b = self.Inception(inception_5a,
                                         384, 192, 384, 48, 128, 128)
            
            inception_avgpool = AveragePooling2D(pool_size = 7, strides = 1)(inception_5b)
            inception_avgpool_dropout = Dropout(0.4)(inception_avgpool)
            
            
        if mini:
            
            conv1 = self.conv_def(filters = 96, kernel_size = 3, prev_layers = input_layers)
            
            inception_3a = self.mini_Inception(prev_layers = conv1,
                                              conv1_1 = 32, conv3_1 = 32)
            inception_3b = self.mini_Inception(inception_3a,
                                              32, 48)
            
            downsample_1 = self.downsample(prev_layers = inception_3b, filters = 80)
            
            inception_4a = self.mini_Inception(downsample_1,
                                              112, 48)
            inception_4b = self.mini_Inception(inception_4a,
                                              96, 64)
            inception_4c = self.mini_Inception(inception_4b,
                                              80, 80)
            inception_4d = self.mini_Inception(inception_4c,
                                              48, 96)
            
            downsample_2 = self.downsample(inception_4d, 96)
            
            inception_5a = self.mini_Inception(downsample_2,
                                              176, 160)
            inception_5b = self.mini_Inception(inception_5a,
                                              176, 160)
            
            inception_avgpool = AveragePooling2D(pool_size = 7, strides = 1)(inception_5b)
            inception_avgpool_dropout = Dropout(0.5)(inception_avgpool)
        
        
        flat = Flatten()(inception_avgpool_dropout)
        out_dense = Dense(self.num_class, kernel_initializer = self.ki, kernel_regularizer = self.kr)(flat)
        out = Activation('softmax')(out_dense)
        
        model = Model(inputs = input_layers, outputs = out)
        
        opt = self.opt
        
        if self.y_shape == 1:
            model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
        if self.y_shape > 1:
            model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
        
        self.model = model
        
        return model
    
    def fit(self, X_train, y_train, *, X_valid = None, y_valid = None, **kwargs):
        if X_valid is not None:
            history = self.model.fit(X_train, y_train, epochs = self.epochs, validation_data = (X_valid, y_valid),
                                    batch_size = self.batch_size, shuffle = self.shuffle, **kwargs)
            
        else:
            history = self.model.fit(X_train, y_train, epochs = self.epochs,
                                    batch_size = self.batch_size, shuffle = self.shuffle, **kwargs)
        return history
    
        
    def fit_generator(self, X_train, y_train, *, X_valid = None, y_valid = None, **kwargs):
        if X_valid is not None and self.datagen is not None:
            history = self.model.fit_generator(self.datagen.flow(X_train, y_train, batch_size = self.batch_size),
                                              epochs = self.epochs,
                                              steps_per_epoch = len(X_train)/self.batch_size,
                                              validation_data = (X_valid, y_valid), **kwargs)
            
        else:
            history = self.model.fit_generator(self.datagen.flow(X_train, y_train, batch_size = self.batch_size),
                                              epochs = self.epochs,
                                              steps_per_epoch = len(X_train)/self.batch_size,
                                              **kwargs)
            
            
        return history
    
    
    def evaluate(self, X_test, y_test):
        score = self.model.evaluate(X_test, y_test)
        return score
    
    def predict(self, X_new):
        pred = self.model.predict(X_new)
        return pred
    

#original googlenet - input 224 x 224 RGB
googlenet_org = googlenet(input_shape = (224,224,3), y_shape = 1, activation = 'relu', num_class = 1000)
googlenet_org.build_model(mini = False)
googlenet_org.model.summary()

#mini googlenet - input 32 x 32 RGB
googlenet_mini = googlenet(input_shape = (32,32,3), y_shape = 1, activation = 'relu', num_class = 10)
googlenet_mini.build_model(mini = True)
googlenet_mini.model.summary()


#fitting and predicting
#googlenet_mini.fit(X_train = X_train, y_train = y_train, X_valid = X_valid, y_valid = y_valid)
#googlenet_mini.evaluate(X_valid = X_valid, y_valid = y_valid)
#googlenet_mini.predict(X_new = X_test)
