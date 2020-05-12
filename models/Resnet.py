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
from keras.utils import plot_model

from keras.datasets import cifar10






class resnet():
    def __init__(self, input_shape, y_shape, activation, num_class, *,
                batch_size = 128, padding = 'same', 
                kernel_initializer = None, kernel_regularizer = None,
                epochs = 30, shuffle = False, opt = None):
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
        
        
    @staticmethod
    def maxpool(x):
        return MaxPooling2D(pool_size = 3, strides = 2, padding = self.padding)(x)
    
    @staticmethod
    def bn_def(x):
        return BatchNormalization(momentum = 0.9, epsilon = 1e-5)(x)
    
    def conv_def(self, filters, kernel_size, *, strides = 1, x = None, **kwargs):
        
        x = Conv2D(filters = filters, kernel_size = kernel_size, padding = self.padding, strides = strides,
                  kernel_initializer = self.ki, kernel_regularizer = self.kr, **kwargs)(x)
        
        return x
    
    
    def residual2(self, filters, prev_layers, *, first = False, first_of_first = False):
        '''Residual module for [3 x 3]'''
        
        a1 = Activation(self.activation)(prev_layers)
        
        #first layer but not the first block
        if first:
            strides = 2
            shortcut = self.conv_def(filters = filters, kernel_size = 1, strides = strides, x = a1)
            
        #second layer and onwards
        if first is False:
            strides = 1
            shortcut = a1
            
        #very first layer of the first block
        if first_of_first:
            strides = 1
            a1 = Activation(self.activation)(self.bn_def(x = prev_layers))
            shortcut = a1
        
        c1 = self.conv_def(filters = filters, kernel_size = 3, strides = strides, x = a1)
        a2 = Activation(self.activation)(self.bn_def(x = c1))
        c2 = self.conv_def(filters = filters, kernel_size = 3, strides = 1, x = a2)
        c2 = self.bn_def(x = c2)

        block = add([c2, shortcut])

        return block
    
    
    def residual3(self, filter1, filter2, prev_layers, *, first = False, first_of_first = False):
        '''Residual Module for [1 x 3 x 1]'''
        
        
        a1 = Activation(self.activation)(self.bn_def(x = prev_layers))
        
        #first layer but not the first block
        if first:
            strides = 2
            shortcut = self.conv_def(filters = filter2, kernel_size = 1, strides = strides, x = prev_layers)
            
        #second layer and onwards
        if first is False:
            strides = 1
            shortcut = prev_layers
        
        #very first layer of the first block
        if first_of_first:
            strides = 1
            shortcut = self.conv_def(filters = filter2, kernel_size = 1, strides = strides, x = a1)
            
        
        c1 = self.conv_def(filters = filter1, kernel_size = 1, strides = strides, x = a1)
        a2 = Activation(self.activation)(self.bn_def(x = c1))
        c2 = self.conv_def(filters = filter1, kernel_size = 3, strides = 1, x = a2)
        
        a3 = Activation(self.activation)(self.bn_def(x = c2))
        c3 = self.conv_def(filters = filter2, kernel_size = 1, strides = 1, x = a3)
        
        block = add([c3, shortcut]) 
        
        return block
    
    
    def residual_structure(self, filter_structure, prev_layers, structure_stack, *, 
                           filter_structure2 = None, res3 = False):
        
        #[3 x 3]
        if res3 is False:
            
            #Very virst layer of the first residual blocks, which the strides is 1
            x = self.residual2(filters = filter_structure[0], prev_layers = prev_layers, first_of_first = True)
            #from the second layer of the first residual block
            for i in range(1, structure_stack[0]):
                x = self.residual2(filters = filter_structure[0], prev_layers = x)
                
                
            #from second block and onwards, which the strides of the first layer is 2
            for f,s in zip(filter_structure[1:], structure_stack[1:]):
                x = self.residual2(filters = f, prev_layers = x, first = True)
                #stride is 1
                for k in range(1, s):
                    x = self.residual2(filters = f, prev_layers = x)
        
        #[1 x 3 x 1]
        if res3:
            
            
            #Very virst layer of the first residual blocks, which the strides is 1
            x = self.residual3(filter1 = filter_structure[0], filter2 = filter_structure2[0], prev_layers = prev_layers, first_of_first = True)
            for i in range(1, structure_stack[0]):
                x = self.residual3(filter1 = filter_structure[0], filter2 = filter_structure2[0], prev_layers = x)
            
            #from second block and onwards, which the strides of the first layer is 2
            for f1, f2, s in zip(filter_structure[1:], filter_structure2[1:], structure_stack[1:]):
                x = self.residual3(filter1 = f1, filter2 = f2, prev_layers = x, first= True)
                for k in range(1,s):
                    x = self.residual3(filter1 = f1, filter2 = f2, prev_layers = x)
                
        return x
    
    
    def build_custom_resnet(self, filter_structure, structure_stack, *,
                            filter_structure2 = None, start_filter = 64 , start_kernel = 7, start_strides = 2):
        
        """
        
        #Arguments
        
            filter_structure (list or tuple): int list for the filters in the residual module
            filter_structure2 (list or tuple) : int list for the filters in the residual module for the stacks of (1 x 1) - (3 x 3) - (1 x 1) known as bottlenet layer
                This is not necessary for the stacks of (3 x 3) - (3 x 3) 
            start_filter (int) : number of filter for the first convolution layer of resnet
            start_kernel (int) : number of kernel_size for the first convolution layer of resnet
            start_strides (int) : number of strides for the first convolution layer of resnet
        
        #Returns
            
            model (Model) : Keras model instance
        
        
        ex1) filter_structure = [16,32,64], structure_stack = [5,5,5]
        You are structuring total 32 layers that contains..
        1 conv
        5 resnet module [3 x 3] (5 x 2 = 10 layers), filter of each conv is 16
        5 resnet module [3 x 3] (5 x 2 = 10 layers), filter of each conv is 32
        5 resnet module [3 x 3] (5 x 2 = 10 layers), filter of each conv is 64
        1 FC 
        
        ex2) 
        filter_structure = [64, 128, 256, 512],
        filter_structure2 = [256, 512, 1024, 2048], structure_stack = [3,4,6,3], 
        
        You are structuring total 50 layers that contains..
        1 conv
        3 resnet module [1 x 3 x 1] (3 x 3 = 9 layers), filters are 64, 64, and 256 respectively.
        4 resnet module [1 x 3 x 1] (3 x 4 = 12 layers), filters are 128, 128, and 512 respectively.
        6 resnet module [1 x 3 x 1] (3 x 6 = 18 layers), filters are 256, 256, and 1024 respectively.
        3 resnet module [1 x 3 x 1] (3 x 3 = 9 layers), filters are 512, 512, and 2048 respectively.
        1 FC
        
        
        
        """
        
        input_layers = Input(shape = self.input_shape)
        
        
        conv1 = self.conv_def(filters = start_filter, kernel_size = start_kernel,
                             strides = start_strides, x = input_layers)
        
        #small resnet [3 x 3]
        if filter_structure2 is None:
            x = self.residual_structure(filter_structure = filter_structure, prev_layers = conv1, 
                                        structure_stack = structure_stack, res3 = False)
            final_act = Activation(self.activation)(x)
        
        #deep resnet [1 x 3 x 1] bottleneck layers
        if filter_structure2 is not None:
            x = self.residual_structure(filter_structure = filter_structure, filter_structure2 = filter_structure2, 
                                        prev_layers = conv1, structure_stack = structure_stack, res3 = True)
            final_bn = self.bn_def(x = x)
            final_act = Activation(self.activation)(final_bn)
        
        global_pool = GlobalAveragePooling2D()(final_act)
        out_dense = Dense(self.num_class, activation = 'softmax', 
                          kernel_initializer= self.ki, kernel_regularizer = self.kr)(global_pool)
        
        opt = self.opt
        
        model = Model(inputs = input_layers, outputs = out_dense)
        
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
        '''Return predictied value'''
        pred = self.model.predict(X_new)
        return pred
    
    

resnet18 = resnet(input_shape = (224,224,3), y_shape = 1, activation = 'relu', num_class = 1000)
resnet18.build_custom_resnet(filter_structure = [64, 128, 256, 512], structure_stack = [2,2,2,2],
                             start_filter = 64, start_kernel = 7, start_strides = 2, after_pooling = True)

resnet18.model.summary()

resnet50 = resnet(input_shape = (224,224,3), y_shape = 1, activation = 'relu', num_class = 1000)
resnet50.build_custom_resnet(filter_structure = [64,128,256,512], filter_structure2 = [256,512,1024,2048], structure_stack = [3,4,6,3], 
                             start_filter = 64, start_kernel = 7, start_strides = 2, after_pooling = False) 

resnet50.model.summary()

resnet152 = resnet(input_shape = (224,224,3), y_shape = 1, activation = 'relu', num_class =1000)
resnet152.build_custom_resnet(filter_structure = [64,128,256,512], filter_structure2 = [256,512,1024,2048], structure_stack = [3,8,36,3], after_pooling = True)

resnet152.model.summary()


#customized for input shape (32,32,3) - cifar10 as stated in the paper
#https://arxiv.org/pdf/1512.03385.pdf

#These versions of resnet are exactly the same as in the keras example
#https://keras.io/examples/cifar10_resnet/

weight_decay = 1e-5

#version 1 [3 x 3]
#Total layers : 6 * n + 2 for n = 3      == 20
resnet_ver1 = resnet(input_shape = (32,32,3), y_shape = 1, activation = 'relu', num_class = 10, kernel_regularizer = keras.regularizers.l2(weight_decay), kernel_initializer = 'he_normal')
resnet_ver1.build_custom_resnet(filter_structure = [16, 32, 64], structure_stack = [3,3,3], start_filter = 16, start_kernel = 3, start_strides = 1)

resnet_ver1.model.summary()


#version 2 [1 x 3 x 1]
#Total layers : 9 * n + 2 for n = 3      == 29
resnet_ver2 = resnet(input_shape = (32,32,3), y_shape = 1, activation = 'relu', num_class = 10, kernel_regularizer = keras.regularizers.l2(weight_decay), kernel_initializer = 'he_normal')
resnet_ver2.build_custom_resnet(filter_structure = [16,64,128], filter_structure2 = [64, 128, 256], structure_stack = [3,3,3], start_filter = 16, start_kernel = 3, start_strides = 1)

resnet_ver2.model.summary()
