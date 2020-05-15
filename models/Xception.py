##import packages
import keras
from keras import optimizers
from keras import initializers
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input, concatenate, Dropout, BatchNormalization, add, Layer, SeparableConv2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

#reference
#https://github.com/beinanwang/tf-slim-xception-cifar-10

class xception():
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
        
        self.input_layers = None
        
        
        
    def image_aug(self, X_train, **kwargs):
        datagen = ImageDataGenerator(**kwargs)
        datagen.fit(X_train)
        self.datagen = datagen
        
    @staticmethod
    def _maxpool(x, *, strides = 2):
        return MaxPooling2D(pool_size = 3, strides = strides, padding = 'same')(x)
    
    @staticmethod
    def _bn_def(x):
        return BatchNormalization(momentum = 0.9, epsilon = 1e-5)(x)
    
    #convolution layer for convenient uses
    def _conv_def(self, filters, *, kernel_size = 3, strides = 1, x = None, **kwargs):
        
        x = Conv2D(filters = filters, kernel_size = kernel_size, padding = self.padding, strides = strides,
                  kernel_initializer = self.ki, kernel_regularizer = self.kr, **kwargs)(x)
        x = self._bn_def(x = x)
        
        return x
    
    #seperable conv layer for convenient uses
    def _sep_conv_def(self, filters, *, kernel_size = 3, strides = 1, x = None, last=False, **kwargs):
        
        if last:
            x = SeparableConv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = self.padding,
                           kernel_initializer = self.ki, kernel_regularizer = self.kr, **kwargs)(x)
            x = self._bn_def(x = x)
        
        else:
            x = SeparableConv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = self.padding,
                           kernel_initializer = self.ki, kernel_regularizer = self.kr, **kwargs)(x)
            x = self._bn_def(x = x)
            x = Activation(self.activation)(x)
        
        return x
    
    
    
    #flow structure function
    def _xception_flow(self, filters, prev_layers, *, flow = 'entry', middleflow_stack = 8, downsample_strides = 2):

        if len(filters) == 1 and flow != 'middle':
            raise ValueError("You need different filters for each modules in the flow, unless you are structuring middle flow")

        if len(filters) > 1 and flow == 'middle':
            raise ValueError("If you are structuring the middle flow of the Xception, you need only 1 filter")

        if flow not in set(['entry', 'middle', 'exit']):
            raise ValueError("You need to specify proper flow")


        #The 36 convolutional layers are structured into 14 modules, 
        #all of which have linear residual connections around them, except for the first and last modules.
        
        if flow == 'entry':

            if len(filters) != 3:
                raise ValueError('You need 3 filters for each modules in the entry flow')

            #1
            shortcut = self._conv_def(filters = filters[0], kernel_size = 1, strides = downsample_strides, 
                                     use_bias = False, x = prev_layers)

            x = self._sep_conv_def(filters = filters[0], use_bias = False, x = prev_layers)
            x = self._sep_conv_def(filters = filters[0], last = True, use_bias = False, x = x)
            x = self._maxpool(x = x, strides = downsample_strides)

            block = add([x, shortcut])

            #2
            shortcut = self._conv_def(filters = filters[1], kernel_size = 1, strides = downsample_strides, 
                                     use_bias = False, x = block)

            x = Activation(self.activation)(block)
            x = self._sep_conv_def(filters = filters[1], use_bias = False, x = x)
            x = self._sep_conv_def(filters = filters[1], last = True, use_bias = False, x = x)
            x = self._maxpool(x = x, strides = downsample_strides)

            block = add([x,shortcut])

            #3
            shortcut = self._conv_def(filters = filters[2], kernel_size = 1, strides = 2, 
                                     use_bias = False, x = block)

            x = Activation(self.activation)(block)
            x = self._sep_conv_def(filters = filters[2], use_bias = False, x = x)
            x = self._sep_conv_def(filters = filters[2], last = True, use_bias = False, x = x)
            x = self._maxpool(x = x)

            flow_out = add([x,shortcut])


        if flow == 'middle':
            filters = filters[0]

            shortcut = prev_layers

            #repeat 8 times as stated in the paper
            
            x = Activation(self.activation)(prev_layers)
            x = self._sep_conv_def(filters = filters, x = x, use_bias = False)
            x = self._sep_conv_def(filters = filters, x = x, use_bias = False)
            x = self._sep_conv_def(filters = filters, x = x, last = True, use_bias = False)
            x = add([x, shortcut])
            
            for i in range(1, middleflow_stack):
                shortcut = x
                x = Activation(self.activation)(shortcut)
                x = self._sep_conv_def(filters = filters, x = x, use_bias = False)
                x = self._sep_conv_def(filters = filters, x = x, use_bias = False)
                x = self._sep_conv_def(filters = filters, x = x, last = True, use_bias = False)
                flow_out = add([x, shortcut])


        if flow == 'exit':

            if len(filters) != 4:
                raise ValueError("You need 4 filters for each modules in exit flow")

            shortcut = self._conv_def(filters = filters[1], kernel_size = 1, strides = 2, 
                                     use_bias = False, x = prev_layers)

            x = Activation(self.activation)(prev_layers)
            x = self._sep_conv_def(filters = filters[0], x = x, use_bias = False)
            x = self._sep_conv_def(filters = filters[1], x = x, last = True, use_bias = False)

            x = self._maxpool(x = x)

            block = add([x, shortcut])

            #the last module
            x = self._sep_conv_def(filters = filters[2], x = x, use_bias = False)

            flow_out = self._sep_conv_def(filters = filters[3], x = x, use_bias = False)


        return flow_out
    
    def define_flow(self, filters, flow, *, prev_flow = None, downsample_strides = 2):
        
        
        """
        ------------------------------------------------------------------------------
        #Arguments
        
            filters (list or tuple) : int list or tuple for the structure of the filters of convolution layers in each flows
            flow (string) : string value to specify which flow is built among, 'entry', 'middle', and 'exit'
            prev_flow (Layer) : Keras layer instance, which is the flow layers
            downsample_strides (int) : int value for the strides in the first two conv and maxpooling layers in the entry flow
                This can be 1 in small version of xception
        
        #Returns
        
            layer (Layer) : Keras layer instance, which is the flow
        ------------------------------------------------------------------------------
        """
        
        if flow == 'entry':
            
            input_layers = Input(shape = self.input_shape)

            #the first module
            
            #32 x 32 x 32
            x = self._conv_def(32, strides = 1, use_bias = False, x = input_layers)
            x = Activation(self.activation)(x)
            
            #32 x 32
            x = self._conv_def(64, use_bias = False, x = x)
            x = Activation(self.activation)(x)
            
            x = self._xception_flow(filters = filters, prev_layers = x, flow = 'entry',
                                   downsample_strides = downsample_strides)
            
            self.input_layers = input_layers
            
        else:
            
            x = self._xception_flow(filters = filters, flow = flow, prev_layers = prev_flow)
            
        return x
        
        
    def build_model(self, flow):
        
        
        """
        ------------------------------------------------------------------------------
        #Arguments
        
            flow (Layer) : Keras layer instance, which is one of flows as stated in paper; entry, middle, and exit flow
            
        #Returns
        
            model (Model) : Keras model instance, compiled
            
            
        Before structure the whole model, the flows must be structured first with define_flow function
        
        #Example
            
            xcp = xception(...) #instantiate xception class
            entry_flow = xcp.define_flow(filters = [128, 256, 728], flow = 'entry')
            middle_flow = xcp.define_flow(filters = [728], flow = 'middle', prev_flow = entry_flow)
            exit_flow = xcp.define_flow(filters = [728, 1024, 1536, 2048], flow = 'exit', prev_flow = middle_flow)
            xcp_model = xcp.build_model(flow = exit_flow)
            ...
        ------------------------------------------------------------------------------
        """
        
        if flow is None:
            raise ValueError("You need the flow of Xception model")
            
        global_pool = GlobalAveragePooling2D()(flow)
        out_dense = Dense(self.num_class, activation = 'softmax', 
                          kernel_initializer= self.ki, kernel_regularizer = self.kr)(global_pool)
        
        opt = self.opt
        
        model = Model(inputs = self.input_layers, outputs = out_dense)
        
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
    
    
    
#original Xception model
xcp_org = xception(input_shape = (299, 299, 3), y_shape = 1000, num_class = 1000, activation = 'relu')

#define each flow and connect along with the flow
entry_flow = xcp_org.define_flow(filters = [128, 256, 728], flow = 'entry')
middle_flow = xcp_org.define_flow(filters = [728], flow = 'middle', prev_flow = entry_flow)
exit_flow = xcp_org.define_flow(filters = [728, 1024, 1536, 2048], flow = 'exit', prev_flow = middle_flow)

xception_original_model = xcp_org.build_model(flow = exit_flow)


#Customized Xception model, which is small version

xcp_mini = xception(input_shape = (32,32,3), y_shape = 1, num_class = 10, activation = 'relu')

#define flow
entry_flow = xcp_mini.define_flow(filters = [128, 256, 728], flow = 'entry', downsample_strides = 1)
exit_flow = xcp_mini.define_flow(filters = [728,1024,1536,2048], prev_flow = entry_flow, flow = 'exit')

xcp_mini_model = xcp_mini.build_model(exit_flow)


#fitting and predicting
#xcp_mini.fit(X_train = X_train, y_train = y_train, X_valid = X_valid, y_valid = y_valid)
#xcp_mini.evaluate(X_valid = X_valid, y_valid = y_valid)
#xcp_mini.predict(X_new = X_test)




(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

y_train = keras.utils.to_categorical(y_train, 10)
y_valid = keras.utils.to_categorical(y_valid, 10)
y_test = keras.utils.to_categorical(y_test, 10)

X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')

X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims = True)
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std


xcp_mini = xception(input_shape = X_train.shape[1:], y_shape = 10, num_class = 10, activation = 'relu',
                   epochs = 5)

#define flow
entry_flow = xcp_mini.define_flow(filters = [128, 256, 728], flow = 'entry', downsample_strides = 1)
exit_flow = xcp_mini.define_flow(filters = [728,1024,1536,2048], prev_flow = entry_flow, flow = 'exit')

xcp_mini_model = xcp_mini.build_model(exit_flow)


#fitting and predicting
#xcp_mini.fit(X_train = X_train, y_train = y_train, X_valid = X_valid, y_valid = y_valid)
#xcp_mini.evaluate(X_valid = X_valid, y_valid = y_valid)
#xcp_mini.predict(X_new = X_test)
