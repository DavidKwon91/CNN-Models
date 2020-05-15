import keras
from keras import optimizers
from keras import initializers
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input, concatenate, Dropout, BatchNormalization, Add, Layer, DepthwiseConv2D, Reshape, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


#reference
#https://arxiv.org/pdf/1801.04381.pdf - original paper
#https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py
#https://github.com/xiaochus/MobileNetV2/blob/master/mobilenet_v2.py


class mobilenet_v2():
    def __init__(self, input_shape, y_shape, num_class, *,
                batch_size = 128, padding = 'same', 
                 bn_momentum = 0.999, bn_eps = 1e-5,
                kernel_initializer = None, kernel_regularizer = None,
                epochs = 30, shuffle = False, opt = None):
        
        self.input_shape = input_shape
        self.y_shape = y_shape
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
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        
        self.model = None
        self.datagen = None
        
        self.inv_block = None
        
    @staticmethod
    def _make_divisible(x, *, divisor = 8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_x = max(min_value, int(x + divisor / 2) // divisor * divisor)
        if new_x < 0.9 * x:
            new_x += divisor
        return new_x

        
    @staticmethod
    def _relu6(x):
        return K.relu(x, max_value = 6.0)
    
    def _bn_def(self, prev_layers):
        return BatchNormalization(momentum = self.bn_momentum, epsilon = self.bn_eps)(prev_layers)
    
    @staticmethod
    def _correct_pad(x, *, kernel_size = 3):
        prev_size = K.int_shape(x)[1:3] #channels_last
        
        adjust = (1 - prev_size[0] % 2, 1 - prev_size[1] % 2)
        
        correct = kernel_size // 2
        
        return ((correct - adjust[0], correct),
               (correct - adjust[1], correct))
    
    
    def _zero_pad(self, prev_layers):
        return ZeroPadding2D(padding = self._correct_pad(prev_layers))(prev_layers)
    
    def _conv_def(self, filters, kernel_size, prev_layers, *,
                 use_bias = False, padding = 'same', strides = 1, **kwargs):
        x = Conv2D(filters = filters, kernel_size = kernel_size, padding = padding,
                   strides = strides,
                   use_bias = use_bias,
                  kernel_initializer = self.ki, 
                   kernel_regularizer = self.kr, **kwargs)(prev_layers)
        return x
    
    
    def _bottleneck(self, filters, strides, t, prev_layers, depth_kernel, alpha, *,
                      res_con = False, first = False, downsample = True):
        
        tk = K.int_shape(prev_layers)[-1] * t
        k_prime = self._make_divisible(int(filters*alpha)) #pointwise filter

        if first:
            
            first_cv_filter = self._make_divisible(x = 32 * alpha)
            
            if downsample:
                x = self._conv_def(filters = 32, kernel_size = 3, strides = 2, 
                              padding = 'valid', prev_layers = prev_layers)
            else:
                x = self._conv_def(filters = 32, kernel_size = 3, strides = 1, 
                              padding = 'valid', prev_layers = prev_layers)
                
            x = self._bn_def(prev_layers = x)
            x = Activation(self._relu6)(x)


        elif first is False:
            x = self._conv_def(tk, kernel_size = 1, prev_layers = prev_layers)
            x = self._bn_def(prev_layers = x)
            x = Activation(self._relu6)(x)

        if strides > 1:
            padding = 'valid'
            zpad = self._zero_pad(prev_layers = x)
            dcv = DepthwiseConv2D(kernel_size = depth_kernel, strides = strides, padding = padding, use_bias = False)(zpad)
            
        elif strides == 1:
            padding = 'same'
            dcv = DepthwiseConv2D(kernel_size = depth_kernel, strides = strides,
                               padding = padding, use_bias = False)(x)
        x = self._bn_def(prev_layers = dcv)
        x = Activation(self._relu6)(x)

        x = self._conv_def(k_prime, kernel_size = 1, prev_layers = x)
        x = self._bn_def(prev_layers = x)

        if res_con:
            return Add()([prev_layers, x])
        
        return x


    def _inverted_residual_block(self, filters, strides, t, n,
                                prev_layers, depth_kernel, alpha, *, 
                                 downsample = True, first = False):

        x = self._bottleneck(filters = filters, strides = strides, t = t, prev_layers = prev_layers,
                           depth_kernel = depth_kernel, first = first, downsample = downsample, alpha = alpha)
        
        if n > 1:
            for i in range(1,n):
                x = self._bottleneck(filters = filters, strides = 1, t = t, prev_layers = x, 
                                   depth_kernel = depth_kernel, alpha = alpha, res_con = True)
            
        return x


    def build_model(self, t, c, n, s,
                        *, out_filter = 1280, depth_kernel = 3, alpha = 1.0, downsample = True, dropout = 0.0):
        
        """
        ------------------------------------------------------------------------------
        #Arguments
        For t, c, n, and s, it's exactly the same as the paper stated.
        
            t (int list) : int list for the expansion factor - for pointwise convolution filters, "The exapansion factor is applied to the input size"
            c (int list) : int list for the output channels of each blocks - the out filters of convoltuion layers of each blocks, "All layers in the same sequence have the same number of c of ouput channels"
            n (int list) : int list for the number of repeated layers - layers repeated n times
            s (int list) : the first strides of the first layer of each sequence and all others use stride 1
            
            out_filter (int) : int value for the filter of very last convolution layer
            depth_kernel (int) : int value for the kernel size of depthwise layer, it's fixed value in the paper
            alpha (int) : int value for the width parameter (or width multiplier), alpha = 1.0 is used for the primary network in the paper
            downsample (boolean) : boolean value for the stride of the very first convolution layer
            dropout (float) : float value for the dropout ratio, (0.0 - 0.9), if its value is 0, then no dropout layers is applied
        
        #Returns
        
            model (Model) : Keras model instance, compiled
        ------------------------------------------------------------------------------
        """

        input_layers = Input(shape = self.input_shape)
        zpad1 = self._zero_pad(prev_layers = input_layers)
        
        

        x = self._inverted_residual_block(filters = c[0], 
                                         strides = s[0],
                                         t = t[0], 
                                         n = n[0], 
                                         depth_kernel = depth_kernel, alpha = alpha,
                                         first = True, downsample = downsample, prev_layers =zpad1)


        #from second mb and onwards
        for c,t,s,n in zip(c[1:], t[1:], s[1:], n[1:]):
            x = self._inverted_residual_block(filters = c, t = t, strides = s, n = n, 
                                             depth_kernel = depth_kernel, alpha = alpha, 
                                             prev_layers = x)
        if dropout:
            x = Dropout(dropout)(x)
        
            
        out_filter = self._make_divisible(out_filter * alpha)

        x = self._conv_def(out_filter, kernel_size = 1, prev_layers = x)
        x = self._bn_def(prev_layers = x)
        x = Activation('relu')(x)
        
        if dropout: 
            x = Dropout(dropout)(x)
        
        x = GlobalAveragePooling2D()(x)
        
        x = Dense(self.num_class, use_bias = True, activation = 'softmax', 
                          kernel_initializer= self.ki, kernel_regularizer = self.kr)(x)
        
        model = Model(inputs = input_layers, outputs = x)
        
        opt = self.opt
        
        if self.y_shape == 1:
            model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
        
        elif self.y_shape > 1:
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

    
#Original
mb2_org = mobilenet_v2(input_shape = (224,224,3), y_shape = 1, num_class = 10)
mb2_org.build_model(c = [16, 24, 32, 64, 96, 160, 320],
                   t = [1, 6, 6, 6, 6, 6, 6],
                   s = [1, 2, 2, 2, 1, 2, 1],
                   n = [1, 2, 3, 4, 3, 3, 1], alpha = 1.0)
mb2_org.model.summary()
#Global Avg layer takes (7 x 7 x 1280)


#Customized version - cifar 10
weight_decay = 1e-5
mb2_custom = mobilenet_v2(input_shape = (32,32,3), y_shape = 10, num_class = 10, 
                         kernel_initializer = initializers.he_normal(), 
                          kernel_regularizer = regularizers.l2(weight_decay),
                         opt = optimizers.RMSprop(lr = 1e-3),
                         bn_momentum = 0.9)

mb2_custom.build_model(c = [16, 24, 32, 64, 96, 160, 320],
                       t = [1, 6, 6, 6, 6, 6, 6],
                       s = [1, 1, 1, 2, 1, 2, 1],
                       n = [1, 2, 3, 4, 3, 3, 1], 
                       downsample = False, #default True
                       dropout = 0.3, alpha = 1.0, out_filter = 1280)

mb2_custom.model.summary()
#Global Avg layer takes (8 x 8 x 1280)

mb2_custom.build_model(c = [16, 24, 32, 64, 96, 160, 320],
                       t = [1, 6, 6, 6, 6, 6, 6],
                       s = [1, 1, 1, 2, 1, 2, 1],
                       n = [1, 2, 3, 4, 3, 3, 1], 
                       downsample = True,
                       dropout = 0.3, alpha = 1.0, out_filter = 1280)

mb2_custom.model.summary()
#Global Avg layer takes (4 x 4 x 1280)