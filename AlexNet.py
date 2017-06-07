"""
This is a implement of AlexNet with tensorflow and fork from Frederik Kratzert
"""
import tensorflow as tf
import numpy as np

class AlexNet(object):
    def __init__(self, input_x, keep_prob, num_classes, skip_layer, weights_path = 'Default'):
        # Initialization the parameters
        self.input_x = input_x
        self.keep_prob = keep_prob
        self.skip_layer = skip_layer
        if weights_path == 'Default' :
            self.weights_path = 'bvlc_alexnet.npy'
        else:
            self.weights_path = weights_path
        self.num_classes = num_classes
        # Create the AlexNet
        self.create()

    def create(self):
        conv1 = self.conv(self.input_x,11,96,4,name = 'conv1', padding = 'VALID')
        pool1 = self.maxPooling(conv1, filter_size = 3, stride = 2, name = 'pool1', padding = 'VALID')
        norm1 = self.lrn(pool1,2,2e-05,0.75,name='norm1')


        conv2 = self.conv(norm1,5,256,1,name = 'conv2',padding_num = 0, groups = 2)
        pool2 = self.maxPooling(conv2, filter_size = 3, stride = 2, name = 'pool2', padding = 'VALID')
        norm2 = self.lrn(pool2,2,2e-05,0.75,name='norm2')

        conv3 = self.conv(norm2, 3, 384, 1, name = 'conv3')

        conv4 = self.conv(conv3, 3, 384, 1, name = 'conv4',groups = 2)

        conv5 = self.conv(conv4, 3, 256, 1, name = 'conv5', groups = 2)
        pool5 = self.maxPooling(conv5, filter_size = 3, stride = 2, name= 'pool5', padding = 'VALID')

        flattened = tf.reshape(pool5, [-1,6*6*256])
        fc6 = self.fc(input = flattened, num_in = 6*6*256, num_out = 4096, name = 'fc6', drop_ratio = 1.0-self.keep_prob, relu = True)
        
        fc7 = self.fc(input = fc6, num_in = 4096, num_out = 4096, name = 'fc7', drop_ratio = 1.0 - self.keep_prob, relu = True)

        self.fc8 = self.fc(input = fc7, num_in = 4096, num_out = self.num_classes, name = 'fc8', drop_ratio = 0, relu = False)

    def load_weights(self, session):
        weights_dict = np.load(self.weights_path, encoding = 'bytes').item()

        for op_name in weights_dict:
             if op_name not in self.skip_layer:
                 with tf.variable_scope(op_name, reuse = True):
                     for data in weights_dict[op_name]:
                         if len(data.shape) == 1:
                             var = tf.get_variable('biases',trainable=False)
                             session.run(var.assign(data))
                         else:
                            var = tf.get_variable('weights',trainable=False)
                            session.run(var.assign(data))
    
    def conv(self, x, kernel_height, num_kernels, stride, name, padding = 'SAME',padding_num = 0,groups = 1):
        print ('name is {} np.shape(input) {}'.format(name, np.shape(x)))
        input_channels = int(np.shape(x)[-1])
        if not padding_num == 0:
            x = tf.pad(x,[[0,0],[padding_num,padding_num],[padding_num,padding_num],[0,0]])
        convolve = lambda i,k:tf.nn.conv2d(i,k, strides = [1, stride, stride ,1], padding = padding)
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape = [kernel_height, kernel_height, input_channels/groups, num_kernels])
            biases = tf.get_variable('biases', shape = [num_kernels])
        if groups == 1:
            conv = convolve(x,weights)
        else:
            input_groups = tf.split(axis=3,num_or_size_splits = groups, value = x)
            weights_groups = tf.split(axis = 3, num_or_size_splits = groups, value = weights)
            output_groups = [convolve(i,k) for i,k in zip(input_groups,weights_groups)]

            conv = tf.concat(axis = 3, values = output_groups)

        # add biases and avtive function
        withBias = tf.reshape(tf.nn.bias_add(conv,biases),conv.get_shape().as_list())
        relu = tf.nn.relu(withBias)
        return relu

    def maxPooling(self, input,filter_size,stride,name,padding = 'SAME'):
        print ('name is {} np.shape(input) {}'.format(name,np.shape(input)))
        return tf.nn.max_pool(input,ksize=[1,filter_size,filter_size,1],strides = [1,stride,stride,1],padding = padding, name = name)

    def lrn(self, input,radius,alpha,beta,name,bias = 1.0):
        print ('name is {} np.shape(input) {}'.format(name,np.shape(input)))
        return tf.nn.local_response_normalization(input,depth_radius=radius, alpha=alpha,beta=beta,bias=bias,name=name)

    def fc(self, input,num_in,num_out,name,drop_ratio=0,relu = True):
        print ('name is {} np.shape(input) {}'.format(name,np.shape(input)))
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights',shape = [num_in,num_out],trainable=True)
            biases = tf.get_variable('biases',[num_out],trainable=True)
            # Linear 
            act = tf.nn.xw_plus_b(input,weights,biases,name=scope.name)

            if relu == True:
                relu = tf.nn.relu(act)
                if drop_ratio == 0:
                    return relu
                else:
                    return tf.nn.dropout(relu,1.0-drop_ratio)
            else:
                if drop_ratio == 0:
                    return act
                else:
                    return tf.nn.dropout(act,1.0-drop_ratio)

    





