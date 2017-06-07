import os
import numpy as np 
import cv2
import tensorflow as tf 
from datetime import datetime
from AlexNet import AlexNet
from Datagenerator import ImageDataGenerator

def test_image(path_image,num_class,path_classes):
    #x = tf.placeholder(tf.float32, [1,227,227,3])
    
    x = cv2.imread(path_image)
    x = cv2.resize(x,(227,227))
    x = x.astype(np.float32)
    x = np.reshape(x,[1,227,227,3])
    y = tf.placeholder(tf.float32,[None,num_class])
    model = AlexNet(x,0.5,1000,'')
    score = model.fc8
    max = tf.arg_max(score,1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.load_weights(sess)
        #score = model.fc8
        label_id = sess.run(max)[0]

        with open(path_classes) as f:
            lines = f.readlines()
            label = lines[label_id]
            print('image name is {} class_id is {} class_name is {}'.format(path_image,label_id,label))
        f.close()

test_image('C:/Users/Rain/finetune_alexnet_with_tensorflow/images/zebra.jpeg',1000,'caffe_classes.py')