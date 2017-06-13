import os
import numpy as np 
import cv2
import tensorflow as tf 
from datetime import datetime
from AlexNet import AlexNet
from Datagenerator import ImageDataGenerator

def test_image(path_image,num_class,path_classes,weights_path = 'Default'):
    #x = tf.placeholder(tf.float32, [1,227,227,3])
    x = cv2.imread(path_image)
    x = cv2.resize(x,(227,227))
    x = x.astype(np.float32)
    
    x = np.reshape(x,[1,227,227,3])
    y = tf.placeholder(tf.float32,[None,num_class])
    model = AlexNet(x,0.5,1000,skip_layer = '', weights_path = weights_path)
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
            cv2.imshow(label,cv2.imread(path_image))
            cv2.waitKey(0)
        f.close()
        

test_image('C:/Users/Rain/finetune_alexnet_with_tensorflow/images/zebra.jpeg',1000,'caffe_classes.py')

def trainModels():
    train_file = 'car-train.txt'
    val_file = 'car-test.txt'
    learning_rate = 0.01
    num_epochs = 10
    batch_size = 1
    dropout_rate = 0.5
    num_class = 15
    train_layers = ['fc8','fc7']
    display_step = 1
    filewreiter_path = 'filewriter/cars'
    checkpoint_path = 'filewriter/'
    
    x = tf.placeholder(tf.float32,shape=[batch_size,227,227,3])
    y = tf.placeholder(tf.float32,[None,num_class])
    keep_prob = tf.placeholder(tf.float32)

    model = AlexNet(input_x = x,keep_prob = keep_prob,num_classes = num_class, skip_layer = train_layers,weights_path = 'Default')
    score = model.fc8
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
    with tf.name_scope('cross_ent'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,labels=y))
    with tf.name_scope('train'):
        gradients = tf.gradients(ys = loss, xs = var_list)
        gradients = list(zip(gradients,var_list))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars = gradients)

        for gradient,var in gradients:
            tf.summary.histogram(var.name+'/gradient',gradient)
        for var in var_list:
            tf.summary.histogram(var.name,var)
        tf.summary.scalar('cross_entropy',loss)
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.arg_max(score,1),tf.arg_max(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        
        tf.summary.scalar('accuracy',accuracy)
        merged_summary = tf.summary.merge_all()

        writer = tf.summary.FileWriter(filewreiter_path)
        saver = tf.train.Saver()
        train_generator = ImageDataGenerator(class_list = train_file, n_class = num_class, batch_size = batch_size, flip = True,shuffle=True)
        val_generator = ImageDataGenerator(class_list = val_file, n_class = num_class, batch_size=1, shuffle = False,flip = False)
        
        train_batchs_per_epochs = np.floor(train_generator.data_size/batch_size).astype(np.int16)
        val_batchs_per_epochs = np.floor(val_generator.data_size/batch_size).astype(np.int16)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer.add_graph(sess.graph)
            model.load_weights(sess)

            print('{} start training ...'.format(datetime.now()))
            print('{} open TensorBoard at --logdir {}'.format(datetime.now(),filewreiter_path))

            for epoch in range(num_epochs):
                print('{} Epochs number:{}'.format(datetime.now(),epoch+1))
                step = 1
                while step < train_batchs_per_epochs:
                    batch_xs, batch_ys = train_generator.getNext_batch()
                    sess.run(train_op,feed_dict={x:batch_xs,y:batch_ys,keep_prob:dropout_rate})
                    if step%display_step == 0:
                        s,get_loss = sess.run([merged_summary,loss],feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
                        writer.add_summary(s,epoch*train_batchs_per_epochs+step)
                        print('{} steps number:{} loss is {}'.format(datetime.now(),epoch*train_batchs_per_epochs+step,get_loss))
                    step += 1
                print('{} start validation'.format(datetime.now()))
                test_acc = 0.
                test_count = 0
                for _ in range(val_batchs_per_epochs):
                    batch_xs, batch_ys = val_generator.getNext_batch()
                    acc = sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.})
                    test_acc += acc
                    test_count +=1
                test_acc /= test_count
                print('{} validation Accuracy = {:.4f}'.format(datetime.now(),test_acc))

                val_generator.reset_pointer()
                train_generator.reset_pointer()

                print('{} saving checkpoint of model ...'.format(datetime.now()))

                checkpoint_name = os.path.join(checkpoint_path,'model_epoch'+str(epoch+1)+'.ckpt')
                save_path = saver.save(sess,checkpoint_name)
                print('{} Model checkpoint saved at {}'.format(datetime.now(),checkpoint_name))

#trainModels()

            
            

