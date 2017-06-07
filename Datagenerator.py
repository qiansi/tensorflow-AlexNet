import numpy as np 
import cv2

class ImageDataGenerator:
    def __init__(self, class_list, n_class, batch_size = 1, flip = True, shuffle = False, mean = np.array([104., 117., 124.]), scale_size = (227,227)):
        #initial params
        self.horizontal = flip
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_list = class_list
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0
        self.n_class = n_class

    def read_class_list(self,class_list):
        with open(class_list) as f:
            lines = f.readlines()
            self.images = []
            self.labels = []
            for line in lines:
                items = line.split()
                self.images.append(items[0])
                self.labels.append(items[1])
            self.data_size = len(self.labels)
    
    def shuffle_data(self):
        images = self.images.copy()
        labels = self.labels.copy()
        self.images = []
        self.labels = []
        idx = np.random.permutation(self.data_size)
        for id in idx:
            self.images.append(images[id])
            self.labels.append(labels[id])

    def reset_pointer(self):
        self.pointer = 0
        if self.shuffle:
            self.shuffle_data()
    
    def getNext_batch(self):
        paths = self.images[self.pointer:self.pointer+self.batch_size]
        labels = self.labels[self.pointer:self.pointer+self.batch_size]
        self.pointer += self.batch_size
        
        images = np.array([self.batch_size,self.scale_size[0],self.scale_size[1],3])
        for i in range(len(paths)):
            image = cv2.imread(paths[i])
            if self.horizontal and np.random.random()<0.5:
                image = cv2.flip(image,1)
            image = cv2.resize(image,(self.scale_size[0],self.scale_size[1]))
            image = image.astype(np.float32)

            image -= self.mean
            images[i] = image

        one_hot_labels = np.zeros((self.batch_size,self.n_class))
        for i in range(labels):
            one_hot_labels[i][labels[i]] = 1

        return images,one_hot_labels