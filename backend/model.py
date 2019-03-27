from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.applications import ResNet50
# import numpy as np
# import panda as pd 
from keras.utils.data_utils import get_file
# import matplotlib.pyplot as plt
# from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


class Model:
    def __init__(self):

        self.resnet = ResNet50(include_top=False, pooling='avg')
        self.model = Sequential()
        self.model.add(self.resnet)
        self.model.add(Dense(1))
        self.model.layers[0].trainable = False
        self.model.compile(loss='mean_squared_error', optimizer=Adam())
        self.weights_path = get_file(
            'fit4.h5',
            'https://s3-us-west-2.amazonaws.com/hotornot-bucket/fit4.h5')
        print("fetched weight")
        self.model.load_weights(self.weights_path)
        print("model init")

        
    def predict(self,img):

       
        score =  self.model.predict(img)
         
        return score