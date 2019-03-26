
# coding: utf-8

# In[1]:


from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import cv2
import pandas as pd

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras_frcnn.mylayer import mylayer
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

K.set_learning_phase(1) #set learning phase

# sys.setrecursionlimit(40000)


# In[2]:


network ='vgg'

if network == 'vgg':
	from keras_frcnn import vgg as nn
elif network == 'resnet50':
	from keras_frcnn import resnet as nn
else:
	print('Not a valid model')
	raise ValueError


# In[3]:


from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
else:
	input_shape_img = (400, 400, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(6, 4))
nb_classes = 575

#pdb.set_trace()
num_rois = 6
# compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

if K.backend() == 'tensorflow':
    pooling_regions = 4
    input_shape = (num_rois,4,4,512)
elif K.backend() == 'theano':
    pooling_regions = 7
    input_shape = (num_rois,512,7,7)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=False)


out_roi_pool = mylayer(pooling_regions, 6)([shared_layers, roi_input])
print out_roi_pool

out = (Flatten(name='flatten'))(out_roi_pool)
out = (Dense(4096, activation='relu', name='fc5'))(out)
out = (Dropout(0.5))(out)
out = (Dense(2048, activation='relu', name='fc6'))(out)
#out = (Dropout(0.5))(out)
#out = (Dense(512, activation='relu', name='fc7'))(out)
out_class = (Dense(1000, activation='sigmoid'))(out)

model_classifier = Model([img_input, roi_input], out_class)


try:
	print('loading weights from pretrained model')
	model_classifier.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder')

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_classifier.compile(optimizer=optimizer_classifier, loss = 'binary_crossentropy', metrics= ['accuracy'])



# In[4]:


def name(x):
    name = str.split(x,'/')
    return '/fashion-attribute/att_recognition/img_bbox_data/img_n/' + name[1] + '/' + name[2]

# server path : /fashion-attribute/att_recognition/img_bbox_data/img_n/
# local path : /Users/1024329/Downloads/DeepFashion/img_n/
    
df3 = pd.read_pickle('upper_body_atts.pkl')
df3['img_name'] = df3['img_name'].apply(lambda x:name(x))


# In[5]:


df3 = df3[['img_name','landmarks','atts']]


# removing the samples without any attribute present in it
sample = df3['atts'].apply(lambda x: x.shape[0]>0)
df4 =df3[sample]
df4.shape, df3.shape
# checking the maximum value of the attribute label
df4['atts'].apply(lambda x: np.max(x)).max()

# reset index of df4
df4 = df4.reset_index(drop = True)


# In[17]:


def get_batch(num):
    img = []
    landmarks = []
    attribs = []
    b_size = 32
    for i in range(b_size*num, b_size*(num+1)):
        #print i
        a = cv2.imread(df4.loc[i,'img_name'])
        a = cv2.resize(a,(400,400))
        a = np.array(a).reshape(400,400,3)
        img.append(a/255)
        lnd = df4.loc[i,'landmarks']
        landmarks.append(lnd)
        atts = df4.loc[i,'atts']
        labels = np.array([0 if item not in atts else 1 for item in range(1000)]).reshape(1000,)
        attribs.append(labels)
    return np.array(img), (np.array(landmarks)/32).astype(int), np.array(attribs)


train_loss = []
# train_acc = []
for j in range(50):
    print 'epoch number:', j
    los = 0
    acc = 0
    for i in range(2000):
        if i % 1000 ==0: print 'epoch and sample:', j, i
        X,X2,Y = get_batch(i)
        history = model_classifier.fit(x=[X,X2], y= Y, verbose = 2)
        los = los + np.array(history.history['loss'])
        #acc = acc + np.array(history.history['acc'])
        #print los
    train_loss.append(los)
    model_classifier.save_weights('roi_atts.h5')
    #train_acc.append(acc)
    print train_loss
