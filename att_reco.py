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
import pdb
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

K.set_learning_phase(1) #set learning phase

network ='vgg'

if network == 'vgg':
	from keras_frcnn import vgg as nn
elif network == 'resnet50':
	from keras_frcnn import resnet as nn
else:
	print('Not a valid model')
	raise ValueError


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
shared_layers = nn.nn_base(img_input, trainable=True)


out_roi_pool = mylayer(pooling_regions, 6)([shared_layers, roi_input])
#print out_roi_pool

out = (Flatten(name='flatten'))(out_roi_pool)
out = (Dense(4096, activation='relu', name='fc1'))(out)
out = (Dropout(0.5))(out)
out = (Dense(2048, activation='relu', name='fc2'))(out)
out = (Dropout(0.5))(out)
print 'no of classes', nb_classes
out_class = (Dense(73, activation='sigmoid', kernel_initializer='zero'))(out)

model_classifier = Model([img_input, roi_input], out_class)


try:
	print('loading weights from {}'.format(C.base_net_weights))
	#model_classifier.load_weights(C.base_net_weights, by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_classifier.compile(optimizer=optimizer_classifier, loss = 'binary_crossentropy')

# data loading
df3 = pd.read_pickle('temp.pkl')
import pdb
pdb.set_trace()
def name(x):
    name = str.split(x,'/')
    return '/fashion-attribute/storage_bucket/deep_fashion/img_bbox_data/img_n/' + name[1] + '/' + name[2]
    
df3['img_name'] = df3['img_name'].apply(lambda x:name(x))
pdb.set_trace()
img = []
landmarks = []
attribs = []
<<<<<<< HEAD
for i in range(30):
    print i
=======
for i in range(100):
    #print i
>>>>>>> 454166c4b00ff9b0aa2d7b09f86c6a7752a5f600
    a = cv2.imread(df3.loc[i,'img_name'])
    a = cv2.resize(a,(400,400))
    img.append(a)
    lnd = df3.loc[i,'landmarks']
    landmarks.append(lnd)
    atts = df3.loc[i,'atts']
    attribs.append(atts)

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

X = np.array(img)
X2 = np.array(landmarks)
Y = mlb.fit_transform(attribs)
X2 = (X2/16).astype(int)
print 'shapes of input and output', X.shape, X2.shape, Y.shape
pdb.set_trace()
checkpoint = ModelCheckpoint("upper_body_atts_model.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model_classifier.fit(x = [X,X2],y= Y, batch_size=1, epochs=2, validation_split=0.2,callbacks = [checkpoint])


