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

from keras.models import Sequential
"""Import from keras_preprocessing not from keras.preprocessing, because Keras may or maynot contain the features discussed here depending upon when you read this article, until the keras_preprocessed library is updated in Keras use the github version."""
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

print 'loading dataframe'
df = pd.read_pickle("./upper_body_atts.pkl")
#df= df[0:10000]
df = df[['img_name','landmarks','atts']]

df["labels"] = df["atts"].apply(lambda x:list(x))

def name(x):
    name = str.split(x,'/')
    return '/fashion-attribute/att_recognition/img_bbox_data/img_n/'+ name[1]+'/'  + name[2]
    #return '/fashion-attribute/storage_bucket/deep_fashion/img_bbox_data/img_n/' + name[1] + '/' + name[2]
#/fashion-attribute/storage_bucket/small_data_set/work/train/Blazer/Double-Breasted_Blazer

# local path : /Users/1024329/Downloads/DeepFashion/img_n/
# server path : /fashion-attribute/storage_bucket/deep_fashion/img_bbox_data/img_n/
    
df['Filenames'] = df['img_name'].apply(lambda x:name(x))

#df['Filenames'] = df['img_name'].apply(lambda x: '/Users/1024329/Downloads/DeepFashion/' + x)

# removing the samples without any attribute present in it
sample = df['atts'].apply(lambda x: x.shape[0]>0)
df4 =df[sample]

print 'creating image generators'
#import pdb
#pdb.set_trace()
img_reso = 224
datagen=ImageDataGenerator(rescale=1./255.)
test_datagen=ImageDataGenerator(rescale=1./255.)
train_generator=datagen.flow_from_dataframe(dataframe=df4[0:100000],directory="", 
        x_col="Filenames",y_col="labels",batch_size=32,
        seed=42,shuffle=True,class_mode="categorical",
        classes=range(1000),target_size=(img_reso,img_reso))

print 'validation generator'
valid_generator=test_datagen.flow_from_dataframe(
        dataframe=df4[100000:120000],
directory="",
x_col="Filenames",
y_col="labels",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
classes=range(1000),
target_size=(img_reso,img_reso))
print 'test generator'
test_generator=test_datagen.flow_from_dataframe(dataframe=df4[120000:],
directory="",
x_col="Filenames",
batch_size=1,
seed=42,
shuffle=False,
class_mode=None,
target_size=(img_reso,img_reso))



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
	input_shape_img = (224,224, 3)

img_input = Input(shape=input_shape_img)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=False)


out = (Flatten(name='flatten'))(shared_layers)
out = (Dense(4096, activation='relu', name='fc5'))(out)
out = (Dropout(0.5))(out)
out = (Dense(2048, activation='relu', name='fc6'))(out)
out = (Dropout(0.5))(out)
#out = (Dense(512, activation='relu', name='fc7'))(out)
out_class = (Dense(1000, activation='sigmoid'))(out)

model_classifier = Model(img_input, out_class)


try:
	print('loading weights from pretrained model')
	#model_classifier.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

model_classifier.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])

model_classifier.load_weights('vgg16_atts.h5')
import pdb
pdb.set_trace()

#checkpoint = ModelCheckpoint("vgg16_atts.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)


#STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
#STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
#model_classifier.fit_generator(generator=train_generator,
#                    steps_per_epoch=STEP_SIZE_TRAIN,
#                    validation_data=valid_generator,
#                    validation_steps=STEP_SIZE_VALID,
#                    epochs=50, use_multiprocessing=True, callbacks = [checkpoint])

