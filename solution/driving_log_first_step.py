import os
import csv

samples = []

TRAININGPATH = './../../../training_data_1/'
with open(TRAININGPATH+'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

TRAININGPATH = './../../../training_data_2/'
with open(TRAININGPATH+'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

TRAININGPATH = './../../../training_data_3/'
with open(TRAININGPATH+'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

TRAININGPATH = './../../../training_udacity/'
with open(TRAININGPATH+'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

TRAININGPATH = './../../../training_data_4/'
with open(TRAININGPATH+'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

TRAININGPATH = './../../../training_data_5/'
with open(TRAININGPATH+'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

TRAININGPATH = './../../../training_data_6/'
with open(TRAININGPATH+'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

TRAININGPATH = './../../../training_data_7/'
with open(TRAININGPATH+'driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader:
	   samples.append(line)

TRAININGPATH = './../../../training_data_8/'
with open(TRAININGPATH+'driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader:
	   samples.append(line)

TRAININGPATH = './../../../training_data_9/'
with open(TRAININGPATH+'driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader:
	   samples.append(line)

TRAININGPATH = './../../../training_data_10/'
with open(TRAININGPATH+'driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader:
	   samples.append(line)

TRAININGPATH = './../../../training_data_11/'
with open(TRAININGPATH+'driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader:
	   samples.append(line)

TRAININGPATH = './../../../training_data_12/'
with open(TRAININGPATH+'driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader:
       samples.append(line)

TRAININGPATH = './../../../training_data_13/'
with open(TRAININGPATH+'driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader:
       samples.append(line)

TRAININGPATH = './../../../training_data_14/'
with open(TRAININGPATH+'driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader:
       samples.append(line)

TRAININGPATH = './../../../training_data_15/'
with open(TRAININGPATH+'driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader:
       samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib
from skimage import exposure

b_size = 32
ch, row, col = 3, 160, 320  # Trimmed image format


def augment_brightness_camera_images(image):
	image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
	image1 = np.array(image1, dtype = np.float64)
	random_bright = .5+np.random.uniform()
	image1[:,:,2] = image1[:,:,2]*random_bright
	image1[:,:,2][image1[:,:,2]>255]  = 255
	image1 = np.array(image1, dtype = np.uint8)
	image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
	return image1

def add_random_shadow(image):
	top_y = 320*np.random.uniform()
	top_x = 0
	bot_x = 160
	bot_y = 320*np.random.uniform()
	image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
	shadow_mask = 0*image_hls[:,:,1]
	X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
	Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

	shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
	#random_bright = .25+.7*np.random.uniform()
	if np.random.randint(2)==1:
		random_bright = .5
		cond1 = shadow_mask==1
		cond0 = shadow_mask==0
		if np.random.randint(2)==1:
			image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
		else:
			image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
	image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

	return image

def trans_image(image,steer,trans_range):
	# Translation
	tr_x = trans_range*np.random.uniform()-trans_range/2
	steer_ang = steer + tr_x/trans_range*2*.2
	tr_y = 40*np.random.uniform()-40/2
	#tr_y = 0
	Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
	image_tr = cv2.warpAffine(image,Trans_M,(col,row))
	
	return image_tr,steer_ang


def generator(samples, batch_size=b_size):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				name = batch_sample[0].split('/')[-1].strip()

				#-----------------------------------------------------
				# center image
				#-----------------------------------------------------

				center_image = cv2.imread(name)
				center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
				center_angle = float(batch_sample[3])
 
				images.append(center_image)
				angles.append(center_angle)
				img_trans,angle_trans = trans_image(center_image,center_angle,np.random.randint(20)+1)
				images.append(img_trans)
				angles.append(angle_trans)

				if np.random.randint(shadowRate)==1:

					images.append(add_random_shadow(center_image))
					angles.append(center_angle)
					#images.append(np.fliplr(center_image))
					#angles.append(-center_angle)
					#images.append(augment_brightness_camera_images(center_image))
					#angles.append(center_angle)
				else:
					continue
				

				#-----------------------------------------------------
				# left image
				#-----------------------------------------------------

				name = batch_sample[1].split('/')[-1].strip()
				center_image1 = cv2.imread(name)
				center_image1 = cv2.cvtColor(center_image1, cv2.COLOR_BGR2RGB)
				center_angle = float(batch_sample[3])-float(batch_sample[3])*0.1

				images.append(center_image1)
				angles.append(center_angle)
				img_trans,angle_trans = trans_image(center_image1,center_angle,np.random.randint(20)+1)
				images.append(img_trans)
				angles.append(angle_trans)

				if np.random.randint(shadowRate)==1:

					images.append(add_random_shadow(center_image1))
					angles.append(center_angle)
					#images.append(np.fliplr(center_image1))
					#angles.append(-center_angle)
					#images.append(augment_brightness_camera_images(center_image1))
					#angles.append(center_angle)
				else:
					continue


				#-----------------------------------------------------
				# right image
				#-----------------------------------------------------

				name = batch_sample[2].split('/')[-1].strip()
				center_image2 = cv2.imread(name)
				center_image2 = cv2.cvtColor(center_image2, cv2.COLOR_BGR2RGB)
				center_angle = float(batch_sample[3])+float(batch_sample[3])*0.1

				images.append(center_image2)
				angles.append(center_angle)
				img_trans,angle_trans = trans_image(center_image2,center_angle,np.random.randint(20)+1)
				images.append(img_trans)
				angles.append(angle_trans)


				if np.random.randint(shadowRate)==1:
	
					images.append(add_random_shadow(center_image2))
					angles.append(center_angle)
				   #images.append(np.fliplr(center_image2))
					#angles.append(-center_angle)
					#images.append(augment_brightness_camera_images(center_image2))
					#angles.append(center_angle)
				else:
					continue

 


			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=b_size)
validation_generator = generator(validation_samples, batch_size=b_size)

#print(samples)
#print(len(validation_samples))




# Initial Setup for Keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Input, AveragePooling2D, GlobalAveragePooling2D, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

# TODO: Build the Final Test Neural Network in Keras Here
#input_tensor = Input(shape=(row, col, ch))
#base_model = ResNet50(input_tensor=input_tensor, include_top=False)
#base_model = InceptionV3(input_tensor=input_tensor, include_top=False)
#base_model = VGG16(input_tensor=input_tensor, include_top=False)

#x = base_model.output
#x = AveragePooling2D((8,8), strides=(8,8))(x)
#x = MaxPooling2D((3,3))(x)
#x = Flatten()(x)
#x = Dense(256, activation='relu')(x)
#x = Dense(1)(x)
#model = Model(base_model.input, x)

# # freeze base model layers
#for layer in base_model.layers:
#    layer.trainable = False

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



model = Sequential()

model.add(Lambda(lambda x: x/255 - .5, input_shape=(row, col , ch)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
#model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
#model.add(BatchNormalization())
#model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
#model.add(BatchNormalization())
#model.add(Dropout(0.50))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(BatchNormalization())
#model.add(Dropout(0.15))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(BatchNormalization())
#model.add(Dropout(0.80))
model.add(Dense(50))
model.add(BatchNormalization())
#model.add(Dropout(0.95))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Dense(1))

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

# freeze base model layers
retrain = False
shadowRate = 25

if retrain == True:
	shadowRate = 20
	del model
	model = load_model('model_TOP.h5')

	for layer in model.layers:
		layer.trainable = False
	model.summary()

	pop_layer(model)
	pop_layer(model)
	pop_layer(model)
	pop_layer(model)
	pop_layer(model)

	x = Dropout(0.5)(model.layers[-1].output)
	model = Model(input=model.input, output=[x])

	x = Dense(100)(model.layers[-1].output)
	model = Model(input=model.input, output=[x])

	x = Dropout(0.5)(model.layers[-1].output)
	model = Model(input=model.input, output=[x])

	x = BatchNormalization()(model.layers[-1].output)
	model = Model(input=model.input, output=[x])

	x = Dense(10)(model.layers[-1].output)
	model = Model(input=model.input, output=[x])
	#model.compile(loss='mse', optimizer='adam')	
	x = BatchNormalization()(model.layers[-1].output)
	model = Model(input=model.input, output=[x])
	#model.compile(loss='mse', optimizer='adam')	
	x = Dense(1)(model.layers[-1].output)
	model = Model(input=model.input, output=[x])

	#model.add(Dense(1))
	#model.save('model_afterNewLayers.h5'))

	model.compile(loss='mse', optimizer='adam')
	model.summary()


# Preprocess incoming data, centered around zero with small standard deviation 
#model.add(... finish defining the rest of your model architecture here ...)


model.fit_generator(train_generator, samples_per_epoch=
			len(train_samples), validation_data=validation_generator, 
			nb_val_samples=len(validation_samples), nb_epoch=3)

#model.fit(train_samples,validation_samples,validation_split=0.2,shuffle=True, nb_epoch=7)
#for layer in model.layers:
#	layer.trainable = True

model.save('model.h5')

print('Finished....')