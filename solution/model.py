import os
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from skimage import exposure
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Input, AveragePooling2D, GlobalAveragePooling2D, Lambda, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import keras
from keras.optimizers import Adam

# define the parameters on how many additional data to be generated 
augmentRate = 3			# for each data generate 5 shadowed and 5 translated data
augmentFrequency = 2	# do this for every data
epochs = 3				# use 5 epochs

retrain = False
#retrain = True


samples = []
b_size = 32		#batch size
ch, row, col = 3, 160, 320  # Trimmed image format


# loading all recorded data in a generic way (collect all csv files and load all data)
for root, dirs, files in os.walk("./../../../"):
	for file in files:
		if file.endswith(".csv"):
			#print(os.path.join(root, file))
			with open(os.path.join(root, file)) as csvfile:
				reader = csv.reader(csvfile)
				for line in reader:
					samples.append(line)
			

#if retrain == false than start from the beginning using a reduced dataset (take only every third dataset)
if (retrain == False):
	samples = samples[::3]				


####################################
# Adding a shadow in a random way
####################################
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
	random_bright = .5
	cond1 = shadow_mask==1
	cond0 = shadow_mask==0
	if np.random.randint(2)==1:
		image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
	else:
		image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
	image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
	#image = cv2.cvtColor(image_hls,cv2.COLOR_RGB2HSV)

	return image

####################################
# Translates an image 
####################################
def trans_image(image,steer,trans_range):
	# Translation
	tr_x = trans_range*np.random.uniform()-trans_range/2
	steer_ang = steer + tr_x/trans_range*2*.2
	tr_y = 40*np.random.uniform()-40/2
	#tr_y = 0
	Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
	image_tr = cv2.warpAffine(image,Trans_M,(col,row))
	
	return image_tr,steer_ang


##################################################
# Image pipeline which creates new artifical data
##################################################
def image_pipeline(name,ang,corr_factor):
	images = []
	angles = []

	image = cv2.imread(name)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)	#convert image to RGB color space
	#center_image = cv2.cvtColor(center_image,cv2.COLOR_RGB2HSV)
	angle = ang+ang*corr_factor

	# append the original center image
	images.append(image)
	angles.append(angle)

	#if np.random.randint(augmentFrequency+1)==1:
	# left/right flip of the image
	images.append(np.fliplr(image))
	angles.append(-angle)
	if (retrain == True):
		for i in range(1,augmentRate):
			# add random shadow
			if np.random.randint(augmentFrequency+1)==1:

				images.append(add_random_shadow(image))
				angles.append(angle)

				# translate the image
				img_trans,angle_trans = trans_image(image,angle,np.random.randint(250)+1)
				images.append(img_trans)
				angles.append(angle_trans)

	return images,angles


#############################################################
# The generator which does a "lazy loading" of required data
############################################################
def generator(samples, batch_size=b_size):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:

				#-----------------------------------------------------
				# center image, augment data by image_pipeline
				#-----------------------------------------------------
				name = batch_sample[0].split('/')[-1].strip()
				center_angle = float(batch_sample[3]) 
				center_images,center_angles = image_pipeline(name,center_angle,0)
				images.extend(center_images)				
				angles.extend(center_angles)				

				#-----------------------------------------------------
				# left image, augment data by image_pipeline
				#-----------------------------------------------------
				name = batch_sample[1].split('/')[-1].strip()
				left_angle = float(batch_sample[3])-float(batch_sample[3])*0.1
				left_images,left_angles = image_pipeline(name,left_angle,-0*1)
				images.extend(left_images)				
				angles.extend(left_angles)				

				#-----------------------------------------------------
				# right image, augment data by image_pipeline
				#-----------------------------------------------------
				name = batch_sample[2].split('/')[-1].strip()
				right_angle = float(batch_sample[3])+float(batch_sample[3])*0.1
				right_images,right_angles = image_pipeline(name,left_angle,+0*1)
				images.extend(right_images)				
				angles.extend(right_angles)				


			# now yield the generator
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)




# split the training data into training data and validation data (20% is validation data)
train_samples, validation_samples = train_test_split(samples, test_size=0.20)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=b_size)
validation_generator = generator(validation_samples, batch_size=b_size)


##########################################################
#Build the Final Test Neural Network in Keras Here
##########################################################
model = Sequential()

model.add(Lambda(lambda x: x/255 - .5, input_shape=(row, col , ch)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
#model.add(Dropout(0.1))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
#model.add(Dropout(0.1))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
#model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(50))
model.add(BatchNormalization())
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Dense(1))

# use the Adam optimizer with default parameters according to http://arxiv.org/abs/1412.6980v8
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer=adam)


# helper function in order to remove a keras layer and not lose connection to the previous
# found this on any support page - seems to be a bug in keras that normal pop() doesn't work...... 
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


# in retrain mode use an existing model and cut the 1-1 layers 
# - define the model
# - load weights from model to be retrained 
# - set all layers as "not trainable"
# - cut the last 1-1 layers
# - add new 1-1 layers 
# - add Adam optimizer
# - recompile the model
if retrain == True:
	augmentRate = 1
	epochs = 4
	del model
	model = Sequential()
	model.add(Lambda(lambda x: x/255 - .5, input_shape=(row, col , ch)))
	model.add(Cropping2D(cropping=((70,25),(0,0))))
	model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Flatten())
	#model.add(Dropout(0.05))
	model.add(Dense(100))
	model.add(Dropout(0.05))
	model.add(BatchNormalization())
	model.add(Dense(50))
	model.add(BatchNormalization())
	model.add(Dense(10))
	model.add(BatchNormalization())
	model.add(Dense(1))

	model.load_weights('model_TOP.h5')

	for layer in model.layers:
		layer.trainable = False

	pop_layer(model)
	pop_layer(model)
	pop_layer(model)
	pop_layer(model)
	pop_layer(model)
	pop_layer(model)

	#model.summary()

	x = Dense(100)(model.layers[-1].output)
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

    # use the Adam optimizer with default parameters according to http://arxiv.org/abs/1412.6980v8
	adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='mse', optimizer=adam)
	model.summary()


# execute the model training....
model.fit_generator(train_generator, samples_per_epoch=
			len(train_samples), validation_data=validation_generator, 
			nb_val_samples=len(validation_samples), nb_epoch=epochs)


#model.summary()

# in case we are not in retrain mode cut the Dropout layers
# - save the model
# - define the model without dropouts
# - load the before stored weights into this model without dropouts
# - save the weights again
#
# this looks quite insane, but for me it was a generic and easy way to get rid of all dropout layers for retraining
if retrain == False:

	model.save('model_temp.h5')

	del model
	model = Sequential()

	model.add(Lambda(lambda x: x/255 - .5, input_shape=(row, col , ch)))
	model.add(Cropping2D(cropping=((70,25),(0,0))))

	model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(BatchNormalization())
	model.add(Dense(50))
	model.add(BatchNormalization())
	model.add(Dense(10))
	model.add(BatchNormalization())
	model.add(Dense(1))

	model.load_weights('model_temp.h5')

	# use the Adam optimizer with default parameters according to http://arxiv.org/abs/1412.6980v8
	adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='mse', optimizer=adam)

	#model.summary()

	model.save('model.h5')
	model.save('model_temp.h5')
else:
	model.save('model.h5')


print('Finished....')