#***Behavioral Cloning***


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./docu/Behavioral_Cloning.html "Notebook"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

* link to model.py [Model File](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/model.py)
* link to drive.py [Drive Control](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/drive.py)  
* link to network model.h5 [Network file](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/model.h5)
* link to the video of the first track video1.mp4 [Video Track 1](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/video1.mp4)
* link to the video of the first track video2.mp4 [Video Track 2](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/video2.mp4)
* link to the video of the first track video3.mp4 [Video Track 3](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/video3.mp4)

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* P3_writeup_Antonia.md (this file) summarizing the results
* Video1.mp4 which shows the car driving autonomously on the first track
* Video2.mp4 which shows the car driving autonomously on the second track
* Video3.mp4 which shows the car driving autonomously on the third track

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```


dropout does not really work everywhere - after flattening and Dense(100) it works best (most parameters in the model)
training data is not really good - I'm not an expert in gaming...
RGB space works best for me - also tried out other color spaces

random show works well
translation also works well



overfitting
- dropout
- clockwise and anti-clockwise 
- limited epochs (maximum 5)
- 20% validation data

How was trained?

Training data from both tracks have been taken.
- 3 tracks original direction
- 1 track other direction
- training of recovery (how to get back to center) - unfortunately I recognized that I also recorded the phase where I drove towards the hard shoulder also which is a silly mistake of course...
- in the second track 3 curves got a special treatment as I recorded them a couple of times as they made trouble


2 level
- first try to find a model with limited data which is able to detect lanes and lane markings correctly
- train this model using random shadowing and translation


####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior:
- I first recorded two laps on track one using center lane driving 
- 

The following HTML file shows exploration of the test data including center pictures, flipped pictured, shadowed pictures and the distribution of the angles among the whole test set

![alt text][image7]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
