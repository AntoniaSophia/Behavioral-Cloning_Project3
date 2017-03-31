#**Behavioral Cloning**


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


[image5]: ./docu/model_summary.png "Model summary"
[image6]: ./docu/data_distribution.png "Data distribution"
[image7]: https://rawgit.com/AntoniaSophia/Behavioral-Cloning_Project3/master/solution/docu/Behavioral_Cloning.html "Notebook"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

* link to model.py [Model File](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/model.py)
* link to drive.py [Drive Control](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/drive.py)  
* link to network model.h5 [Network file](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/model.h5)
* link to the video of the first track video1.mp4 [Video Track 1](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/videos/videos_track1.mp4)
* link to the video of the first track video2.mp4 [Video Track 2](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/videos/videos_track2.mp4)
* link to the video of the first track video1.mp4 [Video Track 1 HD](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/videos/Videos%20Track1%20Fullhd-1.m4v)
* link to the video of the first track video2.mp4 [Video Track 2 HD](https://github.com/AntoniaSophia/Behavioral-Cloning_Project3/blob/master/solution/videos/Videos%20Track2%20Fullhd-1.m4v)


####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* P3_writeup_Antonia.md (this file) summarizing the results
* Video1.mp4 which shows the car driving autonomously on the first track
* Video2.mp4 which shows the car driving autonomously on the second track


####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```


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


overfitting
- dropout
- clockwise and anti-clockwise 
- limited epochs (maximum 5)
- 20% validation data

How was trained?

dropout does not really work everywhere - after flattening and Dense(100) it works best (most parameters in the model)
RGB space works best for me - also tried out other color spaces



2 level
- first try to find a model with limited data which is able to detect lanes and lane markings correctly
- train this model using random shadowing and translation


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

Training data from both tracks have been taken. In order capture good driving behavior I took:
- 3 tracks in the original direction
- 2 track in the other direction
- training of recovery (how to get back to center) --> unfortunately I recognized that I also recorded the phase where I drove towards the hard shoulder also which is a silly mistake of course...
- in the second track 3 curves got a special treatment as I recorded them a couple of times as they made trouble
- as additional data source I used the Udacity data 

At the end I had around 33000 images in total. 

The following HTML file shows exploration of the test data including center pictures, flipped pictured, shadowed pictures and the distribution of the angles among the whole test set.

![Exploration of test data][image7]

The following image shows the distribtion of the angles of the test data:
![data distribution][image6]


I used the following techniques in order to augment the test data (see the function image_pipeline in line 101):
- random show works well
- translation also works well
- left camera + right camera plus correction factor

What didn't work at all:
- chosing a different color space than RBG
- darkening the whole image (not only random shadowing)
- creation of too much augmented data also lead to worse results from a certain ratio (maximum 1:2 between real data:augmented data )


General remark:
From my point of view the training data is not really good - I'm not an expert in gaming so I had a tough time driving around escpecially the second track. I was often to harsh in the angle and sometimes I just drove like a drunken pingiun.


Finally I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. The reason is simply heuristic: more epochs didn't produce better results and made the difference between training loss and validation even bigger which is and indicator for overfitting. With 3 epochs both values have been pretty close.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

Actually I'm really surprised that the absolute accuracy value doesn't allow a prediction whether the model is good or bad!


####4. Let's give an overall feedback
Thank god I'm done with it.... ;-)

Definitely it was a lot of fun working on that stuff and I'm really proud to succeed in the second track also!!

Again I really learned a lot of things, but on the other hand I feel that even more questions arise after this project:
- how can a neural network be validated? My only criteria was that I tried to observe if the network has understood the rules of center driving
- why are some attempts crazy and some others pretty good
- there seems to be so much of random inside the approach
- accuracy has nearly no meaning, rather the opposite: networks with the smallest loss had the worst results according to my observation


