# **Behavioral Cloning**

## Writeup by Matthew Jones

### Project: CarND-Behavioral-Cloning_P4

---

**Behavioral Cloning Project**

The major steps followed to complete this project included:
* Using the simulator to collect data of good driving behavior
* Building a convolution neural network in Keras that predicts steering angles from images
* Training and validating the model with a training and validation set
* Testing that the model successfully drives for at least one lap around the track without leaving the road
* Summarising the results in a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README
Here is a link to my [project code](https://github.com/matttpj/CarND-Behavioral-Cloning)  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files to run the simulator in autonomous mode and save the output

Key files are:
 * Creates and saves the model: _model.py_   
 * Model output: _models/model_nVidia.h5_
 * Runs the model in the simulator and drives the car in Autonomous mode: _drive.py_     
 * Video output of the car driving round the track: _runX.mp4_      
 * Writeup that summarises results: _writeup_CarND-Behavioral_Cloning_P4.md_


#### 2. Submission includes functional code
Using the Udacity provided simulator and my _drive.py_ file, the car can be driven autonomously around the track by executing
```python drive.py models/model_nVidia.h5 run2
```

#### 3. Submission code is usable and readable

The _model.py_ file contains the code for training and saving the convolution neural network; first using a LeNet configuration and second using a nVidia configuration, as recommended by the Udacity program. The nVidia model proved itself to perform well very quickly; eg. enabling the car to complete a lap of the track without leaving the track. The file shows the pipeline I used for training and validating the models and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My nVidia derived model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

The Udacity provided initial set of Training data _IMG/*.jpg_ and _driving_log.csv_ was used very successfully.  No additional training data seemed to be required.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My overall strategy for deriving a model architecture was to start with a LeNet architecture and then try alternatives, including the referenced nVidia approach.

I first setup my _model.py_ file as described by David Silver in the program videos; but I even struggled to load the Training data images and skip the header row in the _driving_log.csv_.  And then I found DarienMT solution which I used as guide to fix this and other challenging issues; eg. loading center, left and right camera images and data from the driving log.   
https://github.com/darienmt/CarND-Behavioral-Cloning-P3

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

## Pre-processing
| Step      		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Sequential       		|    							|
| Lambda   	| Normalise images to mean = 0; set images input shape to height/width/channels (160,320,3)	|
| Cropping2D  				|	Crop individual images by top/bottom (50,20) and left/right (0,0)											|

## Model parameters derived from nVidia

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 90x320 BGR image   							|
| Layer 1: Convolution 2D    	| 24 filters, 5x5 kernel, 2x2 stride 	|
| RELU					|												|									|
| Layer 2: Convolution 2D     	| 36 filters, 5x5 kernel, 2x2 stride	|
| RELU					|												|
| Layer 3: Convolution 2D     	| 48 filters, 5x5 kernel, 2x2 stride 	|
| RELU					|												|
| Layer 4: Convolution 2D     	| 64 filters, 3x3 kernel	|
| RELU					|												|
| Layer 5: Convolution 2D     	| 64 filters, 3x3 kernel 	|
| RELU					|												|
|	Flatten					|												|
|	Dense		|	100	output									|
|	Dense					|	50 output dimensions											|
|	Dense					|	10 output	dimensions										|
|	Dense					|	1	 Output	dimensions								|
|						|												|

## Training
| Step      		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| model.compile()      		|  loss = 'mse'				| mean square loss to minimise errors |
|  | optimizer = 'adam'  | more efficient algorithim for calculating gradient descent |
| model.fit()   	|  validation_split = 0.2	| to split test data and use 20% of images at end of epoch for testing |
|  | shuffle = True | shuffle the images before saving the model |
|  | epochs = 5 | recommended by Paul Heraty |


![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

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
