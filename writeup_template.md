#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[hist-tr]: ./images/train.png "training set histogram"
[hist-val]: ./images/valid.png "validataion set histogram"
[hist-te]: ./images/test.png "test set histogram"
[rep-images]: ./images/images.png "rep"
[graph]: ./images/graph.png "graph"
[tensorboard]: ./images/tensorboard.png "tensorboard"
[test-1-speed-limit-30]: ./dataset/test/1_speed_limit_30.jpg "1"
[test-2-speed-limit-50]: ./dataset/test/2_speed_limit_50.jpg "2"
[test-14-stop]: ./dataset/test/14_stop.jpg "3"
[test-15-no-vehicles]: ./dataset/test/15_no-vehicles.jpg "4"
[test-17-no-entry]: ./dataset/test/17_no_entry.jpg "5"



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/penny4860/CarND-P2-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is (34799, 32, 32, 3)
* The size of the validation set is (4410, 32, 32, 3)
* The size of test set is (12630, 32, 32, 3)
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

#####2.1 Training dataset histogram

The following histogram shows how many samples in the training set are in each class.

![alt text][hist-tr]

#####2.2 Validataion dataset histogram

The following histogram shows how many samples in the validation set are in each class.

![alt text][hist-val]

#####2.3 Test dataset histogram

The following histogram shows how many samples in the test set are in each class.

![alt text][hist-te]

#####2.4 Representative images of each class

The following figure shows a sample for each class. One sample was selected for each class and histogram equlization for each channel was performed and plotted.

![alt text][rep-images]



###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

To match the range of all samples to [-1, 1], centering and normalization were performed with the following code.
```images = images.astype(float) - 128)/128```

Because I used batch normalization at the stage of constructing the model, I did not use the complicated method in the preprocessing step.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Inspired by VGGnet, I designed the following model:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x64    |
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x64    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 					|
| Fully connected		| outputs 256        							|
| Fully connected		| outputs 43        							|
| Softmax				|         										|


Batch normalization was used for fully connected layers and convolutional layers except for final fully connected layer.

Here is the operation graph generated by tensorflow

![alt text][graph]
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

rmsprop optimizer를 learning rate 0.001로 사용하였다. batch size는 학습속도를 빠르게하기 위해 32로 설정하였다. number of epochs는 20으로 충분히 크게 설정했고, 각 epoch마다 validation 
accuracy를 logging하면서 model을 저장했다. 최종적으로는 validation error가 가장 작은 model을 선택하였다. 


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.9998
* validation set accuracy of 0.9848
* test set accuracy of 0.9758

관련된 code는 https://github.com/penny4860/CarND-P2-Classifier/blob/master/Traffic_Sign_Classifier.ipynb의 [17], [23]에서 확인할 수 있다.
아래의 그림은 tensorboard로 accuracy를 monitoring한 결과이다. 최종적으로는 10870번째 step(10번째 epoch)의 모델을 선택하였다.

![alt text][tensorboard]

* What architecture was chosen?
	* VGG16 architecture를 사용하였고, layer마다 batch normalization을 사용하여 성능을 높였다.

* Why did you believe it would be relevant to the traffic sign application?
	* VGG16은 architecture가 간단하고, 다른 task로 확장이 용이하다고 알려져 있기 때문에 이 architecture를 사용하였다.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
	* Training set에서의 accuracy가 100%에 가깝게 학습되었다(99.98%). 이것은 traffic sign classification task에 모델이 매우 적합하다는 뜻이다.
	* Validation accuracy는 training accuracy와 1.5% 떨어지는 인식률(98.48)을 보였다. Training accuracy와 validation accuracy의 작은 차이는 model이 overfitting되지 않았다는 증거이다.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


![alt text][test-2-speed-limit-50]  
![alt text][test-1-speed-limit-30] ![alt text][test-14-stop] ![alt text][test-15-no-vehicles] ![alt text][test-17-no-entry]

stop sign image의 경우는 object의 위치가 image의 위쪽에 있고, 이미지의 가로, 세로 비율로 정사각형 형태가 아니기 때문에 분류가 어려울 것으로 보인다.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 50      	| Speed Limit 20   									| 
| Speed Limit 30     	| Speed Limit 30 										|
| Stop					| Priority Road											|
| No vehicles	      	| No vehicles					 				|
| No entry				| No entry      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. Test accuracy가 98.48% 인 것을 감안하면 매우 낮은 수치이다.
웹에서 수집한 image들이 정사각형 형태가 아니고 직사각형 형태이기 때문에 이러한 결과가 나온 것 같다. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 35th cell of the Ipython notebook.

For the first image(2-speed-limit-50), the model is not sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .43         			| 0 (Speed limit (20km/h)   				    | 
| .40     				| 2 (Speed limit (50km/h) : truth				|
| .06					| 19 (Dangerous curve to the left)										|
| .01	      			| 17 (No entry)					 				|
| .01				    | 11 (Right-of-way at the next intersection)     							|

For the first image(1-speed-limit-30), the model is not sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 1 (Speed limit (30km/h) : truth  									| 
| .01     				| 5 (Speed limit (80km/h)										|
| .00					| 18 (General caution)						|
| .00	      			| 16 (Vehicles over 3.5 metric tons prohibited)			 				|
| .00			    	| 0 (Speed limit (20km/h)  							|


For the first image(14-stop), the model is not sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 12 (Priority road)  									| 
| .00     				| 23 (Slippery road)										|
| .00					| 1	 (Speed limit (30km/h))										|
| .00	      			| 13 (Yield)					 				|
| .00			    	| 41 (End of no passing)    							|


For the first image(15-no-vehicles), the model is not sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were



| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| 15 (No vehicles) : truth  									| 
| .00     				| 2 (Speed limit (50km/h)										|
| .00					| 22 (Bumpy road)											|
| .00	      			| 14 (Stop)					 				|
| .00			    	| 17 (No entry)    							|

For the first image(17-no-entry), the model is not sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 17 (No entry) : truth  									| 
| .00     				| 9 (No passing)										|
| .00					| 14 (Stop)							|
| .00	      			| 29 (Bicycles crossing)			 				|
| .00			    	| 0 (Speed limit (20km/h))    							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


