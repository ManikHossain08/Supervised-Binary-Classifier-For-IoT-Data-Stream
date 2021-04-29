Presentation

## Supervised Binary Classifier For IoT Data Stream 

Our project is about processing a Data Stream and fitting a Binary Classifier to predict the class label. 
We apply techniques that are different from the methods used to analyse regular stationary tabular data. 

We have met several challenges:
1. How to deal with imbalanced data set when we don’t know in advance how many samples we have.
2. How to build the model incrementally as the data comes in.
3. How to assess the performance of the models in such non-stationary environment.


### Outline of the presentation

1. Introduction – Goal of the project
2. Dataset Description 
3. Data Analysis
4. Results and Discussion
5. Conclusions

###  1. Introduction – Goal of the project

Context – IoT sensors collect various telemetry data used in smart autonomous buildings.
Goal – Process the stream incrementaly and build a binary classifier able to accurately predict the target label.  The target label is “light”. Our goal is to fit a model  that can predict if the light is on or off. Several machine learning models exist and we choose two of them to verify how they will perform this real world data stream: Haive Bayes and Hoeffding Tree.
Applications – This prediction could be interesting in case of emergency if the sensor breaks, and we don’t receive data anymore,  for anomaly detection or to remotely control the conditions in smart building such as hospitals, office buildings or urban plants facilities.


### 2. Dataset Description
 
The data is produced by sensors arrays. 
Each sensor array emits different measurements - temperature, humidity, carbon monoxide level, motion detection, smoke detection. 

![image](https://user-images.githubusercontent.com/39202594/113806869-2121ff80-9731-11eb-87c1-c897e7a87bc6.png)

And all sensors are feeding into a central controller on one input port. 

 - Volume - 405,184 data points
 - Velocity - Data spans a period of 8 days. The average rate is 1 entry every 1.33 seconds
 - Variety - Data gathered from three arrays of IoT sensors located in  different conditions
 - Veracity – Data is accurate, no missing data, only few duplicates
 - Value – Data is useful for monitoring and control of indoor environment
 - Target class – “light” !
 

## Data Analysis

Our dataset is infinite and non-stationary. 

-	Infinite : data is produced constantly by the sensors
-	Non stationary: the distribution can change over time
-	Number of classes: 2
-	Majority class: Light OFF

![image](https://user-images.githubusercontent.com/39202594/113768643-1134fc00-96ee-11eb-94ff-3540804938f1.png)

We have combined the results from all devices in order to obtain generalized model.

The correlation matrix using Pearson Method

![image](https://user-images.githubusercontent.com/39202594/114723490-b2ccd680-9d08-11eb-9406-f3dc2e1b7394.png)

The heatmap of the correlation matrix

![image](https://user-images.githubusercontent.com/39202594/114727840-8f0b8f80-9d0c-11eb-83a5-1925d7a04adc.png)



We have investigated the distribution of the class label overtime. We see that the class label is not evenly distrbuted.


[ Changes in the class distribution overtime - concept drift]
![image](https://user-images.githubusercontent.com/39202594/113768759-31fd5180-96ee-11eb-9011-aa0764e0bb99.png)

The distibution of the majority vs the minority class:
![image](https://user-images.githubusercontent.com/39202594/113768970-74269300-96ee-11eb-9bfd-43a692c68574.png)

## Solution for imbalanced data set

#### 1. Class weights

We used the method **compute_sample_weight** method to adjust the class weights inversely proportional to class frequencies in the input data. With the class weights we give more emphasis of the minority class.

#### 2. Reservoir Sampling

With Reservoir Sampling we fill a preallocated buffer, called a reservoir, with uniformly sampled elements from the datastream. However, we use only data points from the majority class to fill the reservoir. We use every data point from the minority class to train the model. Once we reach a given number of training samples from the minority class, we use the reservoir to train the model with the equal number of data points from the majority class. Our goal is udersample the majority class.

[Illustration of Under-Sampling and Over-Sampling]
![image](https://user-images.githubusercontent.com/45977153/114323928-7ffec480-9af5-11eb-92e6-f39bf1807575.png)

### 3. Apply SMOTE (Over-Sampling)
In order to create the balancced datasets, we applied SMOTE for over-sampling on trainning datasests where it creates synthetic samples by randomly sampling the characteristics from occurrences in the minority class and transformed imbalanced datasets to balanced datasets.


## Machine Learning Models for Classification

1. Hoeffding Tree
2. Naive Bayes

### 1. Hoeffding Tree

Hoeffding Tree is similar to a conventional Decision Tree. The main difference is that it does not need the entire data set to figure out which split is the best. As in the conventional method the Hoeffding Tree also uses Gini index or information gain and recursively splits the features. 
We have noticed that with conventional Decision Tree method, the training is slow. We need the entire data set in order to decide how to split at each node. We calculate the imputity measure by using all data points. With Hoeffding tree we dont need the entire data set, we use the Hoeffding Bound to estimate when the best split is found. This measure allows the leaf to decide what are the first two best splits (in order), we dont need the rest of the splits.

We performed hyperparameter tuning on the following parameters: the grace period, and the leaf learner, and the confidence level. The **grace period** represents how many data points will be processed before a leaf is evaluated for a split. The **leaf learner** represents which method will be used in the leaf to predict the label. We can predict the label simply by majority voting, or we can choose Naive Bayes classification.  We selected a grace period of 10 data points and confidence level of 0.01 and Naive Bayes for the leaf learner. 

### 2. Naive Bayes

Naïve Bayes algorithm is a another supervised machine learning algorithm, which is based on Bayes theorem and used for solving binary classification problems.
Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions. 

There are 4 types of Naive Bayes Model, which are given below: such as 

- Gaussian Naive Bayes: it follow a normal distribution.
- Multinomial Naive Bayes: it is used when the data is multinomial distributed. 
- Bernoulli Naive Bayes: The Bernoulli classifier works similar to the Multinomial classifier, but the predictor variables are the independent Booleans variables.
- Complement Naive Bayes: It is wellknown model for imbalanced datasets. Where Multinomial and Gaussian Naive Bayes may give a low accuracy, Complement Naive Bayes will perform quite well and will give relatively higher accuracy. But with our datasets complemts Naive bayes performance is not good instead Gaussian Naive bayes overperformed accounted for 98% accuracy with both imbalanced and balanced datasets.

Here in this project, We performed all 4 models using our datasets and compared the performance among them. 



## Performance Analysis

#### 1. Using holdout method.

With the holdout method we need some data to set asside for testing purposes. We use the method **train_test_split** from the scikit package to split our data set before we start the training. We make prediction on the selected test set, at regular intervals, for example every 100 data points. We can improve this method by selection a holdout test set from the stream of size **k** every **n** instances. Our results show on average high Accuracy and F-score, but results are not reliable, due to the uneven distribution of the class label:  

[Accuracy]

![image](https://user-images.githubusercontent.com/39202594/113769091-94565200-96ee-11eb-92cd-f45afbf25a55.png)

[F1-SCore]

![image](https://user-images.githubusercontent.com/39202594/113769169-ab953f80-96ee-11eb-8d50-3eca0dbcbca1.png)

[Gaussian Naive Bayes model score]

Among 4 Naive Bayes classifier, Gaussian Naive Bayes gives us highest accuracy with balanced and imbalanced data which is 98% along with high precision and recall. for light off, we achieve 97% precision and 100% recall as well as 99% f1-score. On the other hand, for light on, we got 100% precission, 93% recall as well as 96% f1-score. Therefore, this result indicate good performance and good accuracy for this Gaussian Naive Bayes model.  

![image](https://user-images.githubusercontent.com/45977153/114285961-0dc0ae00-9a29-11eb-8bfe-66b5dc6121ec.png)

[Gaussian Naivee Bayes: Confusion Matrix]

we also generate the confussion matrix with real number and in normalized format for the Gaussian Naive Bayes. this picture indicate the normalized form of confusion matrix where True positive is 26% and True Negative is 72% therefore we got 98% accuracy. 

![image](https://user-images.githubusercontent.com/45977153/114314230-de14b300-9ac7-11eb-9e5c-2d5cfc371e52.png)


#### 2. Using Predictive sequential (prequential metric)

We use this method when there is no data available for testing. In streaming environment the data comes in and we cannot split the data to a test set and a training set.
So, we used the method **"first test then train"**, i.e for each example in the stream, we make a prediction using our current model, and then we use the same data point to update the model. We count the number of correctly predicted label vs the total number of training examples which gives us the following Accuracy and F-score:

[Accuracy and F1-score]

![image](https://user-images.githubusercontent.com/39202594/114331376-daa61980-9b11-11eb-8f6d-176024d7297c.png)

[Confusion Matrix (every 1000)]

![image](https://user-images.githubusercontent.com/39202594/114331133-5a7fb400-9b11-11eb-9e67-35a749222e32.png)
![image](https://user-images.githubusercontent.com/39202594/114331176-77b48280-9b11-11eb-9406-5e00992551b0.png)
![image](https://user-images.githubusercontent.com/39202594/114331210-8ac75280-9b11-11eb-8bc5-a58157818eb0.png)

[Final model]

![image](https://user-images.githubusercontent.com/39202594/114331277-ac283e80-9b11-11eb-9938-028de418fe2b.png)

[Accuracy and F1-score when we remove the feature Temperature]

![image](https://user-images.githubusercontent.com/39202594/114730632-0c380400-9d0f-11eb-8eab-a1d3411110cf.png)


[Gaussian Naive Bayes: Four model classifier comparison]

We also did some performance comaparison among the 4 different types of Naive Bayes Classifier. We found that from 4 types of Naive Bayes Classifier model Gaussian Naive Bayes performed exceptionally well in both balanced (i.e., after applying SMOTE over-sampling) and imbalanced (i.e., real) datasets. Hence, we got the accuracy of 98% where Multinomial and Bernoulli Naive Bayes around 75% as well as Complement Naive Bayes gives us 48% accuracy. 

![image](https://user-images.githubusercontent.com/45977153/114286057-9d665c80-9a29-11eb-8970-ef8ac54c105e.png)




## Conclusions

1. Classification algorithms in data streams would perform well when the number of samples of each class is the same. However in reality this is rarely the case, which is a challenge. Our models are biased towards the majority class.

2. In case of data stream we need techniques that are different from the methods used to analyze regular stationary tabular data. Stream data can continuously evolve over time. Reservoir Sampling was used to sample data from the stream for training.

3. With Hoeffding Tree we do not need the entire data set to build the decision tree. We can build it incrementally as data points arrive. The Hoeffding Tree is has short training time. It performed better than the Naive Bayes model in regards to the F1-score. 

4. Evaluation of  the obtained models was a challenge. We applied the following methods: Holdout of an independent test set and Prequential Error.
5. From all the four Naive Bayes classifiers Gaussian Naïve Bayes gives us 98%  accuracy with this datasets.





## Glossary

**concept drift** - changes in the underlying data distribution overtime


