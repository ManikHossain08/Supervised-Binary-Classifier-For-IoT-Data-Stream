#!/usr/bin/env python
# coding: utf-8

# tested in bigdata-lab environment
# installation skmultiflow: pip install -U scikit-multiflow

'''Result: After a delay of about 5 sec the tree will be mature enough and we will start printing the confusion matrix:
......
363000 | 353289 | 96076  | 5295   | 4416   | 257213 | 0.9519
Total  | Correct| TP     | FP     | FN     | TN     | F-score
364000 | 354289 | 96327  | 5295   | 4416   | 257962 | 0.952 
Total  | Correct| TP     | FP     | FN     | TN     | F-score
365000 | 355284 | 96574  | 5295   | 4421   | 258710 | 0.9521
Trained with majority class - 79300
Trained with minority class - 79300
Final model:*****************************
{'Tree size (nodes)': 127, 'Tree size (leaves)': 64, 'Active learning nodes': 64, 'Tree depth': 12, 
'Active leaf byte size estimate': 0.0, 'Inactive leaf byte size estimate': 0.0, 'Byte size estimate overhead': 1.0}
'''

from skmultiflow.trees import HoeffdingTree, HoeffdingTreeClassifier
import random
import matplotlib
import matplotlib.pyplot as plt
from numpy import array
import numpy as np
import pandas as pd


def data_preparation(filename):
    '''Prepare the data for incremental training'''
    df=pd.read_csv(filename)
    df["light"]=df["light"].astype(int)
    df["motion"]=df["motion"].astype(int)
    select_cols = ['co', 'humidity', 'lpg', 'motion','smoke', 'temp', 'light']
    data = df[select_cols].to_numpy()
    
    return data


def initialize_model(grace_period, split_criterion, split_confidence, leaf_prediction, binary_split):
    '''Initialize the model'''
    ht = HoeffdingTreeClassifier(
        grace_period=grace_period, 
        split_criterion=split_criterion, 
        split_confidence = split_confidence,
        leaf_prediction = leaf_prediction,
        binary_split = binary_split)
    return ht


def get_data_point(data, index):
    '''Return data point by index'''
    X=data[index][:-1]
    # X should be numpy array with shape (n_samples, n_features)
    X=np.array([X])
    
    #print(X)
    return X


def get_class_value(data, index):
    '''Returns the class by index'''
    class_value = data[index][-1]
    
    #print(class_value)
    return np.array([class_value])


def get_f1_score(countTP, countFP, countFN):
    f1_score=0
    
    if (((countTP + countFP)==0) or ((countTP + countFN)==0)):
        f1_score=0
    else:
        f1_score = countTP/(countTP + 0.5*(countFP+countFN))
    
    return f1_score
         

def print_confusion_matrix(countInstances, countCorrectPredictions, countTP, countFP, countFN, countTN, current_f1_score):
    '''print the confusion matrix every 1000 data points'''
    if (countInstances>0) and (countInstances % 1000 ==0):
        print(f"{'Total'.ljust(6)} | {'Correct'.ljust(6)}| {'TP'.ljust(6)} | {'FP'.ljust(6)} | {'FN'.ljust(6)} | {'TN'.ljust(6)} | {'F-score'.ljust(6)}")
        print(f"{str(countInstances).ljust(6)} | {str(countCorrectPredictions).ljust(6)} | {str(countTP).ljust(6)} | {str(countFP).ljust(6)} | {str(countFN).ljust(6)} | {str(countTN).ljust(6)} | {str(round(current_f1_score,4)).ljust(6)}")
        

def train_with_reservoir_sampling(filename, N, ht, maturity_level):
    ''' Train Hoeffding Tree Model with Reservoir sampling
        Print the confusion matric every 1000 samples
        Params: 
            data - the data stream
            ht - the initialized Hoeffding Tree model
            N - the size of reservoir   
    '''
    #initialization of variables
    reservoir =[]          # reservoir for majority class
    minority_array = []    # array of the same size as the reservoir for minority class

    countInstances =0
    countCorrectPredictions=0
    countTP=0              # true positives
    countFP=0              # false positives
    countFN=0              # false negatives
    countTN=0              # true negatives

    result_accuracy=[]
    result_f1_score=[]
    majorityClass =0
    minorityClass =1
    countMinority =0       # 0 to N - to make sure we use equal number of both classes in trainning
    
    nbrMajorityTotal=0     # display total training instances from the majority class
    nbrMinorityTotal=0     # display total training instances from minority class
    
    data = data_preparation(filename)
    
    # stream the data
    for i in range(len(data)):

        # read the next example from the stream
        y = get_class_value(data, i) 
        X = get_data_point(data, i)

        # apply test then train method
        if (ht.get_model_measurements['Tree size (leaves)']>= maturity_level):
            countInstances +=1

            y_predict = ht.predict(X)

            # calculate prequential metrics
            if (y_predict[0] == y):
                countCorrectPredictions+=1
            if (y_predict[0]==1 and y==1):
                countTP+=1
            if (y_predict[0]==1 and y==0):
                countFP+=1
            if (y_predict[0]==0 and y==1):
                countFN+=1
            if (y_predict[0]==0 and y==0):
                countTN+=1

            current_accuracy = countCorrectPredictions/ countInstances        
            current_f1_score = get_f1_score(countTP, countFP, countFN)

            # print current confusion matrix
            print_confusion_matrix(countInstances, countCorrectPredictions, countTP, countFP, countFN, countTN, current_f1_score)

        else:
            current_f1_score=0
            current_accuracy=0

        # record current metric
        result_accuracy.append(current_accuracy)
        result_f1_score.append(current_f1_score)

        # now train the model
        if (y == majorityClass):
            if (len(reservoir) < N):
                # accumulate N elements in the reservoir
                reservoir.append(data[i])
            else:
                # uniformly sample data 
                m=random.randint(0, i)
                if (m<N):
                    reservoir[m]=data[i]          
        else:
            # need N data points in the reservoir before training
            if (len(reservoir) == N) : 
                countMinority+=1

                # train tree with the current sample (minority)   
                ht=ht.partial_fit(X = X,y = y)
                nbrMinorityTotal+=1

            # train with N points from the majority class as well
            if (countMinority ==N):

                for r in reservoir:    
                    ht=ht.partial_fit(X = np.array([r[:-1]]),y=np.array([r[-1]]))
                    nbrMajorityTotal+=1

                # empty the reservoir and start over
                reservoir=[]
                countMinority =0 

    print(f"Trained with majority class - {nbrMajorityTotal}")
    print(f"Trained with minority class - {nbrMajorityTotal}")
    
    return (result_f1_score, result_accuracy)

def main():

    # parameters
    filename="../dataset/iot_telemetry_data.csv"
    N=100               # the size of reservoir (more than the grace perod)
    maturity_level=10   # wait until the tree reaches certain number of leaves (maturity_level) to start prequential error evaluation

    # hyperparameters tuning
    grace_period=10           # default 200 - how many samples the node should observe to male a split
    split_criterion ='gini'   # default 'info-gain'
    split_confidence = 0.001  # default 1e-07
    leaf_prediction = 'mc'    # mc = majority class vote, default 'nba'
    binary_split = False

    # initialize the tree
    ht = initialize_model(grace_period,split_criterion, split_confidence,leaf_prediction,binary_split)

    # train the tree and obtain the prequential metric
    (result_f1_score, result_accuracy) = train_with_reservoir_sampling(filename, N, ht, maturity_level)

    print("Final model:*****************************")
    print(ht.get_model_measurements)

    plt.figure(figsize=(10,5))
    plt.plot(result_f1_score, label="F1-Score")
    plt.plot(result_accuracy, label="Accuracy")
    plt.legend()
    plt.xlabel("Total instances")
    plt.title("Accuracy and F1-Score of Hoeffding Tree")

if __name__=="__main__":
    main()




