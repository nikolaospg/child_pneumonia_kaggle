import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import NearestCentroid

from utils import transform_datasets

#Function so I can easily run the Nearest Centroid and KNN algorithms easily
#Inputs:
#   1->     train_patterns: The patterns of the train set
#   2->     test_patterns:      ...
#   3->     train_classes:  The classes on the train set
#   4->     test_classes:       ...
#   5->     method:     "kNN" for kNN, "NC" for nearest centroid
#   6->     print_results_flag:     If you want to print the results, then set the flag equal to 1
#   7->     num_neighbours:         The number of neighbours (for the kNN classifier)
#It returns the test_set_report, test_set_confusion, train_set_report, train_set_confusion 
def neighbours_classification(train_patterns, test_patterns, train_classes, test_classes, method, print_results_flag=1, num_neighbours=1):
    import time

    if(method!="kNN" and method!="NC"):
        raise NameError('Pass "kNN" for kNN and "NC" for Nearest centroid!')
    
    if(method=="kNN"):
        model=KNeighborsClassifier(n_neighbors=num_neighbours)
    else:
        model=NearestCentroid()

    t1=time.time()
    model.fit(train_patterns, train_classes)
    t2=time.time()
    test_pred=model.predict(test_patterns)          #Test set predictions
    train_pred=model.predict(train_patterns)        #Train set predictions

    if(method=="kNN"):
        print("Time needed for training KNN with %d neighbours is %f" % (num_neighbours, t2-t1))
    else:
        print("Time needed for training Nearest Centroid is ", t2-t1)
    train_set_report=classification_report(train_classes, train_pred, digits=3)
    train_set_confusion=confusion_matrix(train_classes, train_pred)
    test_set_report=classification_report(test_classes, test_pred, digits=3)
    test_set_confusion=confusion_matrix(test_classes, test_pred)

    if(print_results_flag==1):
        print("\nTrain set results for %s:" %(method))
        if(method=="kNN"):
            print(num_neighbours,"Neighbours")
        print("*REPORT*")
        print(train_set_report)
        print("*CONFUSION MATRIX*")
        print(train_set_confusion)

        print("\nTest set results for %s:" %(method))        
        print("*REPORT*")
        print(test_set_report)
        print("*CONFUSION MATRIX*")
        print(test_set_confusion)
    return test_set_report, test_set_confusion, train_set_report, train_set_confusion


#Uncomment one of the following to get one kind of dataset (initial-SIFT/BOVW-WaveletScattering)
#Add the path to where it is saved to properly open it


#Loading my datasets - case of original datasets
# training_set=np.load("train_set_30x40.npy")
# test_set=np.load("test_set_30x40.npy")
# apply_PCA_flag=0
# perc_variance=0.975
# train_patterns,test_patterns,train_classes,test_classes=transform_datasets(training_set, test_set,apply_PCA_flag, perc_variance, 1)

# #Loading my datasets - case of sift dataset
# training_set=np.load("sift_dataset/RED2000train_dataset_cl3000.npy")
# test_set=np.load("sift_dataset/RED2000test_dataset_cl3000.npy")
# apply_PCA_flag=0
# perc_variance=0.975
# print("Working with SIFT features, descriptors per image is 2000 and num clusters is 3000")
# train_patterns,test_patterns,train_classes,test_classes=transform_datasets(training_set, test_set,apply_PCA_flag, perc_variance, 0)


# # #Loading my datasets - case of Wavelet scattering dataset
training_dataset=np.load("scattering_dataset/REDtrainJ3.npy")
test_dataset=np.load("scattering_dataset/REDtestJ3.npy")
train_patterns=training_dataset[:,0:-1]
test_patterns=test_dataset[:,0:-1]
train_classes=training_dataset[:,-1]
test_classes=test_dataset[:,-1]
print("Using Wavelet Dataset with J=3, reduced")


#Classifying 
neighbours_classification(train_patterns, test_patterns, train_classes, test_classes, "kNN", print_results_flag=1, num_neighbours=1)
neighbours_classification(train_patterns, test_patterns, train_classes, test_classes, "kNN", print_results_flag=1, num_neighbours=3)
neighbours_classification(train_patterns, test_patterns, train_classes, test_classes, "NC", print_results_flag=1)
