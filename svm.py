import numpy as np
from utils import transform_datasets, wavelet_scattering
import time

#Uncomment one of the following to get one kind of dataset (initial-SIFT/BOVS-WaveletScattering)
#Add the path to where it is saved to properly open it

#Loading my datasets - case of original dataset
training_set=np.load("reduced_train_1500.npy")
test_set=np.load("reduced_test_1500.npy")
apply_PCA_flag=0
perc_variance=0.975
train_patterns,test_patterns,train_classes,test_classes=transform_datasets(training_set, test_set, apply_PCA_flag, perc_variance, divide255_flag=1)


# #Loading my datasets - case of sift dataset
# training_set=np.load("sift_dataset/RED2000train_dataset_cl3000.npy")
# test_set=np.load("sift_dataset/RED2000test_dataset_cl3000.npy")
# apply_PCA_flag=0
# perc_variance=0.975
# train_patterns,test_patterns,train_classes,test_classes=transform_datasets(training_set, test_set, apply_PCA_flag, perc_variance, divide255_flag=0)
# print("Working with SIFT features, descriptors per image is 2000 and num clusters is 3000,reduced dataset")


# # #Loading my datasets - case of Wavelet scattering dataset
# training_dataset=np.load("scattering_dataset/REDtrainJ3.npy")
# test_dataset=np.load("scattering_dataset/REDtestJ3.npy")
# train_patterns=training_dataset[:,0:-1]
# test_patterns=test_dataset[:,0:-1]
# train_classes=training_dataset[:,-1]
# test_classes=test_dataset[:,-1]


#Creating SVM
from sklearn.svm import SVC 
C=1
kernel="rbf"
param=0.0152    #Degree for the polynomial, gamma for the rbf

if (kernel=="linear"):
    my_SVM=SVC(kernel="linear", C=C)
    print("Using Linear SVM with C=",C)
elif(kernel=="rbf"):
    my_SVM=SVC(kernel="rbf", C=C, gamma=param)
    print("Using %s SVM with C=%f and gamma=%f" % (kernel,C,param))
else:
    my_SVM=SVC(kernel="poly", C=C, degree=param)
    print("Using %s SVM with C=%f and degree=%d" % (kernel,C,param))


#Fitting/Predicting:
t1=time.time()
my_SVM.fit(train_patterns, train_classes)
t2=time.time()
train_preds=my_SVM.predict(train_patterns)
test_preds=my_SVM.predict(test_patterns)

#Results
from sklearn.metrics import classification_report,confusion_matrix
train_set_report=classification_report(train_classes, train_preds, digits=3, zero_division=0)
train_set_confusion=confusion_matrix(train_classes, train_preds)
test_set_report=classification_report(test_classes, test_preds, digits=3, zero_division=0)
test_set_confusion=confusion_matrix(test_classes, test_preds)


print("Time needed for Training=",t2-t1)
num_SVs=my_SVM.n_support_
print("The number of support vectors are",np.sum(num_SVs))
print("Train set results for %s:" %(kernel))
print("*REPORT*")
print(train_set_report)
print("*CONFUSION MATRIX*")
print(train_set_confusion)
print("\nTest set results for %s:" %(kernel))        
print("*REPORT*")
print(test_set_report)
print("*CONFUSION MATRIX*")
print(test_set_confusion)
