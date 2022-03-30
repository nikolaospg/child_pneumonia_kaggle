import numpy as np
from utils import get_dataset
import time

#In this file I use an SVM to solve the classification problem. 



#Loading the dataset:
#Set the paths you want to
# train_path="trainJ3.npy"
train_path="train_set_30x40.npy"
test_path="test_set_30x40.npy"
train_patterns,test_patterns,train_classes,test_classes=get_dataset(train_path, test_path, 0)
print(len(train_patterns), len(test_patterns))

#Creating SVM
from sklearn.svm import SVC 
C=32
kernel="rbf"
param=0.0156    #Degree for the polynomial, gamma for the rbf

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
