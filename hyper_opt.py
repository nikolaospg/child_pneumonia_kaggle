import numpy as np
from sklearn import metrics
from utils import transform_datasets, wavelet_scattering
import time
import pandas as pd

#Uncomment one of the following to get one kind of dataset (initial-SIFT/BOVW-WaveletScattering)
#Add the path to where it is saved to properly open it

# #Loading my datasets - case of original datasets
# training_set=np.load("train_set_30x40.npy")
# test_set=np.load("test_set_30x40.npy")
# apply_PCA_flag=0
# perc_variance=0.975
# train_patterns,test_patterns,train_classes,test_classes=transform_datasets(training_set, test_set,apply_PCA_flag, perc_variance, divide255_flag=1)

# #Loading my datasets - case of sift dataset
# training_set=np.load("sift_dataset/RED2000train_dataset_cl3000.npy")
# test_set=np.load("sift_dataset/RED2000test_dataset_cl3000.npy")
# apply_PCA_flag=0
# perc_variance=0.975
# print("Working with SIFT features, descriptors per image is 2000 and num clusters is 3000")
# train_patterns,test_patterns,train_classes,test_classes=transform_datasets(training_set, test_set,apply_PCA_flag, perc_variance, divide255_flag=0)

# # #Loading my datasets - case of Wavelet scattering dataset
training_dataset=np.load("scattering_dataset/REDtrainJ1.npy")
test_dataset=np.load("scattering_dataset/REDtestJ1.npy")
train_patterns=training_dataset[:,0:-1]
test_patterns=test_dataset[:,0:-1]
train_classes=training_dataset[:,-1]
test_classes=test_dataset[:,-1]





#Creating SVM
from sklearn.svm import SVC 
pows=np.array([-9,-6,-2,-1,0,1,5,9], dtype="float32")
C_vals=2**pows
from sklearn.model_selection import cross_val_score

kernel="rbf"       #Choose "linear", "rbf", or "poly"



#UNCOMMENT ON OF THE FOLLOWING TO APPLY HYPERPARAMETER OPTIMISATION FOR ONE KIND OF KERNEL

#Linear
print("Running linear kernel")
accuracies=[]
for C in C_vals:
    current_SVM=SVC(C=C, kernel="linear")
    scores = cross_val_score(current_SVM, train_patterns, train_classes, n_jobs=4, cv=5)
    accuracies.append(scores.mean())
accuracies=np.array(accuracies)
accuracies=np.reshape(accuracies, (1,len(C_vals)))
accuracies=pd.DataFrame(accuracies, columns=C_vals, index=["Acc:"])
print("Linear SVM")
print("Accuracies (Columns-> C values):")
print(accuracies)
best_C=C_vals[np.argmax(accuracies)]
print("\nThe optimal C is:",best_C)


# #RBF
# print("running rbf kernel")
    
# pows=np.array([-1,0,1,5,9,12], dtype="float32")
# gamma_vals=2**pows

# total_gammas=len(gamma_vals)
# total_C=len(C_vals)
# accuracies=np.zeros(shape=(total_gammas,total_C))
# for i1 in range(len(gamma_vals)):
#     gamma=gamma_vals[i1]
#     for i2 in range(len(C_vals)):
#         C=C_vals[i2]
#         current_SVM=SVC(C=C, kernel="rbf", gamma=gamma)
#         scores = cross_val_score(current_SVM, train_patterns, train_classes, n_jobs=4, cv=5)
#         accuracies[i1][i2]=scores.mean()
        
# accuracies=pd.DataFrame(accuracies, columns=C_vals, index=gamma_vals)
# print("RBF SVM")
# print("Accuracies (columns->C values, Rows-> gamma values")
# print(accuracies)
# accuracies=np.array(accuracies)
# maxindex = accuracies.argmax()
# opt_indices=np.unravel_index(maxindex, accuracies.shape)
# print("Optimal gamma is ",gamma_vals[opt_indices[0]]," Optimal C is ",C_vals[opt_indices[1]])




# #Polynomial
# print("running polynomial kernel")
# degree_vals=[2,3,5,7,10]

# total_degrees=len(degree_vals)
# total_C=len(C_vals)
# accuracies=np.zeros(shape=(total_degrees,total_C))
# for i1 in range(len(degree_vals)):
#     degree=degree_vals[i1]
#     for i2 in range(len(C_vals)):
#         C=C_vals[i2]
#         current_SVM=SVC(C=C, kernel="poly", degree=degree)
#         scores = cross_val_score(current_SVM, train_patterns, train_classes, n_jobs=4, cv=5)
#         accuracies[i1][i2]=scores.mean()

# accuracies=pd.DataFrame(accuracies, columns=C_vals, index=degree_vals)
# print("Polynomial SVM")
# print("Accuracies (columns->C values, Rows-> degree values")
# print(accuracies)
# accuracies=np.array(accuracies)
# maxindex = accuracies.argmax()
# opt_indices=np.unravel_index(maxindex, accuracies.shape)
# print("Optimal degree is ",degree_vals[opt_indices[0]]," Optimal C is ",C_vals[opt_indices[1]])

