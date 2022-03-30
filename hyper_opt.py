import numpy as np
from utils import get_dataset
import pandas as pd

#This file is used to optimise the hyperparameters for the SVM 


#Loading the dataset:
#Set the paths you want to:
train_path="trainJ3.npy"
# train_path="train_set_30x40.npy"
test_path="testJ3.npy"
train_patterns,test_patterns,train_classes,test_classes=get_dataset(train_path, test_path, 0)

#Creating SVM
from sklearn.svm import SVC 
pows=np.array([-9,-6,-2,-1,0,1,5,9], dtype="float32")
C_vals=2**pows
from sklearn.model_selection import cross_val_score

kernel="rbf"       #Choose "linear", "rbf", or "poly"

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

