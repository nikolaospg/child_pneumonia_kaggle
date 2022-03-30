from os import path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import time
import pickle           #Used to save the model


#Gets the histogram for one image:
#Pass:
#1) nearest_centers->The result after the kmeans prediction (finding the nearest centroid cluster center for every descriptor)
#2) num_centers->   The number of centers (clusters)
def extract_histogram(nearest_centers, num_centers):
    norm_hist=np.zeros(shape=(num_centers,))
    
    for i in range(num_centers):
        norm_hist[i]=len(np.where(nearest_centers==i)[0])

    norm_hist=norm_hist/len(nearest_centers)
    return norm_hist    

#Gets the number of clusters and the path to the train descriptors, returns the kmeans model
#Using minibatch kmeans to cluster. Returns the whole kmeans model (including the centers)
def compute_vocabulary(num_clusters, train_descr_path, save_model=0, model_path=""):
    train_descriptors=np.load(train_descr_path)
    
    
    #First converting the 3D array to a 2D, getting rid of the NaN, and computing the centers from the clustering process -> the vocabulary
    import pandas as pd             #Used to load the dropna command
    final_descriptors_dims=train_descriptors.shape
    final_descriptors=np.reshape(train_descriptors, newshape=(final_descriptors_dims[0]*final_descriptors_dims[1], final_descriptors_dims[2]))
    final_descriptors=pd.DataFrame(final_descriptors)
    final_descriptors=final_descriptors.dropna(axis=0)
    final_descriptors=np.array(final_descriptors)
    #Finished converting to 2D and getting rid of the NaN
    #Applying mini batch kmeans:
    t1=time.time()
    kmeans=MiniBatchKMeans(n_clusters=num_clusters, max_iter=1000000, random_state=0, batch_size=4096)          #Using MiniBatch kmeans mainly for memory purposes 
    kmeans.fit(final_descriptors.astype("float32"))
    t2=time.time()

    #Saving only if the flag is 1:
    if(save_model==1):
        pickle.dump(kmeans, open(model_path, 'wb'))             #Using pickle to save my model

    print("Time needed to apply the minibatch kmeans is ",t2-t1)
    return kmeans
    
#This function gets the path to a dataset (either the train or the test) and gets the BOVW features
#The classes input is a tuple which tells us how many pneumonia and how many normal patterns there are on this dataset
#e.g. (3883,1349) for the train set
def feature_extraction(path_to_dataset, kmeans_model, classes):
    import pandas as pd

    descriptors=np.load(path_to_dataset)
    descriptors_dims=descriptors.shape
    num_clusters=len(kmeans_model.cluster_centers_)

    features=np.zeros(shape=(descriptors_dims[0], num_clusters))

    #Every iteration corresponds to one image
    t1=time.time()
    for i in range(descriptors_dims[0]):
        current_descriptors=descriptors[i]

        #Getting rid of the NaN rows
        current_descriptors=pd.DataFrame(current_descriptors)
        current_descriptors=current_descriptors.dropna(axis=0)
        current_descriptors=np.array(current_descriptors)

        current_nearest_centers=kmeans_model.predict(current_descriptors)           #Calculating the nearest centers
        features[i]=extract_histogram(current_nearest_centers, num_clusters)
    t2=time.time()
    print(t2-t1)

    #Adding the column with the classes
    pn_classes=np.ones(shape=(classes[0],))
    nr_classes=np.zeros(shape=(classes[1],))
    classes_col=np.concatenate((pn_classes,nr_classes), axis=0)
    classes_col=np.reshape(classes_col, newshape=(len(classes_col), 1))
    final_dataset=np.concatenate((features,classes_col), axis=1)
    return final_dataset


#Initialising Hyperparams etc:
train_descr_path="descriptors/REDdescriptors_train2000.npy"
num_clusters=3000
model_path="kmeans_models/des2000_cl" + str(num_clusters)           #Path for the model to be saved (in case we want it to be saved)
save_model=0             
print("Num clusters=",num_clusters)                                       #1 if we want it to be saved, 0 if we do not
print("Working with reduced dataset")
#Getting the kmeans model (the centers is the vocabulary)
kmeans=compute_vocabulary(num_clusters, train_descr_path, save_model, model_path)

#Now getting the features for every image
path_to_dataset="descriptors/REDdescriptors_train2000.npy"
classes_train=[1750,1349]           #If i Didnt use the reduced dataset then this value should be [3883,1349]
print("Time needed to get the features for the train dataset is ",end="")
train_dataset=feature_extraction(path_to_dataset, kmeans, classes_train) 

path_to_dataset="descriptors/descriptors_test2000.npy"
classes_test=[390,234]
print("Time needed to get the features for the test dataset is ",end="")
test_dataset=feature_extraction(path_to_dataset, kmeans, classes_test) 

#Saving the datasets (they can be used by an SVM etc.)
np.save("sift_dataset/RED2000train_dataset_cl"+str(num_clusters), train_dataset)
np.save("sift_dataset/RED2000test_dataset_cl"+str(num_clusters), test_dataset)
