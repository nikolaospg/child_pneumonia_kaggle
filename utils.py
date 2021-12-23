import os 
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



def show_image(data, row, im_dims, has_class):

    if(has_class==0):
        current_image=data[row]
    else:
        current_image=data[row,0:-1]

    current_image=np.reshape(current_image, im_dims)

    print(current_image.shape)
    plt.imshow(current_image, cmap="gray")
    plt.show()


#Function to easily apply PCA for dimensionality reduction.
#The num_components is the number of features the new, transformed dataset is going to consist of.
#The transformed datasets are returned. The PCA object is also returned, in case we need to make further inspections.
def apply_PCA(train_patterns, test_patterns, prc_variance):

    #First I standardise the datasets
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    my_scaler=preprocessing.StandardScaler()
    my_scaler.fit(train_patterns)           #The PCA is fit on the train patterns

    train_patterns_std=my_scaler.transform(train_patterns)      #The standardised datasets
    test_patterns_std=my_scaler.transform(test_patterns)

    PCA_object=PCA(n_components=prc_variance, svd_solver="full")
    PCA_object.fit(train_patterns_std)

    train_patterns_std=PCA_object.transform(train_patterns_std)
    test_patterns_std=PCA_object.transform(test_patterns_std)

    return train_patterns_std,test_patterns_std,PCA_object


#I call this function in order to reduce the amount of images in the abundant class (reducing images from train pneumonia)
def reduce_train(train_dataset, test_dataset, train_pn_size):

    #Separating classes and feat.vectors
    train_patterns=train_dataset[:,0:-1]
    test_patterns=test_dataset[:,0:-1]
    train_classes=train_dataset[:,-1]
    test_classes=test_dataset[:,-1]

    #Getting the indices to be deleted with a randomised way
    pneumonia_indices=np.where(train_classes==1)
    pneumonia_indices=pneumonia_indices[0]
    perm=np.random.permutation(len(pneumonia_indices))
    perm=perm[train_pn_size:]  
    delete_indices=pneumonia_indices[perm]
    
    #Deleting some tuples
    train_patterns=np.delete(train_patterns, delete_indices, 0)
    train_classes=np.delete(train_classes, delete_indices, 0)

    #Merging and saving
    train_classes=np.reshape(train_classes, newshape=(len(train_classes), 1))
    test_classes=np.reshape(test_classes, newshape=(len(test_classes), 1))
    train_set=np.concatenate((train_patterns, train_classes), axis=1)
    test_set=np.concatenate((test_patterns, test_classes), axis=1)
    np.save("2reduced_train_"+str(train_pn_size),train_set)
    np.save("2reduced_test_"+str(train_pn_size),test_set)

    #Converting the classes again to 1D vector form
    train_classes=np.reshape(train_classes, newshape=(len(train_classes),))
    test_classes=np.reshape(test_classes, newshape=(len(test_classes),))
    return train_patterns,test_patterns,train_classes,test_classes


#This function takes the train and the test dataset, it works on the feature vectors (preprocessing) and the classes
#and returns the results
def transform_datasets(train_dataset,test_dataset, apply_PCA_flag, perc_variance, divide255_flag=0):


    train_patterns=train_dataset[:,0:-1]
    test_patterns=test_dataset[:,0:-1]
    train_classes=train_dataset[:,-1]
    test_classes=test_dataset[:,-1]

    # train_patterns,train_classes=reduce_train(train_patterns, train_classes, 500)
    # delete_indices=np.load("delete_indices_1500.npy")
    # train_patterns=np.delete(train_patterns, delete_indices, 0)
    # train_classes=np.delete(train_classes, delete_indices, 0)

    #Preprocessing
    if(divide255_flag==1):
        train_patterns=train_patterns/255
        test_patterns=test_patterns/255

    if(apply_PCA_flag==1):
        train_patterns,test_patterns,PCA_object=apply_PCA(train_patterns, test_patterns, perc_variance)
        print("The principal components chosen are",PCA_object.n_components_)
    

    return train_patterns,test_patterns,train_classes,test_classes


#This function takes as inputs the training dataset, the test dataset and the J scaling coefficient, and computes the 
#features based on the wavelet scattering transform.
def wavelet_scattering(training_dataset,test_dataset, J):
    from kymatio.sklearn import Scattering2D
    import time
    
    t1=time.time()
    #Separating classes and feat.vectors
    train_patterns=training_dataset[:,0:-1]
    test_patterns=test_dataset[:,0:-1]
    train_classes=training_dataset[:,-1]
    test_classes=test_dataset[:,-1]

    train_patterns=train_patterns/255
    test_patterns=test_patterns/255
    
    Scatter_object = Scattering2D(J, (30,40))       

    #Calculating the number of scattering features that we will get. I do this calculation so I can create numpy arrays to hold my datasets
    some_image=np.reshape(train_patterns[0], newshape=(30,40))
    scattered=Scatter_object.scattering(some_image)
    scattered=scattered.flatten()
    num_features=len(scattered)         #This is the number of features

    new_train=np.ones(shape=(len(train_classes), num_features))
    new_test=np.ones(shape=(len(test_classes), num_features))
    #Looping over all of the training patterns
    for i in range(len(train_patterns)):
        current_image=np.reshape(train_patterns[i], newshape=(30,40))
        scattered=Scatter_object.scattering(current_image)
        scattered=scattered.flatten()

        new_train[i]=scattered

    for i in range(len(test_patterns)):
        current_image=np.reshape(test_patterns[i], newshape=(30,40))
        scattered=Scatter_object.scattering(current_image)
        scattered=scattered.flatten()

        new_test[i]=scattered
    t2=time.time()

    print("Time needed to extract the Scattering features (both training and test) with J=%d is %f" % (J,t2-t1))

    return new_train,new_test,train_classes,test_classes


