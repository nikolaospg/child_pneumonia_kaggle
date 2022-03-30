import numpy as np


#We use this file to create new datasets, either by undersampling the original one or by applying the scattering transform on it

#This function takes as inputs the training dataset, the test dataset and the J scaling coefficient, and computes the 
#features based on the wavelet scattering transform.
def wavelet_scattering(train_path,test_path, J):
    from kymatio.sklearn import Scattering2D
    import time
    
    t1=time.time()
    #Separating classes and feat.vectors
    training_dataset=np.load(train_path)
    test_dataset=np.load(test_path)
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


#this function is used to apply undersampling (hoping to face the class imbalance problem)
def reduce_train(train_path, final_size):

    training_dataset=np.load(train_path)
    train_patterns=training_dataset[:,0:-1]
    train_classes=training_dataset[:,-1]

    pneumonia_indices=np.where(train_classes==1)
    pneumonia_indices=pneumonia_indices[0]

    perm=np.random.permutation(len(pneumonia_indices))
    perm=perm[final_size:]  
    delete_indices=pneumonia_indices[perm]

    train_patterns=np.delete(train_patterns, delete_indices, 0)
    train_classes=np.delete(train_classes, delete_indices, 0)
    return train_patterns,train_classes


apply_scattering_flag=0
apply_undersampling_flag=1
train_path="train_set_30x40.npy"                #The path of the current train set (Before changing)
test_path="test_set_30x40.npy"


#In case we  want to apply wavelet scattering:
if(apply_scattering_flag==1):
    new_train_path="trainJ3.npy"        #Set the paths for the new datasets
    new_test_path="testJ3.npy"
    J=3                     #Change this if you want to
    new_train,new_test,train_classes,test_classes=wavelet_scattering(train_path,test_path, J)           #Transforming

    train_set=np.concatenate((new_train, np.reshape(train_classes, newshape=(len(train_classes),1))), axis=1)
    test_set=np.concatenate((new_test, np.reshape(test_classes, newshape=(len(test_classes),1))), axis=1)

    np.save(new_train_path, train_set)
    np.save(new_test_path, test_set)

#In case we want to undersample
if(apply_undersampling_flag==1):
    new_train_path="1300reduced_train_set_30x40.npy"        #Set the paths for the new datasets
    final_size=1300                               #Final size for the pneumonia class
    red_train_patterns,red_train_classes=reduce_train(train_path, final_size)
    red_train_set=np.concatenate((red_train_patterns, np.reshape(red_train_classes, newshape=(len(red_train_classes),1))), axis=1)
    np.save(new_train_path, red_train_set)





