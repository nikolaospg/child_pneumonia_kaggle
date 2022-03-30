import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
#This file contains some useful functions and classes (for the CNN)


#In case we want to print an image
def show_image(data, row, im_dims, has_class):

    if(has_class==0):
        current_image=data[row]
    else:
        current_image=data[row,0:-1]

    current_image=np.reshape(current_image, im_dims)

    print(current_image.shape)
    plt.imshow(current_image, cmap="gray")
    plt.show()


#this function is used to apply undersampling (hoping to face the class imbalance problem)
def reduce_train(train_patterns, train_classes, final_size):
    pneumonia_indices=np.where(train_classes==1)
    pneumonia_indices=pneumonia_indices[0]
    perm=np.random.permutation(len(pneumonia_indices))
    perm=perm[final_size:]  
    delete_indices=pneumonia_indices[perm]
    np.save("delete_indices1_"+str(final_size), delete_indices)

    train_patterns=np.delete(train_patterns, delete_indices, 0)
    train_classes=np.delete(train_classes, delete_indices, 0)
    return train_patterns,train_classes


#This function takes the train and the test dataset, it works on the feature vectors (preprocessing) and the classes
#and returns the results. We can add whatever transform we like
def transform_datasets(train_dataset,test_dataset, divide255_flag=0):


    train_patterns=train_dataset[:,0:-1]
    test_patterns=test_dataset[:,0:-1]
    train_classes=train_dataset[:,-1]
    test_classes=test_dataset[:,-1]

    #Preprocessing
    if(divide255_flag==1):
        train_patterns=train_patterns/255
        test_patterns=test_patterns/255



    return train_patterns,test_patterns,train_classes,test_classes


#You give the training set path and the test set path and get the dataset
def get_dataset(train_path, test_path, divide255_flag):
    training_dataset=np.load(train_path)
    test_dataset=np.load(test_path)

    train_patterns,test_patterns,train_classes,test_classes=transform_datasets(training_dataset, test_dataset, divide255_flag)
    return train_patterns,test_patterns,train_classes,test_classes





##################*UTILS REGARDING THE CNN*####################



class SequentialClassifier(nn.Module):

    def __init__(self, layers):
        super(SequentialClassifier, self).__init__()
        self.layers=layers          #Layers up until the final linear output
        self.loss=None            #The standard for the loss is 'None'
        self.optimizer=None       #Standard is none

    def __str__(self):
        return ("Sequential Classifier With:\n\nLayers %s\n\nOptimiser: %s\n\nLoss: %s\n" % (self.layers, self.optimizer, self.loss))

    #Setters for the optimiser and the loss:
    def set_optim(self,optimiser):
        self.optimizer=optimiser
    
    def set_loss(self, loss):
        self.loss=loss

    #This function takes the input and passes it through the model, stops just before applying the sigmoid
    def forward(self,X):
        out=self.layers(X)
        out=torch.reshape(out, shape=(len(out),))
        return out
    
    
    #This function takes the X input and makes a prediction. Flag=0-> Before Sigmoid, Flag=1-> After Sigmoid (Actual probabilities), Flag=2-> Final prediction (Int)
    def predict(self,X, flag):
        y=self.forward(X)
        if(flag==0):
            return y
        y=torch.sigmoid(y)
        if(flag==1):
            return y
        y=torch.round(y)
        return y
    
    #This function passes the X dataset through the model and returns the results (Classification report, Confusion Matrix, Time Needed, Loss) on a dictionary
    #if partial_evaluation=1, then we do not compute the classification report and the confusion matrix (this is done to avoid scikit learn because we cannot easily use tensors on the gpu)
    def evaluate(self, X, y, partial_evaluation=1):
        self=self.eval()
        #Getting the inference up until the linear output, counting the time needed:
        t1=time.time()
        preds=self.predict(X, 0)
        t2=time.time()

        #Converting to actual label predictions:
        preds=torch.sigmoid(preds)
        label_preds=torch.round(preds)

        correct_num = torch.sum(label_preds == y)
        accuracy=correct_num/len(X)
        #Getting Classification Report and Confusion Matrix:
        if(partial_evaluation==1):
            result_dict={"Time": t2-t1, "Accuracy": accuracy}
        else:
            y_cpu=y.to('cpu')                           #Moving to cpu to use sklearn
            label_preds_cpu=label_preds.to('cpu')
            from sklearn.metrics import classification_report, confusion_matrix
            report=classification_report(y_cpu, label_preds_cpu, digits=3)
            confusion_matrix=confusion_matrix(y_cpu, label_preds_cpu)
            result_dict={"Report": report, "Confusion_Matrix": confusion_matrix, "Time": t2-t1, "Accuracy": accuracy}

        if(self.loss!=None):
            J=self.loss(preds, y.float())
            result_dict['Loss']=J
        else:
            result_dict['Loss']='No_loss'
        self=self.train()
        return result_dict

    #Trains the model, gets the loss and accuracy for each epoch
    def fit(self, Train_dataset, my_loader, num_epochs, Validation_dataset=None):

        #Initialising the arrays that will create the history dictionary:
        train_accuracy=np.zeros(shape=(num_epochs,))
        train_loss=np.zeros(shape=(num_epochs,))
        val_accuracy=np.zeros(shape=(num_epochs,))
        val_loss=np.zeros(shape=(num_epochs,))

        #Training loop:
        t1_whole=time.time()
        for epoch in range(num_epochs):
            t1=time.time()
            for patterns,labels in my_loader:
                #Forward pass:
                predictions=self.forward(patterns)
                J=self.loss(predictions, labels.float())
                #Backward pass:
                self.optimizer.zero_grad()
                J.backward()
                self.optimizer.step()

            #Getting the results for this epoch, saving in the arrays:
            with torch.no_grad():
                t2=time.time()
                current_results=self.evaluate(Train_dataset.x, Train_dataset.y)
                train_accuracy[epoch]=current_results['Accuracy']
                train_loss[epoch]=current_results['Loss']
                if(Validation_dataset!=None):
                    val_results=self.evaluate(Validation_dataset.x, Validation_dataset.y)
                    val_accuracy[epoch]=val_results['Accuracy']
                    val_loss[epoch]=val_results['Loss']
                    print("Epoch=%d Train Loss=%f Train Accuracy=%f Val Loss=%f Val Accuracy=%f Time=%f" % (epoch+1, current_results['Loss'], current_results['Accuracy'], val_results['Loss'], val_results['Accuracy'],t2-t1))
                else:
                    print("Epoch=%d Loss=%f Accuracy=%f Time=%f" % (epoch+1, current_results['Loss'], current_results['Accuracy'], t2-t1))
        t2_whole=time.time()
        print("Total Time:", t2_whole-t1_whole)
        if(Validation_dataset!=None):
            return {"Train_accuracy": train_accuracy, "Train_loss": train_loss, "Val_accuracy": val_accuracy, "Val_loss": val_loss}  
        return {"Train_accuracy": train_accuracy, "Train_loss": train_loss}


class CustomDataset(Dataset):
    def __init__(self, X, y, device='cpu'):

        #If the inputs are np arrays convert to tensors:
        if(not torch.is_tensor(X)):
            X=torch.from_numpy(X)
            y=torch.from_numpy(y)
        X=X.to(device)
        y=y.to(device)
        self.x=X
        self.y=y

    def __getitem__(self,index):
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)
    
#This function reads the datasets but gives as Train/Validation/Test, and can give us arrays in the form (-1,1,height,width) to use for a CNN (only if is_CNN is true)
def read_datasets(path_to_train, path_to_test, val_perc=0.3, is_CNN=False):

    #Getting Train, Validation and Test sets:
    from sklearn.model_selection import train_test_split
    train_set=np.load(path_to_train)
    X_train=np.array(train_set[:,0:-1]/255,dtype=np.float32)
    y_train=train_set[:,-1]

    test_set=np.load(path_to_test)
    X_test=np.array(test_set[:,0:-1]/255,dtype=np.float32)
    y_test=test_set[:,-1]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_perc, random_state=42)
    if(is_CNN==True):
        X_train=np.reshape(X_train, newshape=(-1,1,30,40))
        X_val=np.reshape(X_val, newshape=(-1,1,30,40))
        X_test=np.reshape(X_test, newshape=(-1,1,30,40))

    return X_train,X_val,X_test,y_train,y_val,y_test



def plot_results(history):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(history['Train_accuracy'], label = 'Train')
    plt.plot(history['Val_accuracy'], label = 'Validation')
    plt.legend()
    plt.title('Accuracy through epochs')
    plt.show()

    plt.figure()
    plt.plot(history['Train_loss'], label = 'Train')
    plt.plot(history['Val_loss'], label = 'Validation')
    plt.legend()
    plt.title('Learning curves')
    plt.show()

