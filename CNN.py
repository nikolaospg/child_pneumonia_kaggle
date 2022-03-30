import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import CustomDataset, SequentialClassifier,read_datasets,plot_results
import numpy as np
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialising the Sequential Model:#
is_CNN=True
layers=nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),                    #Output is 30x40
    nn.Dropout2d(0.2),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),                                  #Output is 15x20 (halved)
    nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),                   #Output is 11x16
    nn.Dropout2d(0.2),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),                                  #Output is 5x8
    nn.Flatten(),
    nn.Linear(16*5*8,120),  
    nn.Dropout2d(0.3),
    nn.ReLU(),  
    nn.Linear(120,84),
    nn.Dropout2d(0.1),
    nn.ReLU(),
    nn.Linear(84, 1)
)
my_classifier=SequentialClassifier(layers)
my_classifier=my_classifier.to(device)

#Finished initialising

#Learning rate, optimiser, batch size, other params#
batch_size=64
learning_rate=0.001
optimiser=torch.optim.Adam(my_classifier.parameters(), lr=learning_rate)
my_loss=nn.BCEWithLogitsLoss()
my_classifier.set_loss(my_loss)
my_classifier.set_optim(optimiser)
#Finished with the other params#


#Getting Train/Val/Test sets#
X_train,X_val,X_test,y_train,y_val,y_test=read_datasets("reduced_train_set_30x40.npy", "test_set_30x40.npy", is_CNN=True)

Train_dataset=CustomDataset(X_train,y_train)
Validation_dataset=CustomDataset(X_val,y_val)
Test_dataset=CustomDataset(X_test,y_test)
train_loader=DataLoader(Train_dataset, batch_size=batch_size, shuffle=True)     #Dataloader used for batches
#Finished Getting The datasets#

#Fitting to get the learning curves, with the validation set#
fitting_history=my_classifier.fit(Train_dataset, train_loader, 10, Validation_dataset)
plot_results(fitting_history)
# #Finished getting the learning curves#

#Final Training#

#Creating a second model now
fmy_classifier=SequentialClassifier(layers)
my_classifier=my_classifier.to(device)
my_classifier.set_loss(my_loss)
my_classifier.set_optim(optimiser)

X_train=np.concatenate([X_train,X_val])
y_train=np.concatenate([y_train,y_val])
Train_dataset_final=CustomDataset(X_train,y_train)
train_loader_final=DataLoader(Train_dataset_final, batch_size=batch_size, shuffle=True)     #Dataloader used for batches
fitting_history=my_classifier.fit(Train_dataset_final, train_loader_final, 10)
#Got the final model

#Evaluating#
with torch.no_grad():
    results_train=my_classifier.evaluate(Train_dataset.x, Train_dataset.y, 0)
    results_test=my_classifier.evaluate(Test_dataset.x, Test_dataset.y, 0)

    print("\n*The train set results*")
    print("Conf Matrix")
    print(results_train['Confusion_Matrix'])
    print("\nClass report")
    print(results_train['Report'])
    print("Time Needed")
    print(results_train['Time'])

    print("\n*The test set results*")
    print("Conf Matrix")
    print(results_test['Confusion_Matrix'])
    print("\nClass report")
    print(results_test['Report'])
    print("Time Needed")
    print(results_test['Time'])
