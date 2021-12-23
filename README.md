# Child Pneumonia Kaggle
Using knn, Nearest Centroid and SVM classifiers, [the problem preserved](https://www.kaggle.com/andrewmvd/pediatric-pneumonia-chest-xray) is tackled in its binary form (where class imbalance is also present). The project includes trying to solve the problem with a simple image vectorisation, using SIFT features in a Bag of Visual words model, and also using the wavelet scattering transform (using the [kymatio package](https://github.com/kymatio/kymatio)).\
By randomised subsampling on the prevalent class and the wavelet scattering transform, an accuracy of 85.3% was achieved (with an 93% recall on the pneumonia class) and with more conservative subsampling techniques recall>99% can be achieved (although with accuracy close to 80%), using the RBF SVM. The hyperparameters are not yet optimised and there is still room for improvement.
