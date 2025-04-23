# Child Pneumonia Kaggle
In this project I use an SVM classifier, so that [the problem preserved](https://www.kaggle.com/andrewmvd/pediatric-pneumonia-chest-xray) is tackled in its binary form (where class imbalance is also present).  

The project includes trying to solve the problem with an SVM plus:

simple **image vectorisation**,

using SIFT features in a **Bag of Visual words** model implementation,

and finally using the **wavelet scattering transform** (from the [kymatio package](https://github.com/kymatio/kymatio)).  

By randomised subsampling on the prevalent class and the wavelet scattering transform, an accuracy of 85.3% was achieved (with an 93% recall on the pneumonia class) and with more conservative subsampling techniques recall>99% can be achieved (although with accuracy close to 80%), using the RBF SVM.

I proceed by using a **CNN** based on the Lenet-5 architecture using PyTorch, mainly to make comparisons with the previous models. After experimenting, by using a subsampled dataset coupled with Dropout regularisation I managed to attain an accuracy around 83% with a recall around 95%. Therefore according to my experiments, in this not very large dataset there is no clear winner(Scattering Transform + Kernel SVM vs CNN). 

The hyperparameters are not yet optimised and there might be still room for improvement.
