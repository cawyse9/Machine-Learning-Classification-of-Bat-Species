# Machine-Learning-Classification-of-Bat-Species

<img align="right" height=150 src="https://user-images.githubusercontent.com/29300100/196709192-fd2362ec-8d9c-4ebf-a4bd-c7ddd2b3e4b2.png">


## Background

In this project, numeric features of the acoustic properties of calls of a large sample of bats in Mexico are analysed using machine learning techniques to try to predict the familiy of each bat.  Monitoring the prescence and diversity of bats using acoustic surveys is a useful way to detect biodiversity change, especially in remote areas, but machine learning algorithms are required to address the complexity of classification of individial families.

## Objective
The objective of this project is to classify the acoustic information to predict the target variable "Family" which is a categorical variable indicating the following 5 bat families: _Emballonuridae, Molossidae, Mormoopidae, Phyllostomidae, Vespertilionidae_.  

## Data Processing
The performance of a neural network with 2 hidden layers and a neural network with 3 hidden layers was compared with respect to their performance at predicting the type of bat family a call belongs to. Performance was optimised by minimising the expected error evaluated on the training data:  

$$\overline{E}(w) = \frac{1}{N}\sum_{i = 1}^{N}E_i(w)$$

In this case of a categorical classification problem, the error function is the cross entropy: 

$$E(w) = -\sum_{t=1}^{N}\sum_{c=1}^{C}y_{ic}log(o_{ic})$$

Weights are updated based on minimising the expected value of the loss function.  The performance metrics used to compare the two models are accuracy and value of the error function at each epoch for the training and test datasets. The evolution of these parameters over the course of the 100 epochs on the training data is shown in the plots below.  
$~$

The data were recoded to define the bat classes as categorical and one-hot encoding was used to create a series of binary variables based on these categories (see R code below).  

##  Neural Networks

The *keras* package in R was used to fit two and three hidden layer neural networks. L2 regularization was used to apply penalties on the layer parameters that are summed into the loss function and that prevent over fitting of the model.  A dropout layer was included where randomly selected neurons are ignored during training so that their contribution to downstream neurons is temporarily removed.  This allows better generalization and makes the model less likely to overfit the training data.

The performance of the models was estimated by assessing their ability to correctly predict the classification of the bats in the test data, which was not involved in the derivation of the models.  Performance was also evaluated using a classification table which displays the numbers of bats that were correctly and incorrectly classified by each model.  The classification tables for models 2 and 3 are shown below.

Because the training set was used to learn the parameters that reduce the estimated training error, these estimating performance metrics are optimistic estimates, but in this case the test error seems to be greater than or equal to the estimated training error.

## Comparing Classification Performance of the Models
The overall accuracy of Model 2 was 0.8427 while Model 3 had a slightly higher accuracy of 0.8696.  The classification tables for both models above breaks down misclassification by bat species.  Both models perform poorly in classifying Molossidae, (37% and 57% correctly classified for model 2 and 3, respectively.  Model 2 and 3 approach their highest value achieved during the 100 epochs after around 40 epochs,  with the rate of performance improvement continuing at a slower rate thereafter.  Nevertheless, the  rate of improvement of performance for both models is still increasing at a very slow rate at the end of the 100 epochs.  Because the training set was used to learn the parameters that reduce the estimated training error, these estimating performance metrics are optimistic estimates, but in this case the test error seems to be greater than the estimated training error.  This could be because the data used for training and testing the model are not homogenous, that there is some difference between the two datasets that affects our ability to compare the performance of the models.  A more likely reason for higher test compared to training error in this case is the dropout regularization layer which is deactivated when evaluating the test data and can give an improved accuracy on test datasets.  When dropout is applied during model training, a percentage of the network nodes are switched off, leading to deliberate underfitting of the data. In contrast, during the testing stage all the network nodes are employed, giving increased fitting power, robustness and higher test accuracy.  The higher accuracy and lower loss function values seen in the test data fit here for model 2 and 3 suggest the dropout regularisation has had this effect. The loss and accuracy predicted by the models are plotted in Fig 2 below.  

The gap between test and training data is slightly larger for model 3 compared to 2, but again, this is a very small difference that probably doesn’t support a function improvement in performance for model 3.

## Conclusion
The overall accuracy of Model 2 was 0.8427 while Model 3 had a slightly higher accuracy of 0.8696, but this difference is small and probably doesn’t represent a functionally important difference in model performance between the two models. Rather, this small difference indicates that the addition of the regularisation step does not impact the performance of the model sufficiently to justify the increased complexity it brings to the model.  The accuracy increased to over 80% in the test data on both models (Figure 2), and it is possible that longer epochs might have improved the performance of both models further since accuracy was still increasing after 100 epochs.  There is no evidence of overfitting, since the test accuracy was greater than the training accuracy for both models.  The classification tables for both models below breaks down misclassification by bat species.  Both models perform well in classifying Mormoopidae, compared to the other bat species (94% and 89% correctly classified for model 2 and 3, respectively).    

## Acknowledgements
This project was awarded an A grade as part of the coursework required for the module, "Machine Learning" for an a MSc (Data Analytics) course at UCD in 2022. 
