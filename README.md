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

Weights are updated based on minimising the expected value of the loss function.  The performance metrics used to compare the two models are accuracy and value of the error function at each epoch for the training and test datasets. L2 regularization was used to apply penalties on the layer parameters that are summed into the loss function and that prevent over fitting of the model.  A dropout layer was included where randomly selected neurons are ignored during training so that their contribution to downstream neurons is temporarily removed.  This allows better generalization and makes the model less likely to overfit the training data.

Further details of the analysis are available [here](https://github.com/cawyse9/Machine-Learning-Classification-of-Bat-Species/blob/main/Analysis%20and%20Code/Application%20of%20a%20Neural%20Network%20for%20Bat%20Family%20Classification%20from%20Call%20Sounds.pdf)
The R-code is available [here](https://github.com/cawyse9/Machine-Learning-Classification-of-Bat-Species/blob/main/Analysis%20and%20Code/bats.R)

## Conclusion
The overall accuracy of Model 2 was 0.8427 while Model 3 had a slightly higher accuracy of 0.8696, but this difference is small and probably doesn’t represent a functionally important difference in model performance between the two models. Both models perform well in classifying Mormoopidae, compared to the other bat species (94% and 89% correctly classified for model 2 and 3, respectively).    

## Acknowledgements
This project was awarded an A grade as part of the coursework required for the module, "Machine Learning" for an a MSc (Data Analytics) course at UCD in 2022. 
