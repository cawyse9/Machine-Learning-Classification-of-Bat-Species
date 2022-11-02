#####################################################################
# Bat Classification
#####################################################################

#target variable 5 bat families: Emballonuridae,Molossidae, Mormoopidae, Phyllostomidae, Vespertilionidae
#matrix data [training validation]
#matrix data_test [testing]

#load packages
library(keras)
library(tfruns)
library(reticulate)
library(tensorflow)
library(sigmoid)

#####################################################################
# Data Processing
#####################################################################

# load data
load("C:/Users/cwyse/OneDrive - Maynooth University/stats summer/machine learning/assign 2/data_bats_call (1).RData")

# store predictors and standardize
x <- data[,-1]
x_test <-data_test[-1]

x <- scale(x)
x_test<-scale(x_test)

V <- ncol(x) # number of predictors

# find y (outcome classification)
y<-data[1]
y_test<-data_test[1]

#make categorical variable
y<-ifelse(y=="Vespertilionidae",0,
  ifelse(y=='Molossidae',1,  
  ifelse(y=='Mormoopidae',2,   
  ifelse(y=='Phyllostomidae',4,    
  ifelse(y=='Emballonuridae',3,       
  999999)))))

y_test<-ifelse(y_test=="Vespertilionidae",0,
  ifelse(y_test=='Molossidae',1,  
  ifelse(y_test=='Mormoopidae',2,   
  ifelse(y_test=='Phyllostomidae',4,    
  ifelse(y_test=='Emballonuridae',3,       
  999999)))))

# convert from factor to numeric (0,1)
y_test<-as.numeric(y_test) 
y<-as.numeric(y)

# one hot encoding
y <- to_categorical(y)
y_test <- to_categorical(y_test)
K <- ncol(y) # number of classes of bat families

# check few observations
y[c(1,19,105,131),]
data[1][c(1,19,105,131),]

y_test[c(1,19,105,131),]
data_test[1][c(1,19,105,131),]

V <- ncol(x) # number of predictor variables

################################################################################
#  define 2 hidden layer model
################################################################################

#initialize the model 
model2 <- keras_model_sequential()

# we set the batch size to 1% of the training data
N <- nrow(x)
bs <- round(N * 0.01)

#define model with 40 units, 2 hidden layers relu activation function and softmax output activation function
model2 %>%
  layer_dense(units = 40, activation = "relu", name="layer1",input_shape = V, #layer dense sets properties of each layer, V is number of input units to layer, number of features of the hidden layer
    kernel_regularizer = regularizer_l2(0.009)) %>%
  layer_dropout(rate = .5) %>%
  layer_dense(units = 20, activation = "relu", name="layer2",input_shape = V, #layer dense sets properties of each layer, V is number of input units to layer, number of features of the hidden layer
    kernel_regularizer = regularizer_l2(0.009)) %>%
  layer_dropout(rate = .4) %>%
  layer_dense(units = K, activation = "softmax") #

#specify error function and optimization procedure for estimating NN weights
model2 %>% compile( #configure the model
  loss = "categorical_crossentropy",  
  optimizer = optimizer_sgd(), #optimization method
  metrics = "accuracy" #performance measure
)

# train the model and evaluate on test data at each epoch
fit2 <- model2 %>% fit(
  x = x, y = y,
  validation_data = list(x_test, y_test),
  epochs = 100,
  batch_size=bs,
  verbose = 1
)

################################################################################
#  define 3 hidden layer model
################################################################################

#initialize the model 
model3 <- keras_model_sequential()

#define model with 40 units, 2 hidden layers relu activation function and softmax output activation function
model3 %>%
  layer_dense(units = 40, activation = "relu", input_shape = V, #layer dense sets properties of each layer, V is number of input units to layer, number of features of the hidden layer
    kernel_regularizer = regularizer_l2(0.009)) %>%
  layer_dropout(rate = .5) %>%
  layer_dense(units = 20, activation = "relu", input_shape = V, #layer dense sets properties of each layer, V is number of input units to layer, number of features of the hidden layer
    kernel_regularizer = regularizer_l2(0.009)) %>%
  layer_dropout(rate = .4) %>%
  layer_dense(units = 10, activation = "relu", input_shape = V, #layer dense sets properties of each layer, V is number of input units to layer, number of features of the hidden layer
    kernel_regularizer = regularizer_l2(0.009)) %>%
  layer_dropout(rate = .3) %>%
  layer_dense(units = K, activation = "softmax") #
  
#specify error function and optimization procedure for estimating NN weights
model3 %>% compile( #configure the model
  loss = "categorical_crossentropy",  
  optimizer = optimizer_sgd(), #optimization method
  metrics = "accuracy" #performance measure
)

# train the model and evaluate on test data at each epoch
fit3 <- model3 %>% fit(
  x = x, y = y,
  batch_size=bs,
  validation_data = list(x_test, y_test),
  epochs = 100,
  verbose = 1
)

################################################################################
#  classification tables
################################################################################

# look at classification table for both models
y_raw2 <- max.col(y_test) - 1 # to convert back to categorical
class_hat2 <- model2 %>% predict(x_test) %>% max.col() # predicted classes
tab2 <- table(y_raw2, class_hat2)

y_raw3 <- max.col(y_test) - 1 # to convert back to categorical
class_hat3 <- model3 %>% predict(x_test) %>% max.col() # predicted classes
tab3 <- table(y_raw3, class_hat3)

# compute accuracy model2 and 3
acc_model2<-sum( diag(tab2) ) / sum(tab2)
acc_model3<-sum( diag(tab3) ) / sum(tab3)

###############################################################################
#  plots performance
###############################################################################

# to add a smooth line to points
smooth_line <- function(y) {
  x <- 1:length(y)
  out <- predict( loess(y ~ x) )
  return(out)
}
# some colors will be used later
cols <- c("navy", "lightblue", "darkgreen", "lightgreen")

#compare accuracy with 2 and 3 hidden layers
out <- cbind(fit2$metrics$accuracy,
             fit2$metrics$val_accuracy,
             fit3$metrics$accuracy,
             fit3$metrics$val_accuracy)
str(fit2$metrics)

# compare performance with 2 and 3 hidden layers
matplot(out, pch = 19, ylab = "Accuracy", xlab = "Epochs",
        col = adjustcolor(cols, 0.3), log = "y")
matlines(apply(out, 2, smooth_line), lty = 1, col = cols, lwd = 2)
legend("bottomright", legend = c("Train (Model2)", "Test (Model2)", "Train (Model3)", "Test (Model3)"),
       fill = cols, bty = "n")

# compare loss with 2 and 3 hidden layers
loss <- cbind(fit2$metrics$loss,
              fit2$metrics$val_loss,
              fit3$metrics$loss,
              fit3$metrics$val_loss)
matplot(loss, pch = 19, ylab = "Loss", xlab = "Epochs",
        col = adjustcolor(cols, 0.3))
matlines(apply(loss, 2, smooth_line), lty = 1, col = cols, lwd = 2)
legend("topright", legend = c("Train (Model2)", "Test (Model2)", "Train (Model3)", "Test (Model3)"),
       fill = cols, bty = "n")






