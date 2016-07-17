# Practical Machine Learning Project



###Summary:

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement. A group of 6 young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: 

Class A - Exactly according to the specification  
Class B - Throwing the elbows to the front  
Class C - Lifting the dumbbell only halfway  
Class D - Lowering the dumbbell only halfway   
Class E - Throwing the hips to the front   

See http://groupware.les.inf.puc-rio.br/har#ixzz4EURcK6Bp for more detailed information.

The training data for this project are available here:  
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv  
The test data are available here:  
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv  

###Goal:

The goal of the project is to predict the manner in which they did the exercise. The Classe variable is the outcome and of interest is: 

How is the model build?  
How is cross validation done?  
What is expected out of sample error and why?  
20 sample cases need to be predicted with the model for the Course Project Prediction Quiz  


###Loading, Splicing and Cleaning Data


```r
# loading required libraries
library(parallel);library(doParallel);library(caret); library(randomForest); library(rpart)
```



```r
#loading data 
training <- read.csv("pml-training.csv", header=T, na.strings = c("NA", "", "#DIV/0!"))
testing <- read.csv("pml-testing.csv", header=T, na.strings = c("NA", "", "#DIV/0!"))

# splitting 15% of the training data off for final cross validation testing
set.seed(1234)
intrain <- createDataPartition(training$classe, list= F, p=0.85)
mytrain <- training[intrain,] 
mytest <- training[-intrain,]
cbind(dim(mytrain)[1], dim(mytest)[1]) # number of observations in mytrain/mytest
```

```
##       [,1] [,2]
## [1,] 16680 2942
```

```r
# checking for NA values to remove the variables for high NA numbers
nacolumns <- colSums(is.na(mytrain))
quantile(nacolumns)  # shows that columns do have 0 NAs or around 16400 out of 16680
```

```
##    0%   25%   50%   75%  100% 
##     0     0 16332 16332 16680
```

```r
goodcol <- names(which(nacolumns < 1000))
mytrain <- mytrain[,goodcol]
mytrain <- mytrain[,-c(1,2,3,4,5,6,7)] # removing the frist 7 columns, those are no real predictors
dim(mytrain) # 53 possible features are left now
```

```
## [1] 16680    53
```

After removing variables with a very high NA number and also the first 7 columns as those are not real predictors, there are 53 possible features left to fit a model on.

###Model Building

For classification prediction model building "Classification Tree"" or "Random Forest" are the most common approaches to be used. I'm building both models to see which one gives a better accuracy.  

#### a) Classification Tree Model

```r
set.seed(1234)
#for parallel processing 
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

fControl <- trainControl(method = "cv", # cross validation 
                           number = 10, # 10 k-fold
                           allowParallel = TRUE)


# train with rpart
fitTree <-  train(classe ~ ., method="rpart",data=mytrain, trControl = fControl)
stopCluster(cluster)
fitTree # show results
```

```
## CART 
## 
## 16680 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 15013, 15011, 15013, 15012, 15012, 15012, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa     
##   0.03619000  0.5064715  0.36048102
##   0.05945101  0.4276469  0.22863412
##   0.11368015  0.3145638  0.04604001
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03619.
```

The classification model build has just around 50% accuracy. There is no need to do any further validation on this model and go ahead with a different approach.

#### b) Random Forest Model

```r
set.seed(1234)
# for parallel processing 
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

fControl <- trainControl(method = "cv", # cross validation 
                           number = 10, # 10 k-fold
                           allowParallel = TRUE)


# train with rf
fitRf <- train(classe ~ ., method="rf",data=mytrain, trControl = fControl)
stopCluster(cluster)
fitRf
```

```
## Random Forest 
## 
## 16680 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 15013, 15011, 15013, 15012, 15012, 15012, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9939450  0.9923399
##   27    0.9940048  0.9924160
##   52    0.9905273  0.9880173
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
fitRf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.55%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4739    2    1    0    1 0.0008433481
## B   21 3200    7    0    0 0.0086741016
## C    0   11 2887   11    0 0.0075627363
## D    0    1   23 2708    2 0.0095098756
## E    0    1    4    6 3055 0.0035877365
```

The results are much better for the Random Forest Model with an accuracy of 99.40%. 
The Out of Bag error is just 0.55%

###Cross Validation and "Out of Sample Error"

The Caret Train packages includes already cross validation through resampling on the given train data. In the above models trainControl has been defined with method = "cv" and number of k-fold with 10.  

To get the expected "Out of Sample Error", we need to run prediction for the model with new data. We are using the held 15% in mytest data set to cross validate the random forest model fitRf and get the "Out of Sample Error" as well.


```r
myprediction <- predict(fitRf, newdata=mytest)
confusionMatrix(myprediction, mytest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 837   4   0   0   0
##          B   0 565   5   0   0
##          C   0   0 507   3   1
##          D   0   0   1 479   0
##          E   0   0   0   0 540
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9952         
##                  95% CI : (0.992, 0.9974)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.994          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9930   0.9883   0.9938   0.9982
## Specificity            0.9981   0.9979   0.9984   0.9996   1.0000
## Pos Pred Value         0.9952   0.9912   0.9922   0.9979   1.0000
## Neg Pred Value         1.0000   0.9983   0.9975   0.9988   0.9996
## Prevalence             0.2845   0.1934   0.1744   0.1638   0.1839
## Detection Rate         0.2845   0.1920   0.1723   0.1628   0.1835
## Detection Prevalence   0.2859   0.1937   0.1737   0.1632   0.1835
## Balanced Accuracy      0.9990   0.9954   0.9933   0.9967   0.9991
```

The accuracy for the prediction is 99.52%. The "Out of Sample Error" is 0.48%.

###Predicting classes for the Course Project Prediction Quiz

I'm now predicting the classes for the testing data set with the fitted model fitRf.


```r
prediction <- predict(fitRf, newdata=testing)
prediction
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

