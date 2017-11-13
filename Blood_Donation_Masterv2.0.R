###############################################################################
#       PREDICTIVE ANALYTICS OF BLOOD DONATION COMPAIGN

# Problem description:
# https://www.drivendata.org/competitions/2/warm-up-predict-blood-donations/page/7/

# About the Dataset:
# Months since Last Donation: this is the number of monthis since this donor's most recent donation.
# Number of Donations: this is the total number of donations that the donor has made.
# Total Volume Donated: this is the total amount of blood that the donor has donated in cubic centimeters.
# Months since First Donation: this is the number of months since the donor's first donation.
# Made Donation in March 2007: Binary value indicating whether or not donor has donated blood
#                              in March 2007. 0 indicates NO and 1 indicates YES.
# Goal:
# To determine whether or not the donor has donated blood in March 2007
###############################################################################


###############################################################################
#                       REGRESSION/CLASSIFIER MODEL TO BE USED

## Response variable in this problem is a categorial variable which takes either
## yes or no which can be coded as 1 or 0. In this model, to predict a categorical 
## response variable with binary values, logistic Regression approach is used.
## Logistic Regression can be interpreted as both regression and classifier algorithm.
###############################################################################


###############################################################################
#                       APPROACH

## step 1: importing required libraries
## step 2: loading the training and testing dataset
#### step 2.1: Renaming the names of training and testing dataframe
#### step 2.2: Attaching training dataset for analysis
## step 3: correlation and scatter plots of predictor variables with response variable
#### step 3.1: INTERPRETATION of step 3
## step 4: Generating logistic Regression models
## step 5: Selection of best model
#### step 5.1: creating a R list of models
#### step 5.2: comparing models using ANOVA
#### step 5.3: comparing models using likelihood ratio test
#### step 5.4: checking for Multicollinearity in the models using VIF
#### step 5.5: Creating dataframe to store Null deviance, Residual deviance, AIC, McFadden R2
#### step 5.5: INTERPRETATION
## Step 6: standarized coefficients of selected Model
## step 7: Goodness of Fit 
#### step 7.1: Accuracy on Training data
###### step7.1.1: Predicting values for training dataset
###### step7.1.2: ConfusionMatrix with threshold of 0.5
###### step7.1.3: ConfusionMatrix with threshold of 0.3
#### step 7.2: Receiving Operating Characterics(ROC) 
###### step7.2.1: ROC Curve
###### step7.2.2: cutoff for max accuracy
###### step7.2.3: TPR and FPR Evaluation
###### step7.2.4: INTERPRETATION
#### step 7.3: Area Under Curve(AUC)
## step 8: Predicting values for testing dataset
## step 9: Detaching Training data
## step 10: Overall Interpretation

###############################################################################

## step 1: importing required libraries
library(lmtest)
library(lm.beta)
library(caTools)
library(car)
library(pscl)
library(ROCR)


## step 2: loading the training and testing dataset
blood_train = read.csv(file.choose(),header=TRUE)
blood_test <- read.csv(file.choose(),header=TRUE)


#### step 2.1: Renaming the names of training and testing dataframe
# Let predictor variables be represented by x's and be response variable be y
# x1 = Months.since.Last.Donation
# x2 = Number.of.Donations
# x3 = Total.Volume.Donated..c.c..
# x4 = Months.since.First.Donation
# y = Made.Donation.in.March.2007
names(blood_train) <- c('ID','x1','x2','x3','x4','y')
names(blood_test) <- c('ID','x1','x2','x3','x4')
str(blood_train)
str(blood_test)


#### step 2.2: Attaching training dataset for analysis
attach(blood_train)


## step 3: correlation and scatter plots of predictor variables with response variable
blood_cor_matrix = cor(blood_train[,-1])
print (blood_cor_matrix)
par(mfrow = c(2,2))
plot(y~x1+x2+x3+x4,col='blue')
plot(y~jitter(x1)+jitter(x2)+jitter(x3)+jitter(x4),col='blue')


#### step 3.1: INTERPRETATION of step 3
# None of the predictor varibles has significant correlation with response variable. 
# However, we will create the model by including all the predictor variables. Based on the 
# performance of the model, we will exclude the insignificant variables


## step 4: Generating logistic Regression models


# Model 1:
## Response Variable: 'Made Donation in March 2007'
## Predictor Variables: 'Months since Last Donation','Number of Donations',
##                      'Total Volume Donated',Months since First Donation
blood_train_model1 <- glm(y~x1+x2+x3+x4,data = blood_train, family='binomial')
summary(blood_train_model1)
## INTERPRETATION: Total volume donated(x3) and number of donations(x2) is linearly related.
## so both at once shouldn't be used in the model


# Model 2:
## Response Variable: 'Made Donation in March 2007'
## Predictor Variables: 'Months since Last Donation','Number of Donations',
##                      Months since First Donation
blood_train_model2 <- glm(y~x1+x2+x4,data=blood_train,family='binomial')
summary(blood_train_model2)


# Model 3:
## Response Variable: 'Made Donation in March 2007'
## Predictor Variables: 'Months since Last Donation',
##                       'Total Volume Donated', Months since First Donation
blood_train_model3 <- glm(y~x1+x3+x4,data=blood_train,family='binomial')
summary(blood_train_model3)


# Model 4:
## Response Variable: 'Made Donation in March 2007'
## Predictor Variables: 'Months since Last Donation','Number of Donations'
blood_train_model4 <- glm(y~x1+x2,data=blood_train,family='binomial')
summary(blood_train_model4)


# Model 5
## Response Variable: 'Made Donation in March 2007'
## Predictor Variables: 'Total Volume Donated','Months since First Donation'
blood_train_model5 <- glm(y~x3+x4,data=blood_train,family='binomial')
summary(blood_train_model5)



## step 5: Selection of best model using ANOVA, Likelihood ration test, AIC
##         Pseudo-Rsquared value(McFadden R2) and Variance Inflation Factor(VIF)


#### step 5.1: creating a R list of models
model_list = list(model1=blood_train_model1,
                  model2=blood_train_model2,
                  model3=blood_train_model3,
                  model4=blood_train_model4,
                  model5=blood_train_model5)


#### step 5.2: comparing models using ANOVA
anova(blood_train_model1,blood_train_model2)
anova(blood_train_model2,blood_train_model3)
anova(blood_train_model2,blood_train_model4)
anova(blood_train_model2,blood_train_model5)


#### step 5.3: comparing models using likelihood ratio test
lrtest(blood_train_model1,blood_train_model2)
lrtest(blood_train_model2,blood_train_model3)
lrtest(blood_train_model2,blood_train_model4)
lrtest(blood_train_model2,blood_train_model5)


#### step 5.4: checking for Multicollinearity in the models using VIF
# exclude the first model as it has two predictor variables with linear dependency
j=2
for (i in model_list[-1]) {
  print(names(model_list[j]))
  j=j+1
  print(vif(i))
}


#### step 5.5: Creating dataframe to store Null deviance, Residual deviance, AIC, McFadden R2
model_names <- names(model_list)
aic = c()
null_deviance = c()
res_deviance = c()
McFadden_R2 = c()
formula=c()
for (i in model_list){
  aic=c(aic,i$aic)
  null_deviance=c(null_deviance,i$null.deviance)
  res_deviance=c(res_deviance,i$deviance)
  McFadden_R2=c(McFadden_R2,pR2(i)['McFadden'])
  formula=c(formula,i$formula)
}
model_parameters = data.frame(model_names=model_names,
                         aic=round(aic,4),
                         null_deviance=round(null_deviance,4),
                         deviance=round(res_deviance,4),
                         PseudoR2=round(McFadden_R2,4),
                         formula=as.character(formula))


#### step 5.5: INTERPRETATION
## Model 1 has predictor variables(x2 and X3) that are linearly 
## dependent on each other. Hence this model cannot be used. Model2 and Model3 are
## identical. Comparing values with Model4 and Model5, Model2 performs better
## and can be selected as good fit for this dataset.


## Step 6: standarized coefficients of selected Model
lm.beta(blood_train_model2)


## step 7: Goodness of Fit 


#### step 7.1: Accuracy on Training data


###### step7.1.1: Predicting values for training dataset
blood_pred <- predict(blood_train_model2,blood_train,type = 'response')


###### step7.1.2: ConfusionMatrix with threshold of 0.5
blood_cm_50 <- table(actual_value = blood_train$y,predicted_value = blood_pred>0.5)
print (blood_cm_50)
accuracy_cm_50 <- sum(diag(blood_cm_50))/sum(blood_cm_50)
paste('Accuracy = ',round(accuracy_cm_50,2))
paste('Misclassification Rate for 0.5 threshold = ',round(1-accuracy_cm_50,2))


###### step7.1.3: ConfusionMatrix with threshold of 0.3
blood_cm_30 <- table(actual_value = blood_train$y,predicted_value = blood_pred>0.3)
print(blood_cm_30)
accuracy_cm_30 <- sum(diag(blood_cm_30))/sum(blood_cm_30)
paste('Accuracy = ',round(accuracy_cm_30,2))
paste('Misclassification Rate for 0.3 threshold = ',round(1-accuracy_cm_30,2))


#### step 7.2: Receiving Operating Characterics(ROC)


###### step7.2.1: ROC Curve
ROCPred <- prediction(blood_pred,blood_train$y)
ROCPerf <- performance(ROCPred,'acc')
plot(ROCPerf,main = 'Cutoff Vs Accuracy',col='blue',lwd = 1.5)


###### step7.2.2: cutoff for max accuracy
max_accuracy <- max(slot(ROCPerf,'y.values')[[1]])
max_accuracy_ind <- which.max(slot(ROCPerf,'y.values')[[1]])
opt_cutoff <- slot(ROCPerf,'x.values')[[1]][max_accuracy_ind]
abline(h=max_accuracy,v=opt_cutoff,col=c('Red','darkgreen'),lwd=2)
legend(x=0.6,y=0.5,legend = c('Max Accuracy','Optimum Cutoff'),col=c('Red','darkgreen'),lwd=2)
paste('Max Accuracy of ',round(max_accuracy,4),' is obtained at cutoff ',round(opt_cutoff,4))


###### step7.1.3: TPR and FPR Evaluation
ROCeval <- performance(ROCPred,'tpr','fpr')
plot(ROCeval,colorize = TRUE,lwd = 1.5,print.cutoffs.at = seq(0.1,by=0.1),
     main='TPR vs FPR')


###### step7.2.4: INTERPRETATION
## It can be inferred that at cutoff = 0.3, TPR increases steeply
## and FPR (Type I error) is minimum


#### step 7.3: Area Under Curve(AUC)
ROCauc <- performance(ROCPred,'auc')
paste(slot(ROCauc,'y.name')[1],' is ',round(slot(ROCauc,'y.values')[[1]],4))


## step 8: Predicting values for testing dataset
blood_pred_test <- predict(blood_train_model2,blood_test,type='response')
blood_pred_test_df <- data.frame(ID = blood_test$ID,
                                 'Made Donation in March 2007'=round(blood_pred_test,4))
write.csv(x=blood_pred_test_df,file = 'C:/Users/vino/Desktop/Blood Donation/Testing_prediction.csv',row.names = FALSE)

## step 9: Detaching Training data
detach(blood_train)


## step 10: Overall Interpretation
# it is recommended to choose threshold as 0.29 which will avoid 
# false negatives(Type II error) and also give comparitively good accuracy.
# Positive prediction rate(PPR) and Negative prediction Rate(NPR) are almost equal

