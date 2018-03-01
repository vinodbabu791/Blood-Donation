library(lmtest)
library(lm.beta)
library(caTools)
library(car)
library(pscl)
library(ROCR)

blood_donation = read.csv(file.choose(),header=TRUE)
str(blood_donation)

# Let predictor variables be represented by x's and be response variable be y
# x1 = Months.since.Last.Donation
# x2 = Number.of.Donations
# x3 = Total.Volume.Donated..c.c..
# x4 = Months.since.First.Donation
# y = Made.Donation.in.March.2007

names(blood_donation) <- c('ID','x1','x2','x3','x4','y')
attach(blood_donation)
# correlation and scatter plots of predictor variables with response variable

cor(blood_donation[,-1])
par(mfrow = c(2,2))
plot(y~x1+x2+x3+x4,col='blue')
plot(y~jitter(x1)+jitter(x2)+jitter(x3)+jitter(x4),col='blue')

# None of the predictor varibles has significant correlation with response variable. 
# However, we will create the model by including all the predictor variables. Based on the 
# performance of the model, we will exclude the insignificant variable

# split the dataset to training and testing 
blood_donation_split <- sample.split(blood_donation,SplitRatio = 0.85)
blood_train <- blood_donation[blood_donation_split,]
blood_test <- blood_donation[!blood_donation_split,]

detach(blood_donation)
attach(blood_train)


blood_donation_model1 <- glm(y~x1+x2+x3+x4,data = blood_train, family='binomial')
summary(blood_donation_model1)
# Total volume donated(x3) and number of donations(x2) is linearly related. so both at once shouldn't be used in the model
# Hence let's use only one

blood_donation_model2 <- glm(y~x1+x2+x4,data=blood_train,family='binomial')
summary(blood_donation_model2)
vif(blood_donation_model2)

blood_donation_model3 <- glm(y~x1+x3+x4,data=blood_train,family='binomial')
summary(blood_donation_model3)
vif(blood_donation_model3)

blood_donation_model4 <- glm(y~x1+x2,data=blood_train,family='binomial')
summary(blood_donation_model4)
vif(blood_donation_model4)

blood_donation_model5 <- glm(y~x1+x3,data=blood_train,family='binomial')
summary(blood_donation_model5)
vif(blood_donation_model5)


# comparing models using ANOVA

anova(blood_donation_model1,blood_donation_model2)
anova(blood_donation_model2,blood_donation_model3)
anova(blood_donation_model2,blood_donation_model4)
anova(blood_donation_model2,blood_donation_model5)

# comparing models using likelihood ratio test

lrtest(blood_donation_model1,blood_donation_model2)
lrtest(blood_donation_model2,blood_donation_model3)
lrtest(blood_donation_model2,blood_donation_model4)
lrtest(blood_donation_model2,blood_donation_model5)


# based on Residual deviance, AIC values,ANOVA and log likelihood tests, model2 performs better

# pseudo R2 of the model
pR2(blood_donation_model2)

# McFadden R2 value is 0.11079 which indicates accuracy of 11.08%

# Predicting values for testing dataset

blood_donation_pred <- predict(blood_donation_model2,blood_test,type = 'response')

# confusionMatrix with threshold of 0.5
blood_cm_50 <- table(actual_value = blood_test$y,predicted_value = blood_donation_pred>0.5)
#accuracy for 0.5 threshold
accuracy_cm_50 <- sum(diag(blood_cm_50))/sum(blood_cm_50)
print(accuracy_cm_50)
paste('Misclassification Rate for 0.5 threshold = ',round(1-accuracy_cm_50,2))

blood_cm_30 <- table(actual_value = blood_test$y,predicted_value = blood_donation_pred>0.3)
accuracy_cm_30 <- sum(diag(blood_cm_30))/sum(blood_cm_30)
print(accuracy_cm_30)
paste('Misclassification Rate for 0.3 threshold = ',round(1-accuracy_cm_30,2))

# Receiving Operation Characteristics(ROC) curve and Area under curve(AUC) : to find
# correct threshold and evaluate the accuracy of the model
blood_donation_train_pred <- predict(blood_donation_model2,blood_train,type='response')
ROCPred <- prediction(blood_donation_train_pred,blood_train$y)
ROCPerf_acc <- performance(ROCPred,'acc')
# finding cutoff for maximum accuracy
max_loc <- which.max(slot(ROCPerf_acc,'y.values')[[1]])
max_acc <- slot(ROCPerf_acc,'y.values')[[1]][max_loc]
cutoff <- slot(ROCPerf_acc,'x.values')[[1]][max_loc]
paste('cutoff = ',cutoff)
paste('Max Accuracy =',max_acc)
plot(ROCPerf_acc,main = 'Accuracy Vs cutoff',col = 'red',lwd = 1.5)
abline(v=cutoff,h=max_acc,col = c('blue','darkgreen'),lwd=2)

# ROC curve
ROCPerf <- performance(ROCPred,'tpr','fpr')
plot(ROCPerf,colorize = TRUE,main = 'ROC curve',print.cutoffs.at = seq(0.1,by=0.1))
abline(a=0,b=1)
auc <- performance(ROCPred,'auc')
paste('AUC = ',slot(auc,'y.values')[[1]][1])


# it is recommended to choose threshold as 0.3 which will avoid true negatives and also give comparitively good accuracy.
# i.e) identifying people as not donated blood when they actually did

# accuracy of the model is 0.791667.
