setwd("D:\\OneDrive - IESEG\\Working\\03. MBD_SML_Statistical & ML Approaches for Mkt\\SML_Section7\\data\\com1_default")

library(neuralnet)
library(nnet)

# Read data
df <- read.csv('default.csv', sep=';')
str(df)

# Encode as a one hot vector multilabel data
df2 <- cbind(df[, 2:(ncol(df)-1)], class.ind(as.factor(df$Y)))
names(df2) <- c(names(df2)[1:(ncol(df2)-2)], "N", "Y")

# Take 10%
set.seed(1)
df2 <- df2[sample(1:nrow(df2), round(nrow(df2)*0.05)), ]

# Train/test
set.seed(1)
train_idx <- sample(1:nrow(df2), round(nrow(df2)*0.8))
train_df2 <- df2[train_idx, ]
test_df2 <- df2[-train_idx, ]

# Create the formula
nn_formula <- as.formula(paste0('Y + N ~ ', paste(names(df2)[1:(ncol(df2)-2)], collapse=' + ')))
nn_formula

# Fit the Neural Network model
md_nnet <- neuralnet(nn_formula,
                     train_df2,
                     hidden=c(30),        # Size of the hidden layers
                     lifesign='full',       # Print during train
                     algorithm='backprop',  # Algorithm to calculate the network
                     learningrate=0.01,     # Learning rate                      
                     err.fct='ce',          # Error function, cross-entropy
                     act.fct="logistic",    # Function use to calculate the result
                     linear.output=F
)

#deep neural network with multiple layers
dd_nnet <- neuralnet(nn_formula,
                     train_df2,
                     hidden=c(30,30,30),        # Size of the hidden layers
                     lifesign='full',       # Print during train
                     algorithm='backprop',  # Algorithm to calculate the network
                     learningrate=0.01,     # Learning rate                      
                     err.fct='ce',          # Error function, cross-entropy
                     act.fct="logistic",    # Function use to calculate the result
                     linear.output=F
)



logit <- glm(nn_formula, data = train_df2, family = 'binomial')
summary(logit)

#comparing with logistic regressor
library('tidyr')
probabilities <- logit %>% predict(test_df2, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "1", "0")
# Model accuracy
mean(predicted.classes == test_df2$N)

predicted.classes

library('randomForest')

#with random forest
rf_classifier = randomForest(nn_formula, data=as.matrix(train_df2), ntree=100, mtry=2, importance=TRUE)
prediction_for_table <- predict(rf_classifier,test_df2[,1:23])
table(observed=test_df2[,25],predicted=prediction_for_table)


#with SVM
library(e1071)
t_scale <- scale(train_df2)
classifier = svm(nn_formula,
                 data = train_df2,
                 kernel = 'linear')
y_pred = predict(classifier, newdata = test_df2[,1:23])

#with decision tree
library('rpart')
tree = rpart(nn_formula, data=train_df2)
tree.pred = predict(tree, newdata=test_df2[1:23])
table(predicted = tree.pred, observed = test_df2[,25])


#with k-nn
library(class)
data_target_category <- train_df2[,25]
pr <- knn(train_df2,test_df2,cl=data_target_category,k=13)
test_target_category <- test_df2[,25]
tab <- table(pr, test_target_category)
tab

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)

#SO WE CAN SEE THAT KNN IS THE BEST ALGORITHM FOR THIS PROBLEM, AS ARTIFICIAL NEURAL NETWORKS WERE NOT EVEN REACHING CONVERGENCE