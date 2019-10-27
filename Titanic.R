library(titanic)    # loads titanic_train data frame
library(caret)
library(tidyverse)
library(rpart)

# 3 significant digits
options(digits = 3)

# clean the data - `titanic_train` is loaded with the titanic package
titanic_clean <- titanic_train %>%
  mutate(Survived = factor(Survived),
         Embarked = factor(Embarked),
         Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age), # NA age to median age
         FamilySize = SibSp + Parch + 1) %>%    # count family members
  select(Survived,  Sex, Pclass, Age, Fare, SibSp, Parch, FamilySize, Embarked)

# generate training and test sets
set.seed(42, sample.kind="Rounding")
test_index <- createDataPartition(titanic_clean$Survived, times = 1, p = 0.2, list = FALSE)
test_set <- titanic_clean[test_index, ]
train_set <- titanic_clean[-test_index, ]

# what proporiton survived from the training set?
train_survived = length(which (train_set$Survived == 1))
train_survived/count(train_set)

# guessing the outcome and compute the accuracy
set.seed(3, sample.kind="Rounding")
y_hat <- sample(c(0, 1), length(test_index), replace = TRUE)
mean(y_hat == test_set$Survived)

# predicting survival by sex
train_set_female = train_set %>% filter(train_set$Sex == "female")
mean(train_set_female$Survived == 1)
train_set_male = train_set %>% filter(train_set$Sex == "male")
mean(train_set_male$Survived == 1)
y_hat = ifelse(test_set$Sex == "female", 1, 0) %>% factor(levels = levels(test_set$Survived))
#mean(y_hat == test_set$Survived)
#confusionMatrix(data = y_hat, reference = test_set$Survived)
#F_meas(data = y_hat, reference = factor(test_set$Survived))

# predicting survival by passenger class
#train_set %>% group_by(Pclass) %>% summarize(Survived = mean(Survived == 1))
y_hat = ifelse(test_set$Pclass == 1, 1, 0) %>% factor(levels = levels(test_set$Survived))
#mean(y_hat == test_set$Survived)
#confusionMatrix(data = y_hat, reference = test_set$Survived)
#F_meas(data = y_hat, reference = factor(test_set$Survived))

# sex and class model
#train_set %>% group_by(Sex, Pclass) %>%summarize(Survived = mean(Survived == 1))
y_hat = ifelse(test_set$Sex == "male" | test_set$Pclass == 3, 0, 1) %>% factor(levels = levels(test_set$Survived))
#mean(y_hat == test_set$Survived) 
#confusionMatrix(data = y_hat, reference = test_set$Survived)
#F_meas(data = y_hat, reference = factor(test_set$Survived))

# train lda model based on fare
train_lda = train(Survived ~ Fare, method = "lda", data = train_set)
y_hat = predict(train_lda, test_set)
confusionMatrix(data = y_hat, reference = test_set$Survived)$overall["Accuracy"]

# train qda model based on fare
train_qda = train(Survived ~ Fare, method = "qda", data = train_set)
y_hat = predict(train_qda, test_set)
mean(y_hat == test_set$Survived)

# train glm model based on age
train_glm = train(Survived ~ Age, method = "glm", data = train_set)
y_hat = predict(train_glm, test_set)
mean(y_hat == test_set$Survived)

# train glm model based on sex, class, fare, and age
train_glm = train(Survived ~ Sex + Pclass + Fare + Age, method = "glm", data = train_set)
y_hat = predict(train_glm, test_set)
mean(y_hat == test_set$Survived)

# train glm model based on sex, class, fare, and age
train_glm = train(Survived ~., method = "glm", data = train_set)
y_hat = predict(train_glm, test_set)
mean(y_hat == test_set$Survived)

# train kNN model
set.seed(6, sample.kind="Rounding")
train_knn = train(Survived~., method = "knn", data=train_set, tuneGrid = data.frame(k=seq(3, 51, 2)))
#train_knn$bestTune
#plot(train_knn)
y_hat = predict(train_knn, test_set)
mean(y_hat == test_set$Survived)

# train kNN model w/ cross validation
set.seed(8, sample.kind="Rounding")
control <- trainControl(method = "cv", number = 10, p = .1)
train_knn_cv <- train(Survived ~ ., method = "knn", 
                      data = train_set,
                      tuneGrid = data.frame(k = seq(3, 51, 2)),
                      trControl = control)
y_hat = predict(train_knn_cv, test_set)
mean(y_hat == test_set$Survived)

# train a classification tree
set.seed(10, sample.kind="Rounding")
train_rpart = train(Survived ~., data = train_set, method = "rpart", tuneGrid = data.frame(cp = seq(0, 0.05, 0.002)))
y_hat = predict(train_rpart, test_set)
mean(y_hat == test_set$Survived)
text(train_rpart$finalModel)

# train a random forest model
train_rf = train(Survived ~., data = train_set, ntree = 100, method = "rf", tuneGrid = data.frame(mtry = seq(1, 7, 1)))
y_hat = predict(train_rf, test_set)
mean(y_hat == test_set$Survived)
varImp(train_rf$finalModel)

