################################
# Download and Clean Data.
# Create edx, validation, test and train sets.
################################

# Note: this process could take a couple of minutes
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# 'Indian Liver Patient' dataset:
# https://www.kaggle.com/uciml/indian-liver-patient-records
# Read the .csv dataset as a data frame.
# We use the 'stringsAsFactors' argument to ensure our string are not converted into factors.
# Note: The 'indian_liver_patient.csv' file is included in the project.
liverCancer <- read.csv(file = './indian_liver_patient.csv', stringsAsFactors = FALSE)

# The type is a data.frame, and our columns have int, num, and char types
str(liverCancer)

# Split our data into a matrix of predictors 'X' and a vector of labels 'Y'.
liverCancer_x = liverCancer[,1:10]
liverCancer_y = factor(liverCancer[,11:11])

# Convert all columns of 'X' to numeric types
liverCancer_x$Gender <- ifelse(liverCancer_x$Gender == 'Male', 1, 2)
liverCancer_x <- transform(liverCancer_x,
                         Age = as.numeric(Age), 
                         Alkaline_Phosphotase = as.numeric(Alkaline_Phosphotase),
                         Alamine_Aminotransferase = as.numeric(Alamine_Aminotransferase),
                         Aspartate_Aminotransferase = as.numeric(Aspartate_Aminotransferase))

# Convert all NAs to the column mean
for(i in 1:ncol(liverCancer_x)){
  liverCancer_x[is.na(liverCancer_x[,i]), i] <- mean(liverCancer_x[,i], na.rm = TRUE)
}

# Scale the columns
columnMeans = colMeans(liverCancer_x)
columnSDs = apply(liverCancer_x, 2, sd)
x_scaled = sweep(liverCancer_x, 2, columnMeans, "-")
x_scaled = sweep(liverCancer_x, 2, columnSDs, "/")

# Validation set will be 10% of 'Indian Liver Patient' data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(liverCancer_y, times = 1, p = 0.1, list = FALSE)
edx_x <- liverCancer_x[-test_index,]
edx_y <- liverCancer_y[-test_index]
validation_x <- liverCancer_x[test_index,]
validation_y <- liverCancer_y[test_index]

# Split the edx data into train/test sets.
# We will use these to do the intermediate accuracy comparisons.
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx_y, times = 1, p = 0.1, list = FALSE)
train_x <- edx_x[-test_index,]
train_y <- edx_y[-test_index]
test_x <- edx_x[test_index,]
test_y <- edx_y[test_index]

################################
# Explore the Data
################################

# Let's take some time to explore our data.
# How many samples and predictors are there?
numSamples = dim(liverCancer)[1]
numSamples
numPredictors = dim(liverCancer)[2]
numPredictors
# We have only 583 samples to work with. Note this is not sufficient to train a highly accurate model.
# We have 11 predictors, which seems like a lot. These may not all be useful in our predictions.

# What proportion of our samples are patients with liver cancer?
mean(liverCancer_y == 1)
# Notice we have a lot of samples with liver cancer. In an ideal world our samples would be more balanced.

# Which predictor has the highest and lowest standard deviation?
which.max(columnSDs)
which.min(columnSDs)
# The predictor with the highest sd may prove to be more useful in differentiating the patients with and without cancer.

# The average distance between the first sample, which is a patient with liver cancer, and other samples where the patient has cancer
d_samples <- dist(x_scaled)
dist_Cancer <- as.matrix(d_samples)[1, liverCancer_y == 1]
mean(dist_Cancer[2:length(dist_Cancer)])

# heatmap
d_features <- dist(t(x_scaled))
heatmap(as.matrix(d_features), labRow = NA, labCol = NA)
# hierarchical clustering
h <- hclust(d_features)
groups <- cutree(h, k = 5)
split(names(groups), groups)
# A heatmap and hierarchical clustering of the features show there are some relationships among different features.

# Perform a principal component analysis
pca <- prcomp(x_scaled)
summary(pca)
# We can see the proportion of variance explained by the first principal component analysis is only .278.
# We need at least 7 principal components to explain at least 90% of the variance.
# Based on this, we will leverage all features to build our model predictions.
# While a few may not be as useful, most will be.

# The average distance between the first sample and other samples where the patient does not have cancer
dist_Not_Cancer <- as.matrix(d_samples)[1, liverCancer_y == 2]
mean(dist_Not_Cancer)
# Strangely, we see that the patient with cancer is actually closer in distance to patient without cancer.
# This may indicate our features are not going to be very helpful predictors, but we will still try the different machine learning models and see.

################################
# Model Evalutations
################################

# Logistic Regression
train_glm = train(train_x, train_y, method = "glm")
y_hat_glm = predict(train_glm, test_x)
mean(y_hat_glm == test_y)

# LDA
train_lda = train(train_x, train_y, method = "lda")
y_hat_lda = predict(train_lda, test_x)
mean(y_hat_lda == test_y)

# QDA
train_qda = train(train_x, train_y, method = "qda")
y_hat_qda = predict(train_qda, test_x)
mean(y_hat_qda == test_y)

# Loess
set.seed(5, sample.kind="Rounding")
train_loess = train(train_x, train_y, method = "gamLoess")
y_hat_loess = predict(train_loess, test_x)
mean(y_hat_loess == test_y)

# kNN
set.seed(7, sample.kind="Rounding")
tuning <- data.frame(k = seq(3, 21, 2))
train_knn <- train(train_x, train_y,
                   method = "knn", 
                   tuneGrid = tuning)
train_knn$bestTune
y_knn <- predict(train_knn, test_x)
mean(y_knn == test_y)

# Random Forest
set.seed(9, sample.kind = "Rounding") 
train_rf = train(train_x, train_y, ntree = 100, method = "rf", importance=TRUE, tuneGrid = data.frame(mtry = seq(3, 9, 2)))
train_rf$bestTune
y_hat_rf = predict(train_rf, test_x)
mean(y_hat_rf == test_y)
varImp(train_rf$finalModel)
# You can see the most important variable is 'Aspartate_Aminotransferase'
# This matches our earlier data analysis where we speculated that the column with the highest SD may be the most useful factor.

# Ensemble of the previous models
ensemble <- cbind(glm = y_hat_glm == 1,
                  lda = y_hat_lda == 1,
                  qda = y_hat_qda == 1,
                  loess = y_hat_loess == 1,
                  rf = y_hat_rf == 1,
                  knn = y_knn == 1)

y_hat_ensemble <- ifelse(rowMeans(ensemble) > 0.5, 1, 2)
mean(y_hat_ensemble == test_y)

################################
# Final Model Evaluation
################################

# This chart shows the accuracy of different models
models <- c("Logistic regression", "LDA", "QDA", "Loess", "K nearest neighbors", "Random forest", "Ensemble")
accuracy <- c(mean(y_hat_glm == test_y),
              mean(y_hat_lda == test_y),
              mean(y_hat_qda == test_y),
              mean(y_hat_loess == test_y),
              mean(y_knn == test_y),
              mean(y_hat_rf == test_y),
              mean(y_hat_ensemble == test_y))
data.frame(Model = models, Accuracy = accuracy)
# You can see the loess method is the most accurate. We'll choose this method as our final selection and use the validation set to calculate accuracy.

# Let's use our validation set to calculate accuracy of our chosen model, loess
# Loess with Validation Set
y_hat_final = predict(train_loess, validation_x)
mean(y_hat_final == validation_y)
# The accuracy of the final model is .729

# Let's look at the proportion of patients with liver cancer correctly classified
sum(validation_y == 1 & y_hat_final == 1)/sum(validation_y == 1)
# We were able to correctly identify 90% of the patients with liver cancer
# Let's see the proportion of patients without liver cancer that were predicted as having cancer
sum(validation_y == 2 & y_hat_final == 1)/sum(validation_y == 2)
# We can see we've incorrectly classified 70% of patients without cancer as having cancer.
# This is a disappointing result, though it can in part be explained by the various limitations of this dataset.
# recall that our data was unbalanced comprising of 71% patients with cancer, the dataset contained only 583 samples, and the analysis on the distance between patients indicated the predictors may not be particluarly good.