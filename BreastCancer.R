# breast cancer project
options(digits = 3)
library(matrixStats)
library(tidyverse)
library(caret)
library(dslabs)
library(dplyr)
data(brca)

# Dimensions and properties
mean(brca$y == "M")
columnMeans = colMeans(brca$x)
which.max(columnMeans)
columnSDs = apply(brca$x, 2, sd)
which.min(columnSDs)
# OR which.min(colSds(brca$x))

# Scaling
x_scaled = sweep(brca$x, 2, columnMeans, "-")
x_scaled = sweep(x_scaled, 2, columnSDs, "/")
sd(x_scaled[,1])
median(x_scaled[,1])

# Distance
d_samples <- dist(x_scaled)
dist_BtoB <- as.matrix(d_samples)[1, brca$y == "B"]
mean(dist_BtoB[2:length(dist_BtoB)])
dist_BtoM <- as.matrix(d_samples)[1, brca$y == "M"]
mean(dist_BtoM)

# Heat map
d_features <- dist(t(x_scaled))
heatmap(as.matrix(d_features), labRow = NA, labCol = NA)

# Hierarchical Clustering
h <- hclust(d_features)
groups <- cutree(h, k = 5)
split(names(groups), groups)

# SVD
s <- svd(x_scaled)
sum(s$d[1:7]^2) / sum(s$d^2)
# Same analysis using PCA
pca <- prcomp(x_scaled)
summary(pca)     # first value of Cumulative Proportion that exceeds 0.9: PC7

# Plot first two principal components w/ color to represent tumor type
data.frame(pca$x[,1:2], type = brca$y) %>%
  ggplot(aes(PC1, PC2, color = type)) +
  geom_point()

# Box plot of first 10 PCAs
boxplot(pca$x[,10] ~ brca$y, main = paste("PC"))
# another option to show them all at once
data.frame(type = brca$y, pca$x[,1:10]) %>%
  gather(key = "PC", value = "value", -type) %>%
  ggplot(aes(PC, value, fill = type)) +
  geom_boxplot()

# Create train and test sets
set.seed(1, sample.kind = "Rounding")    # if using R 3.6 or later
test_index <- createDataPartition(brca$y, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled[test_index,]
test_y <- brca$y[test_index]
train_x <- x_scaled[-test_index,]
train_y <- brca$y[-test_index]

# K-means Clustering
predict_kmeans <- function(x, k) {
  centers <- k$centers    # extract cluster centers
  # calculate distance to cluster centers
  distances <- sapply(1:nrow(x), function(i){
    apply(centers, 1, function(y) dist(rbind(x[i,], y)))
  })
  max.col(-t(distances))  # select cluster with min distance to center
}
set.seed(3, sample.kind = "Rounding") 
k <- kmeans(train_x, centers = 2)
y_hat_kmeans = predict_kmeans(test_x, k)
test_y_converted = ifelse(test_y_kmeans == "B", 1, 2) 
mean(y_hat == test_y_converted)
# another way of doing the same thing
kmeans_preds <- ifelse(predict_kmeans(test_x, k) == 1, "B", "M")
mean(kmeans_preds == test_y)
# proportion of benign tumors correctly classified
b = sum(test_y == "B" & kmeans_preds == "B")
not_b = sum(test_y == "B" & kmeans_preds != "B")
sensitivity(factor(kmeans_preds), test_y, positive = "B")
sensitivity(factor(kmeans_preds), test_y, positive = "M")

# Logistic Regression
# train glm model based on all predictors
train_glm = train(train_x, train_y, method = "glm")
y_hat_glm = predict(train_glm, test_x)
mean(y_hat_glm == test_y)

# LDA Model
train_lda = train(train_x, train_y, method = "lda")
y_hat_lda = predict(train_lda, test_x)
mean(y_hat_lda == test_y)

# QDA Model
train_qda = train(train_x, train_y, method = "qda")
y_hat_qda = predict(train_qda, test_x)
mean(y_hat_qda == test_y)

# Loess Model
set.seed(5, sample.kind="Rounding")
train_loess = train(train_x, train_y, method = "gamLoess")
y_hat_loess = predict(train_loess, test_x)
mean(y_hat_loess == test_y)

# train kNN model
set.seed(7, sample.kind="Rounding")
tuning <- data.frame(k = seq(3, 21, 2))
train_knn <- train(train_x, train_y,
                   method = "knn", 
                   tuneGrid = tuning)
train_knn$bestTune
knn_preds <- predict(train_knn, test_x)
mean(knn_preds == test_y)

# train a random forest model
set.seed(9, sample.kind = "Rounding") 
train_rf = train(train_x, train_y, ntree = 100, method = "rf", importance=TRUE, tuneGrid = data.frame(mtry = seq(3, 9, 2)))
train_rf$bestTune
y_hat_rf = predict(train_rf, test_x)
mean(y_hat_rf == test_y)
varImp(train_rf$finalModel)

# create ensemble prediction and compute accuracy
predictions = c(y_hat_kmeans, y_hat_glm, y_hat_lda, y_hat_qda, y_hat_loess, knn_preds, y_hat_rf)
votes <- rowMeans(as.matrix(predictions) == 2)
y_hat <- ifelse(votes > 0.5, "M", "B")
mean(y_hat == test_y)

# another attempt
ensemble <- cbind(glm = y_hat_glm == "B",
                  lda = y_hat_lda == "B",
                  qda = y_hat_qda == "B",
                  loess = y_hat_loess == "B",
                  rf = y_hat_rf == "B",
                  knn = knn_preds == "B",
                  kmeans = y_hat_kmeans == "B")

ensemble_preds <- ifelse(rowMeans(ensemble) > 0.5, "B", "M")
mean(ensemble_preds == test_y)

models <- c("K means", "Logistic regression", "LDA", "QDA", "Loess", "K nearest neighbors", "Random forest", "Ensemble")
accuracy <- c(mean(y_hat_kmeans == test_y),
              mean(y_hat_glm == test_y),
              mean(y_hat_lda == test_y),
              mean(y_hat_qda == test_y),
              mean(y_hat_loess == test_y),
              mean(knn_preds == test_y),
              mean(y_hat_rf == test_y),
              mean(ensemble_preds == test_y))
data.frame(Model = models, Accuracy = accuracy)









