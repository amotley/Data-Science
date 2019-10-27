################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

################################
# Prepare data for intermediary model evaluation
################################

# Split the edx data into train/test sets.
# We will use these to do the intermediate RMSE comparisons.
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-test_index,]
temp <- edx[test_index,]
test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")
removed <- anti_join(temp, test)
train <- rbind(train, removed)
rm(test_index, temp, removed)

# Define the RMSE method we will use to evaluate our models.
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# We will explore the properties of various predictors, then decide how to use them to create our prediction model
# Potential preditors are: Movie, User, Genre, and Timestamp.

################################
# Movie
################################

# To see if the movie may be an interesting predictor for our model, let's plot the average movie ratings
average_movie_rating <- edx %>% 
  group_by(movieId) %>% 
  summarize(rating=mean(rating)) %>%
  ggplot(aes(rating)) +
  geom_histogram(bins=30, color="black") +
  ggtitle("Average Movie Rating")
# Since the results have some variety, it seems like movies will be a useful predictor to add to our model.

# Our model in this case would be Y_u,i = mu + b_i + epsilon_u,i.
# mu is the true rating for all movies, which we can estimate using the mean
# b_i is the movie effect, or the average ranking for movie i.
# epsilon represents the error
# Since running the lm function will be too slow, we can instead estimate b_i
# Our b_i is estimated by Y_u,i - mu for each movie i.
mu <- mean(train$rating) 
b_i_estimates <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Let's create the model using the b_i_estimates
movie_effect_model <- mu + test %>% 
  left_join(b_i_estimates, by='movieId') %>%
  pull(b_i)
# Find the RMSE of the model
rmse_movie <- RMSE(movie_effect_model, test$rating)
# RMSE of this model is 0.9429615

# Let's explore what our model is estimating and see if we can do better
movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()
top_10_movies <- b_i_estimates %>% 
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10) %>%
  pull(title)
top_10_movies
bottom_10_movies <- b_i_estimates %>% 
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  slice(1:10) %>%
  pull(title)
bottom_10_movies
# We notice our model's top 10 best and worst rated movies are not very well-known
# Let's see how many ratings each of these movies have
top_10_movie_count_rated <- train %>%
  count(movieId) %>% 
  left_join(b_i_estimates, by="movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10) %>% 
  pull(n)
top_10_movie_count_rated
bottom_10_movie_count_rated <- train %>%
  count(movieId) %>% 
  left_join(b_i_estimates, by="movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  slice(1:10) %>% 
  pull(n)
bottom_10_movie_count_rated
# It looks like most of the movies in the top and bottom 10 don't have a lot of users who rated them

# In general, we can also see that some movies are rated more than others
number_of_ratings_per_movie <- edx %>% 
  group_by(movieId) %>% 
  summarize(numRatings=n()) %>%
  ggplot(aes(numRatings)) +
  geom_histogram(bins=30, color="black") +
  ggtitle("Number of Ratings per Movie")
# Since we know predictions made with a low number of data points tend to be less accurate, we should give these estimates less weight
# To reduce the effect of these predictions made with small data points, we can use regularization
# We will need to add a penalty lambda to our estimate of b_i:
# b_i_regularized = sum(ratings - mu)/(lambda + count_ratings)

# To choose the right value of lambda we'll need to use cross valdation
lambdas <- seq(0, 10, 0.25)
rmses_for_different_lambdas <- sapply(lambdas, function(l){
  mu <- mean(train$rating)
  b_i <- train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating-mu)/(n()+l))
  movie_effect_model_reg <- test %>% 
    left_join(b_i, by='movieId') %>% 
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(movie_effect_model_reg, test$rating))
})
qplot(lambdas, rmses_for_different_lambdas)  
best_lambda <- lambdas[which.min(rmses_for_different_lambdas)]

# Let's create the model using the regularized b_i_estimates and the best lambda (1.5)
b_i_estimates_reg <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+best_lambda), n_i = n())
movie_effect_model_reg <- test %>% 
  left_join(b_i_estimates_reg, by='movieId') %>% 
  mutate(pred = mu + b_i) %>%
  pull(pred)

# Find the RMSE of the model
rmse_movie_reg <- RMSE(movie_effect_model_reg, test$rating)
# RMSE of this model is 0.942937, which is slightly better than before

# Let's see if our top 10 and bottom 10 movies makes more sense after regularization
top_10_movies_reg <- train %>%
  count(movieId) %>% 
  left_join(b_i_estimates_reg, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10) %>% 
  pull(title)
bottom_10_movies_reg <- train %>%
  count(movieId) %>% 
  left_join(b_i_estimates_reg, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(b_i) %>% 
  slice(1:10) %>% 
  pull(title)
# These movies make a lot more sense

# Next, let's explore the users
rm(movie_effect_model_reg,
   b_i_estimates_reg,
   best_lambda,
   rmses_for_different_lambdas,
   lambdas)

################################
# User
################################

# To see if the user may be an interesting preditor for our model, let's plot the average user ratings
average_user_rating <- edx %>%
  group_by(userId) %>% 
  summarize(rating=mean(rating)) %>%
  ggplot(aes(rating)) +
  geom_histogram(bins=30, color="black") +
  ggtitle("Average User Rating")
# Since the results have some variety, it seems like user will be a useful predictor to add to our model.

# Our modified model is: Y_u,i = mu + b_i + b_u + epsilon_u,i.
# b_u is the user effect, or the average ranking for user u.
# Our b_u is estimated by Y_u,i - mu - b_i for each user u.
# Remember that last time we had to use regularization because there were movie that didn't have many ratings
# Let's see if we need to regularize users as well
number_of_ratings_per_user <- edx %>% 
  group_by(userId) %>% 
  summarize(numRatings=n()) %>%
  filter(numRatings < 500) %>%
  ggplot(aes(numRatings)) +
  geom_histogram(bins=30, color="black") +
  ggtitle("Number of Ratings per User")
# We can see that some users hardly rated any movies

# So, we'll need to use regularization by adding a lambda to our estimate of b_u
# b_u_regularized = sum(ratings - mu - b_i)/(lambda + count_ratings)
# To choose the right value of lambda we'll need to use cross valdation
lambdas <- seq(0, 10, 0.25)
rmses_for_different_lambdas <- sapply(lambdas, function(l){
  mu <- mean(train$rating)
  b_i <- train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating-mu)/(n()+l))
  b_u <- train %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u=sum(rating-b_i-mu)/(n()+l))
  movie_user_effect_model_reg <- test %>% 
    left_join(b_i, by='movieId') %>% 
    left_join(b_u, by='userId') %>% 
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(movie_user_effect_model_reg, test$rating))
})
qplot(lambdas, rmses_for_different_lambdas)  
best_lambda <- lambdas[which.min(rmses_for_different_lambdas)]

# Let's create the model using the regularized b_i_estimates, b_u_estimates, and the best lambda (5)
b_i_estimates_reg <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+best_lambda), n_i = n())
b_u_estimates_reg <- train %>%
  left_join(b_i_estimates_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u=sum(rating-b_i-mu)/(n()+best_lambda))
movie_user_effect_model_reg <- test %>% 
  left_join(b_i_estimates_reg, by='movieId') %>% 
  left_join(b_u_estimates_reg, by='userId') %>% 
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Find the RMSE of the model
rmse_movie_user_reg <- RMSE(movie_user_effect_model_reg, test$rating)
# RMSE of this model is 0.8641362, which is a big improvement

# Next, let's explore the genres
rm(movie_user_effect_model_reg,
   b_i_estimates_reg,
   b_u_estimates_reg,
   best_lambda,
   rmses_for_different_lambdas,
   lambdas)

################################
# Genre
################################

# To see if the genre may be an interesting preditor for our model, let's plot the average genre ratings
average_genre_rating <- edx %>%
  group_by(genres) %>% 
  summarize(rating=mean(rating)) %>%
  ggplot(aes(rating)) +
  geom_histogram(bins=30, color="black") +
  ggtitle("Average Genre Rating")
# Since the results have some variety, it seems like genre will be a useful predictor to add to our model.

# Our modified model is: Y_u,i = mu + b_i + b_u + b_g epsilon_u,i.
# b_g is the genre effect, or the average ranking for genre g.
# Our b_g is estimated by Y_u,i - mu - b_i - b_u for each genre g.
# Remember that last time we had to use regularization because there were movie that didn't have many ratings
# Let's see if we need to regularize genres as well
number_of_ratings_per_genre <- edx %>% 
  group_by(genres) %>% 
  summarize(numRatings=n()) %>%
  filter(numRatings < 500) %>%
  ggplot(aes(numRatings)) +
  geom_histogram(bins=30, color="black") +
  ggtitle("Number of Ratings per Genre")
# We can see that some genres don't have many ratings

# So, we'll need to use regularization by adding a lambda to our estimate of b_g
# b_g_regularized = sum(ratings - mu - b_i - b_u)/(lambda + count_ratings)
# To choose the right value of lambda we'll need to use cross valdation
lambdas <- seq(0, 10, 0.25)
rmses_for_different_lambdas <- sapply(lambdas, function(l){
  mu <- mean(train$rating)
  b_i <- train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating-mu)/(n()+l))
  b_u <- train %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u=sum(rating-b_i-mu)/(n()+l))
  b_g <- train %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by='userId') %>% 
    group_by(genres) %>%
    summarize(b_g=sum(rating-b_i-b_u-mu)/(n()+l))
  movie_user_genre_effect_model_reg <- test %>% 
    left_join(b_i, by='movieId') %>% 
    left_join(b_u, by='userId') %>% 
    left_join(b_g, by='genres') %>% 
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)
  return(RMSE(movie_user_genre_effect_model_reg, test$rating))
})
qplot(lambdas, rmses_for_different_lambdas)  
best_lambda <- lambdas[which.min(rmses_for_different_lambdas)]

# Let's create the model using the regularized b_i_estimates, b_u_estimates, and the best lambda (4.75)
b_i_estimates_reg <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+best_lambda), n_i = n())
b_u_estimates_reg <- train %>%
  left_join(b_i_estimates_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u=sum(rating-b_i-mu)/(n()+best_lambda))
b_g_estimates_reg <- train %>%
  left_join(b_i_estimates_reg, by='movieId') %>% 
  left_join(b_u_estimates_reg, by='userId') %>% 
  group_by(genres) %>%
  summarize(b_g=sum(rating-b_i-b_u-mu)/(n()+best_lambda))
movie_user_genre_effect_model_reg <- test %>% 
  left_join(b_i_estimates_reg, by='movieId') %>% 
  left_join(b_u_estimates_reg, by='userId') %>% 
  left_join(b_g_estimates_reg, by='genres') %>% 
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

# Find the RMSE of the model
rmse_movie_user_genres_reg <- RMSE(movie_user_genre_effect_model_reg, test$rating)
# RMSE of this model is 0.8638141, which is a slight improvement

rm(movie_user__genres_effect_model_reg,
   b_i_estimates_reg,
   b_u_estimates_reg,
   b_g_estimates_reg,
   best_lambda,
   rmses_for_different_lambdas,
   lambdas)

################################
# Select final model and find RMSE
################################

# Plot the RMSEs of the method (using the test set)
rmse_results <- tibble(method = c("Movie Effect Model", "Regularized Movie Effect Model", "Regularized Movie + User Effect Model", "Regularized Movie + User + Genre Effect Model"),
                       RMSE = c(rmse_movie, rmse_movie_reg, rmse_movie_user_reg, rmse_movie_user_genres_reg))
rmse_results
# Since our intermediate RMSEs are looking pretty good, let's pick the best and move on to validation

# Choose the best model, run the model on validation set, and find RMSE
best_lambda <- 4.75
b_i_estimates_reg <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+best_lambda), n_i = n())
b_u_estimates_reg <- edx %>%
  left_join(b_i_estimates_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u=sum(rating-b_i-mu)/(n()+best_lambda))
b_g_estimates_reg <- edx %>%
  left_join(b_i_estimates_reg, by="movieId") %>%
  left_join(b_u_estimates_reg, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g=sum(rating-b_i-b_u-mu)/(n()+best_lambda))
movie_user_genre_effect_model_reg <- validation %>% 
  left_join(b_i_estimates_reg, by='movieId') %>% 
  left_join(b_u_estimates_reg, by='userId') %>% 
  left_join(b_g_estimates_reg, by='genres') %>% 
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)
final_rmse <- RMSE(movie_user_genre_effect_model_reg, validation$rating)
final_rmse
# The final RMSE of the validation set is 0.8644514