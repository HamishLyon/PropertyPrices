########################################################################################
#
# Check for required packages; if absent, install.
#
########################################################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(naniar)) install.packages("naniar", repos = "http://cran.us.r-project.org")
if(!require(UpSetR)) install.packages("UpSetR", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(mice)) install.packages("mice", repos = "http://cran.us.r-project.org")
if(!require(Matrix)) install.packages("Matrix", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")

########################################################################################
#
# Import Data from .csv to data frame with headers; using the data.table::fread function 
# to skip blank lines & trim white space.
#
########################################################################################

# HouseData <- read.csv(".../train.csv")


########################################################################################
#
# Produce basic summaries; first observations are that there are a lot of features 
# (columns) to the dataset, there are a lot of categorical data & continuous. There
# are also a lot of NAs present.
#
########################################################################################

dim(HouseData)
tibble(HouseData)
str(HouseData)

########################################################################################
#
# Plot the missing variables; plot the groups of missingness ising naniar
#
########################################################################################

gg_miss_var(HouseData)
HouseData %>%
  as_shadow_upset() %>%
  upset()

########################################################################################
#
# Address the features which have NAs by using r Multivariate Imputation by Chained 
# Equations. It uses a slightly uncommon way of implementing the imputation in 2-steps, 
# using mice() to build the model and complete() to generate the completed data. We are
# performing imputation based on random forests. Add NA to the factor for PoolQC,
# MiscFeature, Alley, & Fence.
#
########################################################################################

MissingData <- c('PoolQC', 'MiscFeature', 'Alley', 'Fence')
ImputableData <- setdiff(names(HouseData), MissingData)
RemovedHouseData <- HouseData[MissingData]
HouseData <- HouseData[ImputableData]

HouseData <- HouseData %>%
  mutate(MSSubClass = as.factor(HouseData$MSSubClass))

imputatedHouse <- mice(HouseData, m=1, method='cart', printFlag=FALSE)
HouseData <- complete(imputatedHouse)

RemovedHouseData <- RemovedHouseData %>% 
  mutate(PoolQC = addNA(PoolQC),
         MiscFeature = addNA(MiscFeature),
         Alley = addNA(Alley),
         Fence = addNA(Fence))

HouseData <- cbind(HouseData, RemovedHouseData)
rm(imputatedHouse, RemovedHouseData, ImputableData, MissingData)

# anyNA(HouseData)

########################################################################################
#
# Check for outliers; remove outliers on TotalBsmtSF, GrLivArea, LotFrontage, & LotArea
#
########################################################################################

HouseData %>%
  ggplot(aes(GrLivArea)) +
  geom_histogram()
  
HouseData %>%
  ggplot(aes(TotalBsmtSF)) +
  geom_histogram()

HouseData %>%
  ggplot(aes(LotArea)) + 
  geom_histogram()

HouseData %>%
  ggplot(aes(LotFrontage)) +
  geom_histogram()

########################################################################################
#
# Remove outliers
#
########################################################################################

HouseData <- HouseData %>%
              filter(GrLivArea < 3999.9,
                     TotalBsmtSF < 2999.9,
                     LotArea < 9999.9,
                     LotFrontage < 199.9)

########################################################################################
#
# Transform some numerical values into factors
#
########################################################################################

HouseData <- HouseData %>%
  mutate(MSSubClass = factor(MSSubClass),
         OverallCond = factor(OverallCond))

########################################################################################
#
# Inspect the SalePrice variable, the target of our prediction. Considering that House 
# Data is skewed to the left we can do a log transform, a little reading reveals that 
# it's common to do a variable + 1 transform to protect your dataset from transformations
# of zero or NA values.
#
########################################################################################

HouseData %>%
  ggplot(aes(x = SalePrice)) +
  geom_histogram(bins=10) + 
  ylab(label = "Frequency") +
  xlab (label = "SalePrice ")

HouseData %>%
  mutate(SalePrice = log(SalePrice + 1)) %>%
  ggplot(aes(x = SalePrice)) +
  geom_histogram() + 
  ylab(label = "Frequency") + 
  xlab(label = "Log(SalePrice + 1)") 

########################################################################################
#
# Transform target variable, SalePrice, into log(SalePrice + c)
#
########################################################################################

HouseData$SalePrice <- log(HouseData$SalePrice + 1)

########################################################################################
#
# Split data into training and testing sets, 10% of data is reserved for testing while
# 90% is kept for training. 
#
########################################################################################

set.seed(2)
indexSet <- createDataPartition(y = HouseData$SalePrice, times = 1, p = 0.1, list = FALSE)
trainingSet <- HouseData[-indexSet, ]
testingSet <- HouseData[indexSet, ]

########################################################################################
#
# Modelling: Price ~ cart(House Data)
#
########################################################################################

set.seed(2)
treeModel <- rpart(SalePrice~. -Id,
                   data = trainingSet,
                   control = rpart.control(cp = 0.01))

rpart.plot(treeModel)

########################################################################################
#
# Analysis into important features by visualising relationships, looking for visual
# correlations of the data.
#
########################################################################################

HouseData %>%
  ggplot(aes(x = GrLivArea, y = SalePrice)) + 
  geom_point() +
  ggtitle("GrLivArea & log(Price)")

HouseData %>%
  ggplot(aes(x = MSZoning, y = SalePrice)) + 
  geom_boxplot() +
  ggtitle("TotalBsmtSF & log(Price)")

HouseData %>%
  ggplot(aes(x = YearBuilt, y = SalePrice)) + 
  geom_point() +
  ggtitle("YearBuilt & log(Price)")

HouseData %>%
  ggplot(aes(x = LotArea, y = SalePrice)) + 
  geom_point() +
  ggtitle("LotArea & log(Price)")

HouseData %>%
  ggplot(aes(x = GarageArea, y = SalePrice)) + 
  geom_point() +
  ggtitle("GarageArea & log(Price)")

########################################################################################
#
# Modelling: Price ~ lm(House Data)
#
########################################################################################

LinearModel_1 <- lm(SalePrice ~ GrLivArea + TotalBsmtSF + YearBuilt + LotArea + GarageArea, data = trainingSet)
summary(LinearModel_1)

########################################################################################
#
# Analysis of some catagorical sets of importance
#
########################################################################################

HouseData %>%
  ggplot(aes(x = Neighborhood, y = SalePrice)) + 
  geom_boxplot() +
  ggtitle("Neighborhood & log(Price)")

########################################################################################
#
# Analysis of some catagorical sets of importance
#
########################################################################################

LinearModel_2 <- lm(SalePrice ~ GrLivArea + TotalBsmtSF + YearBuilt + LotArea + GarageArea + Neighborhood, data = trainingSet)
summary(LinearModel_2)

########################################################################################
#
# Modelling: Price ~ xgboost(House Data)
#
########################################################################################

set.seed(2)
extremeGradientBoosting <- xgboost(
  params = list(
  max_depth = 5,
  eta = 0.02,
  gamma = 0,
  colsample_bytree = 0.65,
  subsample = 0.6,
  min_child_weight = 3), 
  data = sparse.model.matrix(
    SalePrice~ .-Id, 
    trainingSet), 
  label = trainingSet$SalePrice, 
  nrounds = 1000, nfold = 10, 
  showsd = F, 
  stratified = T, 
  print_every_n = 100, 
  early_stopping_rounds = 50, 
  maximize = F)

ImportantFeatures <- xgb.importance(model = extremeGradientBoosting)
ImportantFeatures %>% top_n(10) %>% kable()
ImportantFeatures %>% top_n(-10) %>% kable()

########################################################################################
#
# Predicting Results using the predict function and the models. Also, turning the testing
# data into a sparse matrix so that it is a comparable object to compare with the model.
#
########################################################################################

treePrediction <- predict(object = treeModel, newdata = testingSet)

lmPrediction_1 <- predict(object = LinearModel_1, newdata = testingSet)

lmPrediction_2 <- predict(object = LinearModel_2, newdata = testingSet)

testingData <- sparse.model.matrix(
  SalePrice~ .-Id, 
  testingSet) 

xgbPrediction <- predict(object = extremeGradientBoosting, newdata = testingData)


########################################################################################
#
# Loss function, using the formula from the textbook.
#
########################################################################################

RMSE <- function(predicted, observed){
  sqrt(mean((observed - predicted)^2))
}

########################################################################################
#
# Evaluating Results and adding the results to a tibble to print out for the rmarkdown
# report
#
########################################################################################

Results <- tibble("Prediction Method" = "Cart Model", "RSME" = RMSE(testingSet$SalePrice, treePrediction))
Results <- Results %>% 
  add_row("Prediction Method" = "Linear Regression", "RSME" = RMSE(testingSet$SalePrice, lmPrediction_1))
Results <- Results %>% 
  add_row("Prediction Method" = "Linear Regression, with Suburb", "RSME" = RMSE(testingSet$SalePrice, lmPrediction_2))
Results <- Results%>% 
  add_row("Prediction Method" = "Extreme Gradient Boosting", "RSME" = RMSE(testingSet$SalePrice, xgbPrediction))
Results %>% kable()
