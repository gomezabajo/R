#sources: https://xgboost.readthedocs.io/en/stable/R-package/xgboostPresentation.html
#         https://cran.r-project.org/web/packages/lightgbm/index.html
#         https://www.rdocumentation.org/packages/randomForest/versions/4.7-1.1/topics/randomForest


train <- read.csv(file.choose(), header=TRUE, sep=',')
test <- read.csv(file.choose(), header=TRUE, sep=',')


xTrain <- subset(train, select=-Name)
xTrain <- subset(xTrain, select=-PassengerId)
xTrain <- subset(xTrain, select=-Transported)
xTest <- subset(test, select=-Name)
xTest <- subset(xTest, select=-PassengerId)



vars <- c("HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")
xTrain[vars] <- lapply(xTrain[vars], as.factor)
xTest[vars] <- lapply(xTest[vars], as.factor)

for (col in vars) {
    xTrain[[col]][is.na(xTrain[[col]])] <- names(sort(-table(xTrain[[col]])))[1]
}

for (col in vars) {
    xTest[[col]][is.na(xTest[[col]])] <- names(sort(-table(xTest[[col]])))[1]
}


xTrain$Age[is.na(xTrain$Age)] <- median(xTrain$Age, na.rm = TRUE)
xTest$Age[is.na(xTest$Age)] <- median(xTest$Age, na.rm = TRUE)

library(tidyr)
library(dplyr)

xTrain <- xTrain %>% 
    separate(Cabin, into = c("Deck", "Num", "Side"), sep = "/") %>%
    mutate(Deck = as.factor(Deck))

xTrain <- subset(xTrain, select=-Num)
xTrain <- subset(xTrain, select=-Side)

xTest <- xTest %>% 
    separate(Cabin, into = c("Deck", "Num", "Side"), sep = "/") %>%
    mutate(Deck = as.factor(Deck))

xTest <- subset(xTest, select=-Num)
xTest <- subset(xTest, select=-Side)


unique(xTrain$HomePlanet)
xTrain$HomePlanet <- as.integer(factor(xTrain$HomePlanet, levels=c("Earth", "Europa", "Mars")))
unique(xTrain$Deck)
xTrain$Deck <- as.integer(factor(xTrain$Deck, levels=c("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")))
unique(xTrain$Destination)
xTrain$Destination <- as.integer(factor(xTrain$Destination, levels=c("55 Cancri e", "PSO J318.5-22", "TRAPPIST-1e")))

unique(xTest$HomePlanet)
xTest$HomePlanet <- as.integer(factor(xTest$HomePlanet, levels=c("Earth", "Europa", "Mars")))
unique(xTest$Deck)
xTest$Deck <- as.integer(factor(xTest$Deck, levels=c("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")))
unique(xTest$Destination)
xTest$Destination <- as.integer(factor(xTest$Destination, levels=c("55 Cancri e", "PSO J318.5-22", "TRAPPIST-1e")))

categoricalVars <- c("HomePlanet", "CryoSleep", "Destination", "VIP", "Deck")
xTrain[categoricalVars] <- lapply(xTrain[categoricalVars], as.factor)
xTest[categoricalVars] <- lapply(xTest[categoricalVars], as.factor)

for (col in categoricalVars) {
    xTrain[[col]][is.na(xTrain[[col]])] <- names(sort(-table(xTrain[[col]])))[1]
}

for (col in categoricalVars) {
    xTest[[col]][is.na(xTest[[col]])] <- names(sort(-table(xTest[[col]])))[1]
}

install.packages("corrplot")
library(corrplot)
corrplot(carsCorr, method="color")

xTrainCor <- sapply(xTrain, as.numeric)
xTrainCor <- cor(xTrainCor)
corrplot(xTrainCor, method="color")

xTestCor <- sapply(xTest, as.numeric)
xTestCor <- cor(xTestCor)
corrplot(xTestCor, method="color")

install.packages("xgboost")
library(xgboost)
yTrain <- as.factor(train$Transported)

dTrain <- xgb.DMatrix(data = as.matrix(sapply(xTrain, as.numeric)), label = sapply(yTrain, as.numeric) - 1)

dTest <- xgb.DMatrix(data = as.matrix(sapply(xTest, as.numeric)))

params <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    eta = 0.05,
    max_depth = 8,
    subsample = 0.9,
    colsample_bytree = 0.9
)

model = xgb.train(
    params = params,
    data = dTrain,
    nrounds = 100,
    verbose = 1
)

predictions <- predict(model, dTest)

predictions <- ifelse(predictions > 0.5, 1, 0)

submission <- data.frame(PassengerId = test$PassengerId, Transported = as.logical(predictions))

write.csv(submission, file.choose(), row.names = FALSE)


params_multi <- list(
  objective = "multi:softprob",
  num_class = 2,
  eval_metric = "mlogloss",
  eta = 0.05,
  max_depth = 8,
  subsample = 0.9,
  colsample_bytree = 0.9
)

model_multi <- xgb.train(
    params = params_multi, 
    data = dTrain,
    nrounds = 100)

predictions_multi <- predict(model_multi, dTest)
predictions_multi <- matrix(predictions_multi, ncol = 2, byrow = TRUE)[,2]

predictions_multi <- ifelse(predictions_multi > 0.5, 1, 0)

submission <- data.frame(PassengerId = test$PassengerId, Transported = as.logical(predictions_multi))

write.csv(submission, file.choose(), row.names = FALSE)


install.packages("lightgbm")
library(lightgbm)

dTrain_lgb <- lgb.Dataset(data = as.matrix(sapply(xTrain, as.numeric)), label = sapply(yTrain, as.numeric) - 1)

params_lgb <- list(
  objective = "binary",
  metric = "binary_logloss",
  learning_rate = 0.05,
  num_leaves = 31,
  max_depth = -1
)

model_lgb <- lgb.train(params = params_lgb, data = dTrain_lgb, nrounds = 100)
predictions_lgb <- predict(model_lgb, as.matrix(sapply(xTest, as.numeric)))

predictions_lgb <- ifelse(predictions_lgb > 0.5, 1, 0)

submission <- data.frame(PassengerId = test$PassengerId, Transported = as.logical(predictions_lgb))

write.csv(submission, file.choose(), row.names = FALSE)

install.packages("randomForest")
library(randomForest)

model_rf <- randomForest(sapply(yTrain, as.factor) ~ ., data = sapply(xTrain, as.numeric), ntree = 500)
predictions_rf <- predict(model_rf, newdata = sapply(xTest, as.numeric), type="prob")[,2]

predictions_rf <- ifelse(predictions_rf > 0.5, 1, 0)

submission <- data.frame(PassengerId = test$PassengerId, Transported = as.logical(predictions_rf))

write.csv(submission, file.choose(), row.names = FALSE)

