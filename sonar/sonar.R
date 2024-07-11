# source used: Dimensionality and Reduction https://metudatascience.github.io/datascience/discretization.html

install.packages("caret")
install.packages("mlbench")
install.packages("arules")
library(caret)
library(mlbench)
library(cluster)
library(arules)
data(Sonar)
View(Sonar)
table(Sonar$Class)
sum(is.na(Sonar))
anyNA(Sonar)
summary(Sonar)
nrow(Sonar)
ncol(Sonar)
set.seed(11)
#columns <- sample(x = 1:60, size = 9)
columns <- NULL
for (i in 1:60) {
  columns <- cbind(columns, i)
}
colnames.use <- NULL
for (i in columns) {
    colnames.use <- cbind(colnames.use, colnames(Sonar)[i])
}
colnames(columns) <- colnames.use
#pre_var <- Sonar[, columns]
pre_var <- Sonar
par(mfrow = c(3, 3))
for (i in 1: ncol(pre_var)) {
    boxplot(pre_var[, i], xlab = names(pre_var[i]),
                        main = paste("Boxplot of ", names(pre_var[i])),
						horizontal = TRUE, col = "steelblue")
}

d <- NULL
data.eqw <- NULL
for (i in columns){
    d <- discretize(Sonar[, i], method = "interval", breaks =20)
    data.eqw <- cbind(data.eqw, d)
}
summary(data.eqw)
data.eqw <- cbind(data.eqw, Sonar[61])
colnames.use <- NULL
for (i in columns) {
    colnames.use <- cbind(colnames.use, colnames(Sonar)[i])
}
colnames.use <- cbind(colnames.use, colnames(Sonar[61]))
colnames(data.eqw) <- colnames.use
colnames(data.eqw)
set.seed(3456)
trainIndex <- createDataPartition(data.eqw$Class, p=.7, list=FALSE, times = 1)
mySonarTrain <- data.eqw[trainIndex,]
mySonarTest <- data.eqw[-trainIndex,]
set.seed(825)
fitControl <- trainControl(method = 'cv', number = 5, classProbs = TRUE)

eqw.kknnFit1 <- train(Class ~ ., data=mySonarTrain, method = "kknn", tuneGrid = expand.grid(kmax = 5, kernel = c('gaussian', 'triangular', 'rectangular', 'epanechnikov', 'optimal'), distance = 1), trControl = fitControl)
eqw.kknnPred <- predict(eqw.kknnFit1, mySonarTest)
postResample(pred = eqw.kknnPred, obs = mySonarTest$Class)
eqw.kknnCf <- confusionMatrix(data = eqw.kknnPred, reference = mySonarTest$Class)
str(eqw.kknnCf)
eqw.knnW <- eqw.kknnCf$overall

fitControl <- trainControl(method = 'cv', number = 5, classProbs = TRUE)
eqw.j48Fit1 <- train(Class ~ ., data=mySonarTrain, method = "J48", tuneLength = 1, metric='ROC', preProc = c('center', 'scale'), trControl = fitControl)
eqw.j48Pred <- predict(eqw.j48Fit1, mySonarTest)
postResample(pred = eqw.j48Pred, obs = mySonarTest$Class)
eqw.j48Cf <- confusionMatrix(data = eqw.j48Pred, reference = mySonarTest$Class)
str(eqw.j48Cf)
eqw.j48 <- eqw.j48Cf$overall

fitControl <- trainControl(method = 'cv', number = 5, classProbs = TRUE)
eqw.nbFit1 <- train(Class ~ ., data=mySonarTrain, method= 'nb', trControl = fitControl)
eqw.nbPred <- predict(eqw.nbFit1, mySonarTest)
postResample(pred = eqw.nbPred, obs = mySonarTest$Class)
eqw.nbCf <- confusionMatrix(data = eqw.nbPred, reference = mySonarTest$Class)
str(eqw.nbCf)
eqw.nb <- eqw.nbCf$overall

fitControl <- trainControl(method = 'cv', number = 5, classProbs = TRUE)
eqw.mlp5Fit1 <- train(Class ~ ., data=mySonarTrain, method= 'mlp', tuneGrid = expand.grid(size = 5), preProcess = c('center','scale'), linOut = T, metric = 'Accuracy', trControl = fitControl)
eqw.mlp5Pred <- predict(eqw.mlp5Fit1, mySonarTest)
postResample(pred = eqw.mlp5Pred, obs = mySonarTest$Class)
eqw.mlp5Cf <- confusionMatrix(data = eqw.mlp5Pred, reference = mySonarTest$Class)
str(eqw.mlp5Cf)
eqw.mlp5 <- eqw.mlp5Cf$overall

set.seed(11)
d <- NULL
data.eqf <- NULL
for (i in columns){
    d <- discretize(Sonar[, i], method = "frequency", breaks =11)
    data.eqf <- cbind(data.eqf, d)
}
summary(data.eqf)
data.eqf <- cbind(data.eqf, Sonar[61])
colnames.use <- NULL
for (i in columns) {
    colnames.use <- cbind(colnames.use, colnames(Sonar)[i])
}
colnames.use <- cbind(colnames.use, colnames(Sonar[61]))
colnames(data.eqf) <- colnames.use
colnames(data.eqf)
set.seed(3456)
trainIndex <- createDataPartition(data.eqf$Class, p=.7, list=FALSE, times = 1)
mySonarTrain <- data.eqf[trainIndex,]
mySonarTest <- data.eqf[-trainIndex,]
set.seed(825)
fitControl <- trainControl(method = 'cv', number = 5, classProbs = TRUE)

eqf.kknnFit1 <- train(Class ~ ., data=mySonarTrain, method = "kknn", tuneGrid = expand.grid(kmax = 5, kernel = c('gaussian', 'triangular', 'rectangular', 'epanechnikov', 'optimal'), distance = 1), trControl = fitControl)
eqf.kknnPred <- predict(eqf.kknnFit1, mySonarTest)
postResample(pred = eqf.kknnPred, obs = mySonarTest$Class)
eqf.kknnCf <- confusionMatrix(data = eqf.kknnPred, reference = mySonarTest$Class)
str(eqf.kknnCf)
eqf.knnW <- eqf.kknnCf$overall

fitControl <- trainControl(method = 'cv', number = 5, classProbs = TRUE)
eqf.j48Fit1 <- train(Class ~ ., data=mySonarTrain, method = "J48", tuneLength = 1, metric='ROC', preProc = c('center', 'scale'), trControl = fitControl)
eqf.j48Pred <- predict(eqf.j48Fit1, mySonarTest)
postResample(pred = eqf.j48Pred, obs = mySonarTest$Class)
eqf.j48Cf <- confusionMatrix(data = eqf.j48Pred, reference = mySonarTest$Class)
str(eqf.j48Cf)
eqf.j48 <- eqf.j48Cf$overall

fitControl <- trainControl(method = 'cv', number = 5, classProbs = TRUE)
eqf.nbFit1 <- train(Class ~ ., data=mySonarTrain, method= 'nb', trControl = fitControl)
eqf.nbPred <- predict(eqf.nbFit1, mySonarTest)
postResample(pred = eqf.nbPred, obs = mySonarTest$Class)
eqf.nbCf <- confusionMatrix(data = eqf.nbPred, reference = mySonarTest$Class)
str(eqf.nbCf)
eqf.nb <- eqf.nbCf$overall

fitControl <- trainControl(method = 'cv', number = 5, classProbs = TRUE)
eqf.mlp5Fit1 <- train(Class ~ ., data=mySonarTrain, method= 'mlp', tuneGrid = expand.grid(size = 5), preProcess = c('center','scale'), linOut = T, metric = 'Accuracy', trControl = fitControl)
eqf.mlp5Pred <- predict(eqf.mlp5Fit1, mySonarTest)
postResample(pred = eqf.mlp5Pred, obs = mySonarTest$Class)
eqf.mlp5Cf <- confusionMatrix(data = eqf.mlp5Pred, reference = mySonarTest$Class)
str(eqf.mlp5Cf)
eqf.mlp5 <- eqf.mlp5Cf$overall

set.seed(11)
d <- NULL
data.eqc <- NULL
for (i in columns){
    d <- discretize(Sonar[, i], method = "cluster", breaks = 7)
    data.eqc <- cbind(data.eqc, d)
}
summary(data.eqc)
data.eqc <- cbind(data.eqc, Sonar[61])
colnames.use <- NULL
for (i in columns) {
    colnames.use <- cbind(colnames.use, colnames(Sonar)[i])
}
colnames.use <- cbind(colnames.use, colnames(Sonar[61]))
colnames(data.eqc) <- colnames.use
colnames(data.eqc)
set.seed(3456)
trainIndex <- createDataPartition(data.eqc$Class, p=.7, list=FALSE, times = 1)
mySonarTrain <- data.eqc[trainIndex,]
mySonarTest <- data.eqc[-trainIndex,]
set.seed(825)
fitControl <- trainControl(method = 'cv', number = 5, classProbs = TRUE)

eqc.kknnFit1 <- train(Class ~ ., data=mySonarTrain, method = "kknn", tuneGrid = expand.grid(kmax = 5, kernel = c('gaussian', 'triangular', 'rectangular', 'epanechnikov', 'optimal'), distance = 1), trControl = fitControl)
eqc.kknnPred <- predict(eqc.kknnFit1, mySonarTest)
postResample(pred = eqc.kknnPred, obs = mySonarTest$Class)
eqc.kknnCf <- confusionMatrix(data = eqc.kknnPred, reference = mySonarTest$Class)
str(eqc.kknnCf)
eqc.knnW <- eqc.kknnCf$overall

fitControl <- trainControl(method = 'cv', number = 5, classProbs = TRUE)
eqc.j48Fit1 <- train(Class ~ ., data=mySonarTrain, method = "J48", tuneLength = 1, metric='ROC', preProc = c('center', 'scale'), trControl = fitControl)
eqc.j48Pred <- predict(eqc.j48Fit1, mySonarTest)
postResample(pred = eqc.j48Pred, obs = mySonarTest$Class)
eqc.j48Cf <- confusionMatrix(data = eqc.j48Pred, reference = mySonarTest$Class)
str(eqc.j48Cf)
eqc.j48 <- eqc.j48Cf$overall

fitControl <- trainControl(method = 'cv', number = 5, classProbs = TRUE)
eqc.nbFit1 <- train(Class ~ ., data=mySonarTrain, method= 'nb', trControl = fitControl)
eqc.nbPred <- predict(eqc.nbFit1, mySonarTest)
postResample(pred = eqc.nbPred, obs = mySonarTest$Class)
eqc.nbCf <- confusionMatrix(data = eqc.nbPred, reference = mySonarTest$Class)
str(eqc.nbCf)
eqc.nb <- eqc.nbCf$overall

fitControl <- trainControl(method = 'cv', number = 5, classProbs = TRUE)
eqc.mlp5Fit1 <- train(Class ~ ., data=mySonarTrain, method= 'mlp', tuneGrid = expand.grid(size = 5), preProcess = c('center','scale'), linOut = T, metric = 'Accuracy', trControl = fitControl)
eqc.mlp5Pred <- predict(eqc.mlp5Fit1, mySonarTest)
postResample(pred = eqc.mlp5Pred, obs = mySonarTest$Class)
eqc.mlp5Cf <- confusionMatrix(data = eqc.mlp5Pred, reference = mySonarTest$Class)
str(eqc.mlp5Cf)
eqc.mlp5 <- eqc.mlp5Cf$overall

set.seed(11)
d <- NULL
data <- NULL
for (i in columns){
    d <- Sonar[, i]
    data <- cbind(data, d)
}
summary(data)
data <- cbind(data, Sonar[61])
colnames.use <- NULL
for (i in columns) {
    colnames.use <- cbind(colnames.use, colnames(Sonar)[i])
}
colnames.use <- cbind(colnames.use, colnames(Sonar[61]))
colnames(data) <- colnames.use
colnames(data)
set.seed(3456)
trainIndex <- createDataPartition(data$Class, p=.7, list=FALSE, times = 1)
mySonarTrain <- data[trainIndex,]
mySonarTest <- data[-trainIndex,]
set.seed(825)
fitControl <- trainControl(method = 'cv', number = 5, classProbs = TRUE)

kknnFit1 <- train(Class ~ ., data=mySonarTrain, method = "kknn", tuneGrid = expand.grid(kmax = 5, kernel = c('gaussian', 'triangular', 'rectangular', 'epanechnikov', 'optimal'), distance = 1), trControl = fitControl)
kknnPred <- predict(kknnFit1, mySonarTest)
postResample(pred = kknnPred, obs = mySonarTest$Class)
kknnCf <- confusionMatrix(data = kknnPred, reference = mySonarTest$Class)
str(kknnCf)
knnW <- kknnCf$overall

fitControl <- trainControl(method = 'cv', number = 5, classProbs = TRUE)
j48Fit1 <- train(Class ~ ., data=mySonarTrain, method = "J48", tuneLength = 1, metric='ROC', preProc = c('center', 'scale'), trControl = fitControl)
j48Pred <- predict(j48Fit1, mySonarTest)
postResample(pred = j48Pred, obs = mySonarTest$Class)
j48Cf <- confusionMatrix(data = j48Pred, reference = mySonarTest$Class)
str(j48Cf)
j48 <- j48Cf$overall

fitControl <- trainControl(method = 'cv', number = 5, classProbs = TRUE)
nbFit1 <- train(Class ~ ., data=mySonarTrain, method= 'nb', trControl = fitControl)
nbPred <- predict(nbFit1, mySonarTest)
postResample(pred = nbPred, obs = mySonarTest$Class)
nbCf <- confusionMatrix(data = nbPred, reference = mySonarTest$Class)
str(nbCf)
nb <- nbCf$overall

fitControl <- trainControl(method = 'cv', number = 5, classProbs = TRUE)
mlp5Fit1 <- train(Class ~ ., data=mySonarTrain, method= 'mlp', tuneGrid = expand.grid(size = 5), preProcess = c('center','scale'), linOut = T, metric = 'Accuracy', trControl = fitControl)
mlp5Pred <- predict(mlp5Fit1, mySonarTest)
postResample(pred = mlp5Pred, obs = mySonarTest$Class)
mlp5Cf <- confusionMatrix(data = mlp5Pred, reference = mySonarTest$Class)
str(mlp5Cf)
mlp5 <- mlp5Cf$overall

# Crear un dataframe con los resultados de los modelos
resultados <- data.frame(
  Modelo = c("KKNN", "J48", "NB", "MLP",
                    "KKNN (EqW)", "J48 (EqW)", "NB (EqW)", "MLP (EqW)", 
                    "KKNN (EqF)", "J48 (EqF)", "NB (EqF)", "MLP (EqF)",
                    "KKNN (EqC)", "J48 (EqC)", "NB (EqC)", "MLP (EqC)"),
  Accuracy = c(knnW["Accuracy"], j48["Accuracy"], nb["Accuracy"], mlp5["Accuracy"], 
               eqw.knnW["Accuracy"], eqw.j48["Accuracy"], eqw.nb["Accuracy"], eqw.mlp5["Accuracy"],
               eqf.knnW["Accuracy"], eqf.j48["Accuracy"], eqf.nb["Accuracy"], eqf.mlp5["Accuracy"],
               eqc.knnW["Accuracy"], eqc.j48["Accuracy"], eqc.nb["Accuracy"], eqc.mlp5["Accuracy"]),
  Kappa = c(knnW["Kappa"], j48["Kappa"], nb["Kappa"], mlp5["Kappa"], 
            eqw.knnW["Kappa"], eqw.j48["Kappa"], eqw.nb["Kappa"], eqw.mlp5["Kappa"],
            eqf.knnW["Kappa"], eqf.j48["Kappa"], eqf.nb["Kappa"], eqf.mlp5["Kappa"],
            eqc.knnW["Kappa"], eqc.j48["Kappa"], eqc.nb["Kappa"], eqc.mlp5["Kappa"])
)

# Ordenar los resultados por Accuracy descendente
resultados <- resultados[order(-resultados$Accuracy),]

# Crear la gráfica de barras agrupadas
library(ggplot2)

ggplot(resultados, aes(x = Modelo, y = Accuracy, fill = Modelo)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(Accuracy, 3)), vjust = -0.5, position = position_dodge(width = 0.9)) +
  labs(title = "Comparación de Modelos Predictivos",
       x = "Modelo", y = "Accuracy") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_discrete(name = "Método de\nDiscretización") 

# Crear la gráfica de barras agrupadas para Kappa
ggplot(resultados, aes(x = Modelo, y = Kappa, fill = Modelo)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(Kappa, 3)), vjust = -0.5, position = position_dodge(width = 0.9)) +
  labs(title = "Comparación de Modelos Predictivos",
       x = "Modelo", y = "Kappa") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_discrete(name = "Método de\nDiscretización") 
