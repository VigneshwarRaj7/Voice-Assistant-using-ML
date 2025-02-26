# Load required libraries

library(tidyr)
library(caret)
library(rpart)
library(rpart.plot)
library(ggplot2)
# Load dataset
data <- read.csv("Downloads/college/engingeering project/codes/mh.csv", header = TRUE)

# Assign column names
colnames(data) <- c('Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate', 'RiskLevel')

# Remove 'seq_name' column
data <- data[, -1]

# Encode 'site' column
data$site <- as.factor(data$site)

# Define features and target variable
X <- data[, -6]
Y <- data[, 6]

# Feature selection
selected_features <- caret::nearZeroVar(X)
X <- X[, -selected_features]

# Exploratory Data Analysis (EDA)
summary(data)

plot(data)

# Train Decision Tree model using cross-validation
model <- train(X, Y, method = "rpart", trControl = trainControl(method = "cv", number = 3))

# Visualize decision tree before pruning
rpart.plot(model$finalModel)

# Evaluate model performance before pruning
predictions_before_pruning <- predict(model, X)
accuracy_before_pruning <- confusionMatrix(predictions_before_pruning, Y)$overall['Accuracy']
cat("Accuracy before pruning:", accuracy_before_pruning, "\n")

# Prune decision tree
pruned_model <- prune(model$finalModel, cp = 0.10)  # Adjust the cp value as needed

# Visualize pruned decision tree
rpart.plot(pruned_model)


#View(predictions_after_pruning)
# Evaluate model performance after pruning
predictions_after_pruning <- predict(pruned_model, X)
accuracy_after_pruning <- confusionMatrix(predictions_after_pruning, Y)$overall['Accuracy']
cat("Accuracy after pruning:", accuracy_after_pruning, "\n")