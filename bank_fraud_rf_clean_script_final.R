#############################################################
# SYST 568 Applied Predictive Analytics
# Group 2 - Paige La
# Bank Fraud Detection Project - Random Forest - Final Model 
#
# PURPOSE
#   1) Build a general Random Forest fraud model
#   2) Build a smaller device-focused Random Forest model
#   3) 5-fold CV was used inside caret model tuning, together with a stratified holdout test set.
#   4) Interpret variable importance for Transaction_Device
#      and Device_Type
#
# IMPORTANT
# - Start from the RAW CSV file every time.
# - If you have been running pieces of older scripts, restart R first:
#     Session -> Restart R
#############################################################

###############################
# 1. LOAD LIBRARIES
###############################

# caret = class workflow for training, tuning, and resampling
# randomForest = Random Forest algorithm used by caret method = "rf"
# pROC = ROC curve and AUC
library(caret)
library(randomForest)
library(pROC)

###############################
# 2. LOAD DATA
###############################

# Read the raw CSV file fresh so we do not inherit a previously modified df
df <- read.csv("new_bank_fraud_detection.csv",
               stringsAsFactors = FALSE,
               check.names = FALSE)

# Quick import check
cat("Rows:", nrow(df), "\n")
cat("Columns:", ncol(df), "\n")
print(names(df))

# Remove export index column if present
# Some exports create an extra first column such as "Unnamed: 0" or "X"
drop_export_cols <- intersect(names(df), c("", "X"))
if (length(drop_export_cols) > 0) {
  df <- df[, !(names(df) %in% drop_export_cols), drop = FALSE]
  cat("Dropped export index columns:", paste(drop_export_cols, collapse = ", "), "\n")
}

# Remove true identifier columns if present
id_cols <- intersect(names(df), c("Transaction_ID", "Merchant_ID"))
if (length(id_cols) > 0) {
  df <- df[, !(names(df) %in% id_cols), drop = FALSE]
  cat("Dropped identifier columns:", paste(id_cols, collapse = ", "), "\n")
}

# Basic checks
# Three removed vars at this point: Index column (""), Transaction ID, and Merchant ID. 
str(df)
cat("Total missing values:", sum(is.na(df)), "\n")

###############################
# 3. CONVERT RESPONSE VARIABLE
###############################

# Is_Fraud is 0/1 in the raw file.
# Convert to a 2-level factor to categorical labels for readability.
if (!"Is_Fraud" %in% names(df)) {
  stop("Column 'Is_Fraud' was not found in the raw CSV.")
}

df$Is_Fraud <- factor(df$Is_Fraud,
                      levels = c(0, 1),
                      labels = c("NoFraud", "Fraud"))

# Check class balance
cat("\nClass counts:\n")
print(table(df$Is_Fraud))
cat("\nClass proportions:\n")
print(prop.table(table(df$Is_Fraud)))

###############################
# 4. CONVERT CATEGORICAL VARIABLES
###############################

# Keep categorical variables as factors for Random Forest
factor_cols <- c("Gender",
                 "State",
                 "Bank_Branch",
                 "Account_Type",
                 "Transaction_Type",
                 "Merchant_Category",
                 "Transaction_Device",
                 "Device_Type",
                 "Transaction_Currency")

factor_cols <- intersect(factor_cols, names(df))
df[factor_cols] <- lapply(df[factor_cols], factor)

###############################
# 5. FEATURE ENGINEERING
###############################

# 5A. Parse mixed-format dates
# The raw Transaction_Date column uses:
#   - MM/DD/YYYY
#   - DD-MM-YYYY
parse_mixed_date <- function(x) {
  x <- trimws(as.character(x))
  out <- as.Date(rep(NA_character_, length(x)))

  # Try MM/DD/YYYY first
  d1 <- as.Date(x, format = "%m/%d/%Y")
  out[!is.na(d1)] <- d1[!is.na(d1)]

  # Fill remaining values with DD-MM-YYYY
  still_na <- is.na(out)
  if (any(still_na)) {
    d2 <- as.Date(x[still_na], format = "%d-%m-%Y")
    out[still_na] <- d2
  }

  out
}

if (!"Transaction_Date" %in% names(df)) {
  stop("Column 'Transaction_Date' was not found. Restart R and reload the RAW CSV.")
}
df$Parsed_Date <- parse_mixed_date(df$Transaction_Date)

cat("\nParsed_Date class:\n")
print(class(df$Parsed_Date))
cat("Unparsed dates remaining:", sum(is.na(df$Parsed_Date)), "\n")

# 5B. Create time features
if (!"Transaction_Time" %in% names(df)) {
  stop("Column 'Transaction_Time' was not found. Restart R and reload the RAW CSV.")
}

# Month from parsed date
df$txn_month <- as.integer(format(df$Parsed_Date, "%m"))

# Weekday from parsed date
df$txn_wday <- factor(weekdays(df$Parsed_Date),
                      levels = c("Monday", "Tuesday", "Wednesday",
                                 "Thursday", "Friday", "Saturday", "Sunday"))

# Hour from HH:MM:SS
df$txn_hour <- as.integer(substr(df$Transaction_Time, 1, 2))

# Weekday / Weekend flag
df$is_weekend <- factor(ifelse(df$txn_wday %in% c("Saturday", "Sunday"),
                               "Weekend", "Weekday"))

# 5C. Split location field "City, State"
if ("Transaction_Location" %in% names(df)) {
  location_split <- strsplit(as.character(df$Transaction_Location), ",")

  df$Txn_City <- trimws(sapply(location_split, function(z) {
    if (length(z) >= 1) z[1] else NA
  }))

  df$Txn_State <- trimws(sapply(location_split, function(z) {
    if (length(z) >= 2) z[2] else NA
  }))

  df$Txn_City  <- factor(df$Txn_City)
  df$Txn_State <- factor(df$Txn_State)
}

# 5D. Handle Transaction_Description conservatively
# Keep it only if the number of levels is moderate; otherwise drop it
if ("Transaction_Description" %in% names(df)) {
  desc_nlevels <- length(unique(df$Transaction_Description))
  cat("\nUnique Transaction_Description values:", desc_nlevels, "\n")

  if (desc_nlevels <= 50) {
    df$Transaction_Description <- factor(df$Transaction_Description)
  } else {
    df$Transaction_Description <- NULL
    cat("Dropped Transaction_Description because it has over 50 unique values which is too many for this RF workflow.\n")
  }
}

# 5E. Add richer fraud-context features

# Ratio of transaction amount to current balance
# Add 1 to avoid division by zero if balance is ever zero.
df$amount_to_balance_ratio <- df$Transaction_Amount / (df$Account_Balance + 1)

# Night transaction flag
df$is_night <- factor(ifelse(df$txn_hour %in% 0:5, "Night", "NotNight"))

# High-amount flag based on the 95th percentile
high_amt_cut <- quantile(df$Transaction_Amount, probs = 0.95, na.rm = TRUE)
df$is_high_amount <- factor(ifelse(df$Transaction_Amount >= high_amt_cut,
                                   "HighAmount", "NotHighAmount"))

###############################
# 6. DROP RAW COLUMNS
###############################

# Drop raw fields after engineering features from them
drop_after_engineering <- intersect(names(df),
                                    c("Transaction_Date",
                                      "Transaction_Time",
                                      "Parsed_Date",
                                      "Transaction_Location"))

if (length(drop_after_engineering) > 0) {
  df <- df[, !(names(df) %in% drop_after_engineering), drop = FALSE]
  cat("Dropped raw columns after feature engineering:",
      paste(drop_after_engineering, collapse = ", "), "\n")
}

###############################
# 7. BUILD RF-READY DATASET
###############################

# randomForest used by caret::train(method = "rf") struggles with high-cardinality factor variables. Drop those here for a stable RF model.
drop_high_cardinality <- c("Bank_Branch",
                           "Txn_City",
                           "Txn_State",
                           "Transaction_Currency")

drop_high_cardinality <- intersect(drop_high_cardinality, names(df))
rf_df <- df[, !(names(df) %in% drop_high_cardinality), drop = FALSE]

if (length(drop_high_cardinality) > 0) {
  cat("Dropped high-cardinality / unhelpful columns for RF:",
      paste(drop_high_cardinality, collapse = ", "), "\n")
}

# Remove near-zero variance predictors
predictor_df <- rf_df[, !(names(rf_df) %in% "Is_Fraud"), drop = FALSE]
nzv_idx <- nearZeroVar(predictor_df)

if (length(nzv_idx) > 0) {
  removed_nzv <- names(predictor_df)[nzv_idx]
  predictor_df <- predictor_df[, -nzv_idx, drop = FALSE]
  rf_df <- cbind(predictor_df, Is_Fraud = rf_df$Is_Fraud)
  cat("Dropped near-zero-variance predictors:", paste(removed_nzv, collapse = ", "), "\n")
} else {
  cat("No near-zero-variance predictors were removed.\n")
}

# Quick factor-level check
factor_level_counts <- sapply(rf_df, function(x) if (is.factor(x)) nlevels(x) else NA)
cat("\nFactor level counts:\n")
print(factor_level_counts[!is.na(factor_level_counts)])

###############################
# 7B. CREATE TRAIN / TEST SPLIT
###############################

# Create a stratified split so both train and test keep similar fraud proportions.
set.seed(123)
train_index <- createDataPartition(rf_df$Is_Fraud, p = 0.80, list = FALSE)

train_df <- rf_df[train_index, ]
test_df  <- rf_df[-train_index, ]

cat("Training rows:", nrow(train_df), "\n")
cat("Testing rows:", nrow(test_df), "\n")

cat("\nTraining class proportions:\n")
print(prop.table(table(train_df$Is_Fraud)))

cat("\nTesting class proportions:\n")
print(prop.table(table(test_df$Is_Fraud)))

###############################
# 8. TRAIN MODEL - FINAL UPGRADE
###############################

# FINAL UPGRADE GOALS
# 1. Use a manageable but larger training sample than the pilot
# 2. Compare imbalance handling strategies
# 3. Tune mtry using a small grid
# 4. Keep ntree large enough to stabilize results without exceeding memory

# Create a manageable stratified training sample from the TRAINING set only
set.seed(123)

fraud_idx_train   <- which(train_df$Is_Fraud == "Fraud")
nofraud_idx_train <- which(train_df$Is_Fraud == "NoFraud")

# Adjust these sizes if machine still struggles.
general_fraud_n   <- min(2000, length(fraud_idx_train))
general_nofraud_n <- min(10000, length(nofraud_idx_train))

general_sample_idx <- c(
  sample(fraud_idx_train, general_fraud_n),
  sample(nofraud_idx_train, general_nofraud_n)
)

general_train_small <- train_df[general_sample_idx, ]

# General model predictors and response
general_x <- general_train_small[, !(names(general_train_small) %in% "Is_Fraud"), drop = FALSE]
general_y <- general_train_small$Is_Fraud

# Small but real tuning grid for mtry
p_general <- ncol(general_x)
general_grid <- expand.grid(
  mtry = unique(pmax(1, c(floor(sqrt(p_general)) - 1,
                          floor(sqrt(p_general)),
                          floor(sqrt(p_general)) + 1)))
)

# Base trainControl
ctrl_base <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Down-sampling trainControl
ctrl_down <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  sampling = "down"
)

# Up-sampling trainControl
ctrl_up <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  sampling = "up"
)

# Train three general-model variants
# Base trainControl RF model
set.seed(123)
rf_general_base <- train(
  x = general_x,
  y = general_y,
  method = "rf",
  metric = "ROC",
  ntree = 250,
  importance = TRUE,
  tuneGrid = general_grid,
  trControl = ctrl_base
)

# Down-sampling trainControl RF model
set.seed(123)
rf_general_down <- train(
  x = general_x,
  y = general_y,
  method = "rf",
  metric = "ROC",
  ntree = 250,
  importance = TRUE,
  tuneGrid = general_grid,
  trControl = ctrl_down
)

# Up-sampling trainControl RF model
set.seed(123)
rf_general_up <- train(
  x = general_x,
  y = general_y,
  method = "rf",
  metric = "ROC",
  ntree = 250,
  importance = TRUE,
  tuneGrid = general_grid,
  trControl = ctrl_up
)

cat("\n===== GENERAL MODEL COMPARISON =====\n")
print(rf_general_base)
print(rf_general_down)
print(rf_general_up)


##########################################################
# 8B. DEVICE-FOCUSED MODEL - FINAL UPGRADE
##########################################################

device_vars <- c("Transaction_Device",
                 "Device_Type",
                 "Transaction_Amount",
                 "Transaction_Type",
                 "Merchant_Category",
                 "Account_Balance",
                 "txn_hour",
                 "txn_month",
                 "txn_wday",
                 "is_weekend",
                 "amount_to_balance_ratio",
                 "is_night",
                 "is_high_amount",
                 "Age",
                 "Gender",
                 "Is_Fraud")

device_vars <- intersect(device_vars, names(train_df))
device_train <- train_df[, device_vars, drop = FALSE]

# Manageable stratified sample from the training set
set.seed(123)

fraud_idx_d   <- which(device_train$Is_Fraud == "Fraud")
nofraud_idx_d <- which(device_train$Is_Fraud == "NoFraud")

device_fraud_n   <- min(2000, length(fraud_idx_d))
device_nofraud_n <- min(10000, length(nofraud_idx_d))

device_sample_idx <- c(
  sample(fraud_idx_d, device_fraud_n),
  sample(nofraud_idx_d, device_nofraud_n)
)

device_train_small <- device_train[device_sample_idx, ]

device_x <- device_train_small[, !(names(device_train_small) %in% "Is_Fraud"), drop = FALSE]
device_y <- device_train_small$Is_Fraud

p_device <- ncol(device_x)
device_grid <- expand.grid(
  mtry = unique(pmax(1, c(floor(sqrt(p_device)) - 1,
                          floor(sqrt(p_device)),
                          floor(sqrt(p_device)) + 1)))
)

# Train three device-focused variants - base, down sampling, and upsampling.
set.seed(123)
rf_device_base <- train(
  x = device_x,
  y = device_y,
  method = "rf",
  metric = "ROC",
  ntree = 250,
  importance = TRUE,
  tuneGrid = device_grid,
  trControl = ctrl_base
)

set.seed(123)
rf_device_down <- train(
  x = device_x,
  y = device_y,
  method = "rf",
  metric = "ROC",
  ntree = 250,
  importance = TRUE,
  tuneGrid = device_grid,
  trControl = ctrl_down
)

set.seed(123)
rf_device_up <- train(
  x = device_x,
  y = device_y,
  method = "rf",
  metric = "ROC",
  ntree = 250,
  importance = TRUE,
  tuneGrid = device_grid,
  trControl = ctrl_up
)

cat("\n===== DEVICE-FOCUSED MODEL COMPARISON =====\n")
print(rf_device_base)
print(rf_device_down)
print(rf_device_up)

###############################
# 8C. SELECT BEST MODELS
###############################

# Compare general-model variants by best ROC
general_results <- rbind(
  data.frame(Model = "General_Base", BestROC = max(rf_general_base$results$ROC)),
  data.frame(Model = "General_Down", BestROC = max(rf_general_down$results$ROC)),
  data.frame(Model = "General_Up",   BestROC = max(rf_general_up$results$ROC))
)

device_results <- rbind(
  data.frame(Model = "Device_Base", BestROC = max(rf_device_base$results$ROC)),
  data.frame(Model = "Device_Down", BestROC = max(rf_device_down$results$ROC)),
  data.frame(Model = "Device_Up",   BestROC = max(rf_device_up$results$ROC))
)

print(general_results)
print(device_results)

# IMPORTANT: Check printed ROC comparison table before setting these.
# Manually set the winning models after checking the printed table
rf_general_final <- rf_general_base   # change if another one wins
rf_device_final  <- rf_device_down    # change if another one wins

# See the CV predictions 
plot(varImp(rf_general_final))
rf_general_final$pred
# See where fraud was predicted
subset(rf_general_final$pred, pred == "Fraud")
# Compare wrong predictions 
subset(rf_general_final$pred, pred != obs)

# Predictions on the new test set.
general_test_pred <- predict(rf_general_final, newdata = test_x_general)
general_test_prob <- predict(rf_general_final, newdata = test_x_general, type = "prob")
results <- data.frame(
  Actual = test_y_general,
  Predicted = general_test_pred,
  Prob_Fraud = general_test_prob[, "Fraud"]
)
head(results)

# Distribution of predicted probabilities
hist(results$Prob_Fraud,
     main = "Predicted Fraud Probabilities",
     xlab = "Probability of Fraud")

#Fraud vs Non-Fraud distributions
boxplot(Prob_Fraud ~ Actual,
        data = results,
        main = "Predicted Probabilities by Actual Class")

###############################
# 9. EVALUATE FINAL SELECTED MODELS
###############################

##########################################################
# 9A. GENERAL MODEL - CV SUMMARY
##########################################################

# CV mean and SD from resampling
general_cv_summary <- data.frame(
  ROC_Mean  = mean(rf_general_final$resample$ROC),
  ROC_SD    = sd(rf_general_final$resample$ROC),
  Sens_Mean = mean(rf_general_final$resample$Sens),
  Sens_SD   = sd(rf_general_final$resample$Sens),
  Spec_Mean = mean(rf_general_final$resample$Spec),
  Spec_SD   = sd(rf_general_final$resample$Spec)
)

print(general_cv_summary)

# Out-of-fold predictions for best mtry
general_pred_cv <- rf_general_final$pred
general_best_mtry <- rf_general_final$bestTune$mtry
general_pred_cv <- general_pred_cv[general_pred_cv$mtry == general_best_mtry, ]

general_pred_cv$pred <- factor(general_pred_cv$pred, levels = levels(general_y))
general_pred_cv$obs  <- factor(general_pred_cv$obs,  levels = levels(general_y))

general_cm <- confusionMatrix(
  data = general_pred_cv$pred,
  reference = general_pred_cv$obs,
  positive = "Fraud"
)

general_roc <- roc(
  response = general_pred_cv$obs,
  predictor = general_pred_cv$Fraud,
  levels = c("NoFraud", "Fraud"),
  direction = "<"
)

general_precision <- general_cm$byClass["Pos Pred Value"]
general_recall    <- general_cm$byClass["Sensitivity"]
general_f1        <- 2 * (general_precision * general_recall) /
  (general_precision + general_recall)

cat("\n===== GENERAL MODEL CV RESULTS =====\n")
print(general_cm)
cat("Best mtry:", general_best_mtry, "\n")
cat("AUC:", as.numeric(auc(general_roc)), "\n")
cat("Precision:", as.numeric(general_precision), "\n")
cat("Recall:", as.numeric(general_recall), "\n")
cat("F1 Score:", as.numeric(general_f1), "\n")

##########################################################
# 9B. GENERAL MODEL - TEST SET EVALUATION
##########################################################

# Predict on the holdout test set
test_x_general <- test_df[, !(names(test_df) %in% "Is_Fraud"), drop = FALSE]
test_y_general <- test_df$Is_Fraud

general_test_pred <- predict(rf_general_final, newdata = test_x_general)
general_test_prob <- predict(rf_general_final, newdata = test_x_general, type = "prob")[, "Fraud"]

general_test_cm <- confusionMatrix(
  data = factor(general_test_pred, levels = levels(test_y_general)),
  reference = test_y_general,
  positive = "Fraud"
)

general_test_roc <- roc(
  response = test_y_general,
  predictor = general_test_prob,
  levels = c("NoFraud", "Fraud"),
  direction = "<"
)

general_test_precision <- general_test_cm$byClass["Pos Pred Value"]
general_test_recall    <- general_test_cm$byClass["Sensitivity"]
general_test_f1        <- 2 * (general_test_precision * general_test_recall) /
  (general_test_precision + general_test_recall)

cat("\n===== GENERAL MODEL TEST RESULTS =====\n")
print(general_test_cm)
cat("Test AUC:", as.numeric(auc(general_test_roc)), "\n")
cat("Test Precision:", as.numeric(general_test_precision), "\n")
cat("Test Recall:", as.numeric(general_test_recall), "\n")
cat("Test F1:", as.numeric(general_test_f1), "\n")

##########################################################
# 9C. DEVICE-FOCUSED MODEL - CV SUMMARY
##########################################################

# CV mean and SD from resampling
device_cv_summary <- data.frame(
  ROC_Mean  = mean(rf_device_final$resample$ROC),
  ROC_SD    = sd(rf_device_final$resample$ROC),
  Sens_Mean = mean(rf_device_final$resample$Sens),
  Sens_SD   = sd(rf_device_final$resample$Sens),
  Spec_Mean = mean(rf_device_final$resample$Spec),
  Spec_SD   = sd(rf_device_final$resample$Spec)
)

print(device_cv_summary)

# Out-of-fold predictions for best mtry
device_pred_cv <- rf_device_final$pred
device_best_mtry <- rf_device_final$bestTune$mtry
device_pred_cv <- device_pred_cv[device_pred_cv$mtry == device_best_mtry, ]

device_pred_cv$pred <- factor(device_pred_cv$pred, levels = levels(device_y))
device_pred_cv$obs  <- factor(device_pred_cv$obs,  levels = levels(device_y))

device_cm <- confusionMatrix(
  data = device_pred_cv$pred,
  reference = device_pred_cv$obs,
  positive = "Fraud"
)

device_roc <- roc(
  response = device_pred_cv$obs,
  predictor = device_pred_cv$Fraud,
  levels = c("NoFraud", "Fraud"),
  direction = "<"
)

device_precision <- device_cm$byClass["Pos Pred Value"]
device_recall    <- device_cm$byClass["Sensitivity"]
device_f1        <- 2 * (device_precision * device_recall) /
  (device_precision + device_recall)

cat("\n===== DEVICE-FOCUSED MODEL CV RESULTS =====\n")
print(device_cm)
cat("Best mtry:", device_best_mtry, "\n")
cat("AUC:", as.numeric(auc(device_roc)), "\n")
cat("Precision:", as.numeric(device_precision), "\n")
cat("Recall:", as.numeric(device_recall), "\n")
cat("F1 Score:", as.numeric(device_f1), "\n")

##########################################################
# 9D. DEVICE-FOCUSED MODEL - TEST SET EVALUATION
##########################################################

# Build a test set with the same predictor columns used in the device-focused model
device_test_vars <- c("Transaction_Device",
                      "Device_Type",
                      "Transaction_Amount",
                      "Transaction_Type",
                      "Merchant_Category",
                      "Account_Balance",
                      "txn_hour",
                      "txn_month",
                      "txn_wday",
                      "is_weekend",
                      "amount_to_balance_ratio",
                      "is_night",
                      "is_high_amount",
                      "Age",
                      "Gender",
                      "Is_Fraud")

device_test_vars <- intersect(device_test_vars, names(test_df))
device_test <- test_df[, device_test_vars, drop = FALSE]

test_x_device <- device_test[, !(names(device_test) %in% "Is_Fraud"), drop = FALSE]
test_y_device <- device_test$Is_Fraud

device_test_pred <- predict(rf_device_final, newdata = test_x_device)
device_test_prob <- predict(rf_device_final, newdata = test_x_device, type = "prob")[, "Fraud"]

device_test_cm <- confusionMatrix(
  data = factor(device_test_pred, levels = levels(test_y_device)),
  reference = test_y_device,
  positive = "Fraud"
)

device_test_roc <- roc(
  response = test_y_device,
  predictor = device_test_prob,
  levels = c("NoFraud", "Fraud"),
  direction = "<"
)

device_test_precision <- device_test_cm$byClass["Pos Pred Value"]
device_test_recall    <- device_test_cm$byClass["Sensitivity"]
device_test_f1        <- 2 * (device_test_precision * device_test_recall) /
  (device_test_precision + device_test_recall)

cat("\n===== DEVICE-FOCUSED MODEL TEST RESULTS =====\n")
print(device_test_cm)
cat("Test AUC:", as.numeric(auc(device_test_roc)), "\n")
cat("Test Precision:", as.numeric(device_test_precision), "\n")
cat("Test Recall:", as.numeric(device_test_recall), "\n")
cat("Test F1:", as.numeric(device_test_f1), "\n")

###############################
# 10. PLOTS + INTERPRETATION
###############################

##########################################################
# 10A. ROC CURVE
##########################################################

plot(
  general_roc,
  col = "blue",
  lwd = 2,
  main = "ROC Curve - General Random Forest (Final Selected Model)"
)

##########################################################
# 10B. VARIABLE IMPORTANCE
##########################################################

general_varimp <- varImp(rf_general_final, scale = FALSE)

cat("\n===== VARIABLE IMPORTANCE (GENERAL MODEL) =====\n")
print(general_varimp)

plot(
  general_varimp,
  top = 10,
  main = "Top Predictors - General Model"
)

##########################################################
# 10C. FRAUD PROBABILITY BY DEVICE TYPE
##########################################################

general_prob <- predict(rf_general_final, newdata = general_x, type = "prob")[, "Fraud"]

device_prob_summary <- aggregate(
  general_prob,
  by = list(Device_Type = general_train_small$Device_Type),
  FUN = mean
)

names(device_prob_summary)[2] <- "Mean_Fraud_Prob"

barplot(
  device_prob_summary$Mean_Fraud_Prob,
  names.arg = device_prob_summary$Device_Type,
  las = 2,
  col = "steelblue",
  main = "Predicted Fraud Probability by Device Type in General Model",
  ylab = "Average Predicted Fraud Probability"
)

##########################################################
# 10D. DEVICE-FOCUSED ROC CURVE
##########################################################

plot(
  device_roc,
  col = "darkgreen",
  lwd = 2,
  main = "ROC Curve - Device-Focused Random Forest (Final Selected Model)"
)

##########################################################
# 10E. DEVICE-FOCUSED VARIABLE IMPORTANCE
##########################################################

device_varimp <- varImp(rf_device_final, scale = FALSE)

cat("\n===== VARIABLE IMPORTANCE (DEVICE-FOCUSED MODEL) =====\n")
print(device_varimp)

plot(
  device_varimp,
  top = 10,
  main = "Top Predictors - Device-Focused Model"
)

##########################################################
# 10F. DEVICE-FOCUSED FRAUD PROBABILITY BY DEVICE TYPE
##########################################################

# Use the DEVICE-FOCUSED final model and DEVICE-FOCUSED data
device_prob <- predict(rf_device_final, newdata = device_x, type = "prob")[, "Fraud"]

device_prob_summary <- aggregate(
  device_prob,
  by = list(Device_Type = device_train_small$Device_Type),
  FUN = mean
)

names(device_prob_summary)[2] <- "Mean_Fraud_Prob"

device_prob_summary <- device_prob_summary[order(device_prob_summary$Mean_Fraud_Prob), ]

barplot(
  device_prob_summary$Mean_Fraud_Prob,
  names.arg = device_prob_summary$Device_Type,
  las = 2,
  col = "darkgreen",
  main = "Predicted Fraud Probability by Device Type in Device-Focused Model",
  ylab = "Average Predicted Fraud Probability"
)

##########################################################
# 10G. PARTIAL DEPENDENCE - DEVICE VARIABLES
##########################################################

if ("Transaction_Device" %in% names(device_x)) {
  randomForest::partialPlot(
    x = rf_device_final$finalModel,
    pred.data = device_x,
    x.var = "Transaction_Device",
    which.class = "Fraud",
    main = "Partial Dependence - Transaction_Device",
    xlab = "Transaction_Device",
    ylab = "Centered logit / relative fraud tendency"
  )
}

if ("Device_Type" %in% names(device_x)) {
  randomForest::partialPlot(
    x = rf_device_final$finalModel,
    pred.data = device_x,
    x.var = "Device_Type",
    which.class = "Fraud",
    main = "Partial Dependence - Device_Type",
    xlab = "Device_Type",
    ylab = "Centered logit / relative fraud tendency"
  )
}
