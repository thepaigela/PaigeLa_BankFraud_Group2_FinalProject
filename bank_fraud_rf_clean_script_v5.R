#############################################################
# SYST 568 Applied Predictive Analytics
# Group 2 - Paige La
# Bank Fraud Detection Project - Random Forest Model. 
# 
#
# PURPOSE
#   1) Build a general Random Forest fraud model
#   2) Build a smaller device-focused Random Forest model
#   3) Evaluate both models with repeated 10-fold CV
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
# 8. TRAIN MODEL
###############################

# The full dataset was too large for the current machine when using repeated CV, multiple tuning values, and many trees.
# I changed the code to have 5-fold CV instead of repeated CV, smaller stratified samples, one mtry value, and fewer trees.
# This allows the model to finish and still gives useful pilot results.

ctrl_midterm <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

##########################################################
# 8A. GENERAL MODEL - small stratified sample
##########################################################

set.seed(123)

fraud_idx   <- which(rf_df$Is_Fraud == "Fraud")
nofraud_idx <- which(rf_df$Is_Fraud == "NoFraud")

# Small pilot sample for the general model
general_fraud_n   <- min(1000, length(fraud_idx))
general_nofraud_n <- min(5000, length(nofraud_idx))

general_sample_idx <- c(
  sample(fraud_idx, general_fraud_n),
  sample(nofraud_idx, general_nofraud_n)
)

generalData_small <- rf_df[general_sample_idx, ]

general_x <- generalData_small[, !(names(generalData_small) %in% "Is_Fraud"), drop = FALSE]
general_y <- generalData_small$Is_Fraud

# Use one mtry value to reduce total model fits
p_general <- ncol(general_x)
general_grid <- expand.grid(
  mtry = floor(sqrt(p_general))
)

set.seed(123)
rf_general <- train(
  x = general_x,
  y = general_y,
  method = "rf",
  metric = "ROC",
  ntree = 150,
  importance = TRUE,
  tuneGrid = general_grid,
  trControl = ctrl_midterm
)

cat("\n================ GENERAL MODEL (MIDTERM PILOT) ================\n")
print(rf_general)

##########################################################
# 8B. DEVICE-FOCUSED MODEL - slightly larger sample
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
                 "Age",
                 "Gender",
                 "Is_Fraud")

device_vars <- intersect(device_vars, names(rf_df))
deviceData <- rf_df[, device_vars, drop = FALSE]

set.seed(123)

fraud_idx_d   <- which(deviceData$Is_Fraud == "Fraud")
nofraud_idx_d <- which(deviceData$Is_Fraud == "NoFraud")

# Device-focused sample can be a bit larger because it uses fewer predictors
device_fraud_n   <- min(1500, length(fraud_idx_d))
device_nofraud_n <- min(8500, length(nofraud_idx_d))

device_sample_idx <- c(
  sample(fraud_idx_d, device_fraud_n),
  sample(nofraud_idx_d, device_nofraud_n)
)

deviceData_small <- deviceData[device_sample_idx, ]

device_x <- deviceData_small[, !(names(deviceData_small) %in% "Is_Fraud"), drop = FALSE]
device_y <- deviceData_small$Is_Fraud

p_device <- ncol(device_x)
device_grid <- expand.grid(
  mtry = floor(sqrt(p_device))
)

set.seed(123)
rf_device <- train(
  x = device_x,
  y = device_y,
  method = "rf",
  metric = "ROC",
  ntree = 150,
  importance = TRUE,
  tuneGrid = device_grid,
  trControl = ctrl_midterm
)

cat("\n================ DEVICE-FOCUSED MODEL (MIDTERM PILOT) ================\n")
print(rf_device)

###############################
# 9. EVALUATE MODEL (PILOT SETUP)
###############################

# We are evaluating using cross-validation predictions from a smaller pilot model.
# Results should be interpreted as preliminary due to reduced data size and lighter training settings.

##########################################################
# 9A. GENERAL MODEL EVALUATION
##########################################################

# Extract CV predictions
general_pred_cv <- rf_general$pred

# Keep only best mtry results
general_best_mtry <- rf_general$bestTune$mtry
general_pred_cv <- general_pred_cv[general_pred_cv$mtry == general_best_mtry, ]

# Ensure factor levels match
general_pred_cv$pred <- factor(general_pred_cv$pred, levels = levels(general_y))
general_pred_cv$obs  <- factor(general_pred_cv$obs,  levels = levels(general_y))

# Confusion Matrix
general_cm <- confusionMatrix(
  data = general_pred_cv$pred,
  reference = general_pred_cv$obs,
  positive = "Fraud"
)

# ROC Curve
general_roc <- roc(
  response = general_pred_cv$obs,
  predictor = general_pred_cv$Fraud,
  levels = c("NoFraud", "Fraud"),
  direction = "<"
)

# Metrics
general_precision <- general_cm$byClass["Pos Pred Value"]
general_recall    <- general_cm$byClass["Sensitivity"]
general_f1        <- 2 * (general_precision * general_recall) /
  (general_precision + general_recall)

cat("\n===== GENERAL MODEL RESULTS (PILOT) =====\n")
print(general_cm)
cat("Best mtry:", general_best_mtry, "\n")
cat("AUC:", as.numeric(auc(general_roc)), "\n")
cat("Precision:", as.numeric(general_precision), "\n")
cat("Recall:", as.numeric(general_recall), "\n")
cat("F1 Score:", as.numeric(general_f1), "\n")

# Interpretation guidance
cat("\nINTERPRETATION:\n")
cat("- ROC close to 0.5 indicates weak separation ability\n")
cat("- High recall with low specificity suggests over-predicting fraud\n")
cat("- Model likely influenced by class imbalance and reduced dataset size\n")

##########################################################
# 9B. DEVICE-FOCUSED MODEL EVALUATION
##########################################################

# Extract CV predictions from the device-focused model
device_pred_cv <- rf_device$pred

# Keep only the rows that match the best mtry chosen by caret
device_best_mtry <- rf_device$bestTune$mtry
device_pred_cv <- device_pred_cv[device_pred_cv$mtry == device_best_mtry, ]

# Make sure factor levels match the original response levels
device_pred_cv$pred <- factor(device_pred_cv$pred, levels = levels(device_y))
device_pred_cv$obs  <- factor(device_pred_cv$obs,  levels = levels(device_y))

# Confusion Matrix
device_cm <- confusionMatrix(
  data = device_pred_cv$pred,
  reference = device_pred_cv$obs,
  positive = "Fraud"
)

# ROC Curve / AUC
device_roc <- roc(
  response = device_pred_cv$obs,
  predictor = device_pred_cv$Fraud,
  levels = c("NoFraud", "Fraud"),
  direction = "<"
)

# Precision, Recall, and F1
device_precision <- device_cm$byClass["Pos Pred Value"]
device_recall    <- device_cm$byClass["Sensitivity"]
device_f1        <- 2 * (device_precision * device_recall) /
  (device_precision + device_recall)

cat("\n===== DEVICE-FOCUSED MODEL RESULTS (PILOT) =====\n")
print(device_cm)
cat("Best mtry:", device_best_mtry, "\n")
cat("AUC:", as.numeric(auc(device_roc)), "\n")
cat("Precision:", as.numeric(device_precision), "\n")
cat("Recall:", as.numeric(device_recall), "\n")
cat("F1 Score:", as.numeric(device_f1), "\n")

# Interpretation guidance
cat("\nINTERPRETATION:\n")
cat("- ROC close to 0.5 indicates weak separation ability\n")
cat("- High recall with very low specificity suggests over-predicting fraud\n")
cat("- Device-related predictors alone may not provide enough signal to distinguish fraud well\n")

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
  main = "ROC Curve - General Random Forest (Pilot)"
)

##########################################################
# 10B. VARIABLE IMPORTANCE
##########################################################

general_varimp <- varImp(rf_general, scale = FALSE)

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

general_prob <- predict(rf_general, newdata = general_x, type = "prob")[, "Fraud"]

device_prob_summary <- aggregate(
  general_prob,
  by = list(Device_Type = generalData_small$Device_Type),
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
  main = "ROC Curve - Device-Focused Random Forest (Pilot)"
)

##########################################################
# 10E. DEVICE-FOCUSED VARIABLE IMPORTANCE
##########################################################

device_varimp <- varImp(rf_device, scale = FALSE)

cat("\n===== VARIABLE IMPORTANCE (DEVICE-FOCUSED MODEL) =====\n")
print(device_varimp)

plot(
  device_varimp,
  top = 10,
  main = "Top Predictors - Device-Focused Model"
)

##########################################################
# 10F. FRAUD PROBABILITY BY DEVICE TYPE
##########################################################

general_prob <- predict(rf_device, newdata = device_x, type = "prob")[, "Fraud"]

device_prob_summary <- aggregate(
  general_prob,
  by = list(Device_Type = deviceData_small$Device_Type),
  FUN = mean
)

names(device_prob_summary)[2] <- "Mean_Fraud_Prob"

barplot(
  device_prob_summary$Mean_Fraud_Prob,
  names.arg = device_prob_summary$Device_Type,
  las = 2,
  col = "steelblue",
  main = "Predicted Fraud Probability by Device Type in Device Model",
  ylab = "Average Predicted Fraud Probability"
)
