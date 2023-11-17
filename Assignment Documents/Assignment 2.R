---
title: "Assignment 1 Computer based assignment"
output: pdf_document
date: "2023-10-24"
author: "Manav Saini (Student ID: 20608893)"
---

# Installing the necessary packages: 
library(readxl)
library(dplyr)
library(tidyverse)
library(tidyr)
library(ggplot2)
library(tree)

## Mutual Funds India detailed Dataset (Source Kaggle)
  
# The data set is downloaded from Kaggle.
# This is a comprehensive data set about Indian mutual funds which contains a 
# mix of categorical as well as numerical variables providing an opportunity to 
#go in depth and explore as well as predict the relationships and risk factors 
#that these investment vehicles present.
  

#Read the file and store it in the variable named CD.
CD = read.csv("/Users/manavsaini/Downloads/comprehensive_mutual_funds_data.csv")


## Checking the first few rows.

# As observed, the data 20 variables of both numeric and categorical variables. 
#15 of these variables are numerical variables and 10 are continues data values
#more suitable towards the regression based tasks.


head(CD)


## Checking the attributes using the "str" function in R.

# The file has 20 variables of 814 observations. It is rich enough to predict 
#and use various regression as well as classification techniques.


str(CD)


## Converting Categorical to Numerical.


# Create a categorical variable
#We selectively converted the categorical variables into factors.
CD$amc_name = as.factor(CD$amc_name)
CD$category = as.factor(CD$category)
CD$sub_category = as.factor(CD$sub_category)

#checking if the variables have been converted to factors successfully.
str(CD)

## Potential research questions:

# Regression Analysis: Predicting Riskiness of Mutual Funds: Can we predict 
#if a mutual fund policy is risky based on attributes like "expense_ratio,
#" "fund_size_cr," "fund_age_yr," "sortino," "alpha," "sd," "beta," "sharpe," 
#"amc_name," "rating," and "category"? This analysis could help investors make informed decisions when choosing mutual funds.
#Classification Analysis: Categorizing Mutual Funds: Can we classify mutual funds 
#into categories like "Low Risk," "Medium Risk," and "High Risk" based on the given attributes. 
#This classification can help investors choose funds that align with their risk tolerance.

## Checking the N/a or Nan values.
#missing values 

#Visualizing the data: 
#Exploring the relationships between the various variables: 

# Select only the numerical columns for pivot_longer
numerical_selection = CD %>%
  select(expense_ratio, sortino, alpha, sd, beta, sharpe,
         fund_size_cr, returns_1yr, returns_3yr, returns_5yr)

plot(numerical_selection)

summary(numerical_selection)

CD1 = CD

summary(CD1)
str(CD1)
#storing the main data frame CD into CD1 as we would be performing several
#manipulations on this and want the original data frame to be untouched. 

#On diagnostic the code, it seems like the code is not numeric in certain 
#variables hence we do so. 
# List of columns that need to be converted from character to numeric
numeric_columns = c('sortino', 'alpha', 'sd', 'beta', 'sharpe')

# Applying the conversion on each column
for(col in numeric_columns) {
  CD1[[col]] = as.numeric(as.character(CD1[[col]]))
}

# Check the structure of CD1 to confirm the changes
str(CD1)

# Assuming these columns are continuous
continuous_cols = c('expense_ratio', 'returns_1yr', 'returns_3yr', 'returns_5yr')
CD1[continuous_cols] = lapply(CD1[continuous_cols], as.numeric)

# Alternatively, using subset()
CD1 = subset(CD1, select = -c(scheme_name, fund_manager, amc_name, sub_category))

# Check the structure of CD1 to confirm the removal
str(CD1)

#Using Tidy format: 
# Load the tidyverse package
library(tidyverse)

#Finding the NA values for each feature: 

na_count = sapply(CD1, function(x) sum(is.na(x)))
na_count

#We observe that some features have "-" being populated. 

# Count the number of hyphen entries ('-') in each column
hyphen_count = sapply(CD1, function(x) sum(x == "-"))
hyphen_count

#replacing "-" with NA for standardizing the data as we would deal with the NA
#values altogether later on. 
CD1[CD1 == "-"] = NA

# Calculate the percentage of NA values in each column
na_percentage = sapply(CD1, function(x) sum(is.na(x)) / nrow(CD1) * 100)
na_percentage

#Checking the distribution of returns_5yr as to see how to deal with the NA 
#values

# To plot a histogram to check the distribution of 'returns_5yr'
hist(CD1$returns_5yr, main = "Histogram of returns_5yr", xlab = "returns_5yr")

#As shown in the histogram, the distribution of the return_5yr is normal hence 
#imputing these values as mean would be an appropriate choice as the mean is 
#a good measure of central tendency for normally distributed data as it represents
#the average of the values.

# Mean imputation for 'returns_5yr'
CD1$returns_5yr[is.na(CD1$returns_5yr)] = mean(CD1$returns_5yr, na.rm = TRUE)

# Remove rows where 'NA' values are in columns with low NA percentages
CD1 = CD1[!is.na(CD1$sortino) & !is.na(CD1$sd) & !is.na(CD1$sharpe) 
         & !is.na(CD1$returns_3yr), ]
CD1 = na.omit(CD1)

# Calculate the percentage of NA values in each column again.
na_percentage = sapply(CD1, function(x) sum(is.na(x)) / nrow(CD1) * 100)
na_percentage

#We have now successfully dealt with all the NA values till this point. 

str(CD1)

#Now we proceed to check the class imbalance in the target variable "risk level". 


# Check class distribution for 'risk_level'
risk_level_distribution = table(CD1$risk_level)


# Removing 'amc_name' and 'sub_category' from CD1
CD1 = select(CD1, -amc_name,-sub_category)

# Print the distribution

# risk_level_distribution 
# > risk_level_distribution
# 
# 1   2   3   4   5   6 
# 53 110 119  65  27 415 

#As shown in the results, there is a clear imbalance in the classes with the 
#risk level of class 6 being far higher than the ones in the rest of the classes 
#hence we need to deal with the class imbalance before we could train our models 
#as the class imbalance of the target variable will largely affect the predictions 
#of our model. 

#to fix the class imbalance, we use re sampling, specifically a modified version
#of bootstrapping called "ROSE" which helps perform random oversampling of the majority 
#class or minor under sampling of the minority class. 

install.packages("ROSE")
library(ROSE)
#on hold for now. 

# 
# set.seed(123) # Set a random seed for reproducibility
# 
# # Apply the ovun.sample function to balance the classes
# balanced_data = ovun.sample(risk_level ~ ., data = CD1, method = "both", 
#                             N = 119*6)$data

# # Check the new distribution
# table(balanced_data$risk_level)

#since we want to predict if the mutual fund is a "low risk" or a "high risk"
#investment, we can convert our risk_level feature into a binary class by 
#taking all the risk levels below and equal to 3 as low risk a above 3 to be 
#high risk investments. 

#but first we need to check if indeed the low risk is below 3 and high risk is 
#above 3 and not the other way around. 

# Aggregate and summarize key variables by risk level
risk_summary = CD1 %>%
  group_by(risk_level) %>%
  summarise(
    avg_return_1yr = mean(returns_1yr, na.rm = TRUE),
    avg_return_3yr = mean(returns_3yr, na.rm = TRUE),
    avg_return_5yr = mean(returns_5yr, na.rm = TRUE),
    avg_sd = mean(as.numeric(sd), na.rm = TRUE)
  )

# View the summary statistics for each risk level
print(risk_summary)

#We observe that the high risk level of 5 and 6 have indeed a higher avg_return 
#and avg standard deviation. Hence using our domain knowledge,
#we can confirm that the high risk is more than 3 and lower risk is less than 3 
#as it aligns with the common financial principal that higher potential rewards
#come with higher potential risks.

  
# Re-code 'risk_level' to a binary variable
CD1$risk_binary = ifelse(CD1$risk_level <= 3, 'Low Risk', 'High Risk')

#Now the risk_binary has been added to the data. 

#checking the distribution of the target variable risk_level again: 
#Here the threshold and creation of the new variables have been done. 
risk_level_distribution = table(CD1$risk_binary)
print(risk_level_distribution)

#Converting the new variable risk_binary to a factor. 
CD1$risk_binary = as.factor(CD1$risk_binary)

#removing risk_level as we have derived the target variable risk_binary from 
#it so that only one variable is used as the predictor. 
library(dplyr)
CD1 = select(CD1, -risk_level)

# 
# High Risk  Low Risk 
# 488       282 
#we observe that the high risk is still more than the low risk which might interfere
#in the model training hence we may proceed to use the re sampling method bootstrap
#to resolve the class imbalance. #We can also use the decision trees such as 
#classification tree as they tend to be less sensitive to class imbalances as compared
#to SVM and Logistic regression. 


# I tried to run the decision tree but receiving this error:
# Error in tree(risk_binary ~ ., data = Trainset) : 
#   factor predictors must have at most 32 levels
# In addition: Warning messages:
#   1: In tree(risk_binary ~ ., data = Trainset) : NAs introduced by coercion
# 2: In tree(risk_binary ~ ., data = Trainset) : NAs introduced by coercion

#Since amc_name and sub_category are not required and creating issues in the 
#decision tree model, we will remove them.  

# You can check the structure of CD1 to confirm the removal
str(CD1)


#Train/test split: 
set.seed(123)
# Calculating the number of rows for the training set (80% of the data)
trainSize = floor(0.8 * nrow(CD1))

# Sampling indices for the training set
trainIndex = sample(1:nrow(CD1), trainSize)

# Creating the training set
Trainset = CD1[trainIndex, ]

# Creating the testing set
Testset = CD1[-trainIndex, ]

# # Identify factor variables with more than 32 levels
# sapply(Trainset, function(x) if(is.factor(x)) length(levels(x)) else NA)

#Decision trees:
# Fitting a Regression Tree to the training set
reg_tree = tree(risk_binary ~ ., data = Trainset)

# Trainset$risk_binary = as.factor(Trainset$risk_binary)
# Testset$risk_binary = as.factor(Testset$risk_binary)

# Plotting the tree
plot(reg_tree)
text(reg_tree, pretty = 0)

summary(reg_tree)

#Constructing the cross validation plot: 
set.seed(1)
cv_regtree = cv.tree(reg_tree)
cv_regtree

#Plotting: 
plot(cv_regtree$size, cv_regtree$dev, type = "b")


#SVM : 
library(e1071)

set.seed(123)  # For reproducibility
tune_model = tune(svm, risk_binary ~ ., data = Trainset, kernel="linear", scale = TRUE,
                  ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100, 1000)))

summary(tune_model)






