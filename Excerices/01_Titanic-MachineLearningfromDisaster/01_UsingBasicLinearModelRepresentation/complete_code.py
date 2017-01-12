#!/usr/bin/python
#Initialize python environment. 
#Since python executable binary in my system is in location /usr/bin/ it is as follows

#import the packages required to read and manipulate our test data
import pandas as pd
#load the training data as pandas DataFrame
titanic_train = pd.read_csv("../dataset/train.csv")

#It is important to understand the data. The description is as follows:
# VARIABLE DESCRIPTIONS:
# survival        Survival
# (0 = No; 1 = Yes)
# pclass          Passenger Class
# (1 = 1st; 2 = 2nd; 3 = 3rd)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# cabin           Cabin
# embarked        Port of Embarkation
# (C = Cherbourg; Q = Queenstown; S = Southampton)
# SPECIAL NOTES:
# Pclass is a proxy for socio-economic status (SES)
# 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower
# Age is in Years; Fractional if Age less than One (1)
# If the Age is Estimated, it is in the form xx.5
# With respect to the family relation variables (i.e. sibsp and parch)
# some relations were ignored.  The following are the definitions used
# for sibsp and parch.
# Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
# Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
# Parent:   Mother or Father of Passenger Aboard Titanic
# Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic
# Other family relatives excluded from this study include cousins,
# nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
# only with a nanny, therefore parch=0 for them.  As well, some
# travelled with very close friends or neighbors in a village, however,
# the definitions do not support such relations.

#lets take a look what the data looks like. 'head' prints the first 5 lines by default
print(titanic_train.head())

#From the exercise description, 
#"Although there was some element of luck involved in surviving the sinking, 
#some groups of people were more likely to survive than others, such as women, children, and the upper-class."
#It is evident that survival list is dominated by female and children.
#Hence gener, age, SibSp, Parch can be direct features that needs to be extracted

#lets check the information about the data loaded for any missing values
print(titanic_train.info())

# 177 Age column values are missing
# 687 Cabin values are missing
# 2 Embarked values are missing

#Lets explore how the data is distributed for age column. We can make use of 'matplotlib' and
#inherint visualization support that pandas.DataFrame offers for this purpose
import matplotlib.pyplot as plt
pd.DataFrame.hist(titanic_train, "Age", bins = 16)
plt.show()

#As we observe that in age distribution there no outliers that will ruin the mean calculation. 
#According to our hypothesis, Mean and Median of age column are supposed to be nearly the same.
#Lets print and check them
print(titanic_train["Age"].mean())
print(titanic_train["Age"].median())

# Since mean is almost equivalent to median, we can choose one these to replace the missing values.
# Lets replace all the missing values  or 'na' (not assigned) values of age with the median of age column
titanic_train["Age"] = titanic_train["Age"].fillna(titanic_train["Age"].median())

#Since column "cabin" is not much relevant to our probelm. We can safely ignore it.
#The column "Embarked" is a string object type.
# One way to fill in the missing values for this column is to take a look at the unique possible values,
# Observe the one that is repeated for maximum (mode) and use that values as a filler.

#lets check what are the unique values for column "Embarked"
print(titanic_train["Embarked"].unique())

#lets check how these values are distributed across the dataset
titanic_train["Embarked"].value_counts().plot(kind="bar")
plt.show()

#As we can see value "S" is repeated maximum times. We can replace the missing values with "S"
titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")

#now lets ensure that data that we require is ready, without any missing values
print(titanic_train.info())


#Most of the ML algorithms expect the input data to be in numeric format for better results.
#In our dataset the non-numeric columns are : name, sex, ticket, embarked, cabin. 
#The columns that we are of interest are "Sex and Embarked".
#Good way to convert this non-numeric data to numeric is to see all the unique values that each column takes.
#And assign those unique values to a descrete set of values like (0, 1, 2, etc.,)

#Lets see what are the unique values for column "Sex"
print(titanic_train["Sex"].unique())

#As the values are ['male' 'female']. Let us assign "male" to '0' and "female" to '1'
titanic_train.loc[titanic_train["Sex"] == "male", "Sex"] = 0
titanic_train.loc[titanic_train["Sex"] == "female", "Sex"] = 1


#Lets see what are the unique values for column "Embarked"
print(titanic_train["Embarked"].unique())

#As the values are ['S' 'C' 'Q']. Lets assign:
# "S" to '0, "C" to '1' and "Q" to '2'
titanic_train.loc[titanic_train["Embarked"] == "S", "Embarked"] = 0
titanic_train.loc[titanic_train["Embarked"] == "C", "Embarked"] = 1
titanic_train.loc[titanic_train["Embarked"] == "Q", "Embarked"] = 2

#Now that we have cleaned up our dataset. We are ready to pass it ML algorithm for making it learn and predict

# Lets use the Linear Regression alogrithm in this case. This can be imported from sklearn kit as follows
# and use cross validation for avoiding overfitting problem

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

alg = LinearRegression()

#Lets use 3 folds to train our algorithm. 
#We set random_state to '1' to generate the same split everytime we run the program
kf = KFold(titanic_train.shape[0], n_folds=3, random_state=1)

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize the predictions array
predictions = []

# Iterate through all the folds and store the predictions
for train, test in kf:
	#select the subset from training data for the 'train' in the current fold
	train_predictors = titanic_train[predictors].iloc[train, :]
	train_target = titanic_train["Survived"].iloc[train]
	#Train the algorithm
	alg.fit(train_predictors, train_target)
	# Since our alogrithm is trained. we can use it to make predictions
	predictions.append(alg.predict(titanic_train[predictors].iloc[test, :]))


#Lets see how the predictions array look like
print(predictions)


#Since preditions is holding the values in 3 different arrays, 
#it will be difficult to compare the predictions with the "Survived" column values
#Thus, lets concatenate them into one using "concatenate" function from "numpy"
import numpy as np
predictions = np.concatenate(predictions, axis = 0)

#Since our predictions array contains the float values ranging from 0 to 1 and
# our "Survived" column has only digital values '0' or '1'. It will be difficult to compare them for accuracy.
# So, lets fix a strategy like everything > 0.5 is survived i.e, 1 and viceversa
predictions[predictions > 0.5] = 1
predictions[predictions < 0.5] = 0

#lets print the predictions array and check
print(predictions)

#lets compare the predictions array with the "Survived" column values for accuracy
#first lets ensure that predictions array and values of "Survived" column that we want to compare are in same type.
print(type(predictions))
print(type(titanic_train["Survived"]))

#Since they are types array and series respecively. Lets convert the series type also into array
survived_arr = np.array(titanic_train["Survived"])


#lets calculate the accuracy of predictions
accuracy = len(titanic_train[predictions == survived_arr]) / len(predictions)
print(accuracy)
# The accuracy computed is 78%. This is descent, but not great. 
#We shall be improving by applying different techniques in our next exercises

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic_train[predictors], titanic_train["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


#With this accuracy in hand. Lets clean up our test_data and pass it our algorithm that has learnt from our training_data

#Load the test data
titanic_test = pd.read_csv("../dataset/test.csv")

#Clean up the data
# Check and Fill the missing values of Age with median of training data set
print(titanic_test.info())
#Note: we are using median of training dataset because 
#test_data set isn't something reliable to summarize on the entire (population) data
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_train["Age"].median())
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

#additionally we have noticed that column "Fare" also has some missing values. 
#lets look at the distribution of "Fare"
pd.DataFrame.hist(titanic_test, "Fare", bins = 20)
plt.show()

#As we noticed there are not outliers, we can use median
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

#Convert the non-numeric values to numeric
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

# Initialize the algorithm class
alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic_train[predictors], titanic_train["Survived"])

# Make predictions using the test set.
test_predictions = alg.predict(titanic_test[predictors])


# Create a new dataframe showing only the required data
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": test_predictions
    })

print(submission)
