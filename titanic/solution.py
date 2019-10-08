import pandas as pd


# STEP 1: GET THE DATA
train_data_raw = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# STEP 2: PREPARE THE DATA
# data needs to be prepared before modeling
# this involves correcting wrong values, completing missing values, creating new features and conversion of features
# to create a new object of the train_data
train_data = train_data_raw.copy(deep=True)

# for cleaning the data, we can pass the data by reference
data_cleaner = [train_data, test_data]

# get some info about the train_data
# here we learn the following:
# 1. there are 891 rows i.e instances in the train_data
# 2. there are 12 rows i.e features in the train_data
# 3. we have int64, float64 and object data types in the features
# 4. 'Age', 'Cabin', and 'Embarked' have some values missing
print(train_data.info())

# 2.1 correcting wrong values
# we do not have obvious wrong values in our dataset, so this step can be ommitted

# 2.2 completing missing values
# we will either drop columns with missing values or filling the missing data points with appropriate values
# missing values in train_data
print(train_data.isnull().sum())  # 'Age': 177, 'Cabin': 687, 'Embarked': 2
# missing values in the test data
print(test_data.isnull().sum())  # 'Age': 86, 'Cabin': 327, 'Fare': 1
# for 'Age' and 'Fare' (numerical columns) - we will fill the missing values with the median of their columns
# for 'Embarked' (categorical column) - we will fill the missing values with the mode of the column
# 'Cabin' has so many missing values, we will therefore drop the column

# fill missing values
for dataset in data_cleaner:
	dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
	dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
	dataset['Embarked'].fillna(dataset['Embarked'].mode(), inplace=True)

# dropping 'Cabin' and other features that are not important
columns_to_drop = ['PassengerId', 'Cabin', 'Ticket']
train_data.drop(columns_to_drop, axis=1, inplace=True)

# 2.3 Creating new features i.e feature engineering
for dataset in data_cleaner:
	# the family size is the number of family members plus the individual
	dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

	# somebody was alone if there was no family member onboard
	# 1 indicates somebody was alone and 0 had a family member present
	dataset['isAlone'] = 1
	dataset['isAlone'].loc[dataset['FamilySize'] > 1] = 0

	# create 'Title' from names
	# for example 'Braund, Mr. Owen Harris' will have a title of 'Mr'
	dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
