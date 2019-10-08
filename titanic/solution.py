import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


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
	dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

# dropping 'Cabin' and other features that are not important
columns_to_drop = ['PassengerId', 'Cabin', 'Ticket']
train_data.drop(columns_to_drop, axis=1, inplace=True)

# 3. CREATING NEW FEATURES i.e feature engineering
for dataset in data_cleaner:
	# the family size is the number of family members plus the individual
	dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

	# somebody was alone if there was no family member onboard
	# 1 indicates somebody was alone and 0 had a family member present
	dataset['IsAlone'] = 1
	dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

	# create 'Title' from names
	# for example 'Braund, Mr. Owen Harris' will have a title of 'Mr'
	dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

	# put 'Fare' and 'Age' to 4 and 5 bins respectively
	dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)  # all bins will have the same number of itesm

	dataset['AgeBin'] = pd.cut(dataset['Age'], 5)  # all bins will have the same range, number of items per bin will depend on the distribution of age


# clean up titles with count of less than 10, they will be grouped into one group called 'Misc'
stat_min = 10
title_names = (train_data['Title'].value_counts() < stat_min)  # series of 'True' if count if greater than 10 and vice versa

# use the series to attempt to rename rare titles
train_data['Title'] = train_data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)


# 4. CONVERTING FEATURES
# converting categorical features to numerical types.

# using label encoder to encode categorical values to values between 0 and n_classes - 1

encoder = LabelEncoder()
for dataset in data_cleaner:
	dataset['Sex_Code'] = encoder.fit_transform(dataset['Sex'])
	dataset['Embarked_Code'] = encoder.fit_transform(dataset['Embarked'])
	dataset['Title_Code'] = encoder.fit_transform(dataset['Title'])
	dataset['AgeBin_Code'] = encoder.fit_transform(dataset['AgeBin'])
	dataset['FareBin_Code'] = encoder.fit_transform(dataset['FareBin'])

#define y variable aka target/outcome
Target = ['Survived']

# define x variables for original features aka feature selection
train_data_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts
train_data_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation
train_data_xy =  Target + train_data_x 


# define x variables for original w/bin features to remove continuous variables
# i.e using 'AgeBin_Code' instead of 'Age' and 'FareBin_Code' instead of 'Fare'
train_data_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
train_data_xy_bin = Target + train_data_x_bin

# define x and y variables for dummy features original
# will get the dummies for categorical features by using OneHotEncoding
train_data_dummy = pd.get_dummies(train_data[train_data_x])
train_data_x_dummy = train_data_dummy.columns.tolist()
train_data_xy_dummy = Target + train_data_x_dummy

# we next split the training data into training and testing set to avoid overfitting by the model
