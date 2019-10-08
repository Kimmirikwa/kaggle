import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

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

# splitting the data with original features plus categorical features encoded using label encoder
train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(train_data[train_data_x_calc], train_data[Target], random_state=0)

# splitting the data with original features plus categorical features encoded using label encoder and 'Age' and 'Fare' in bins
train_data_bin_x, test_data_bin_x, train_data_bin_y, test_data_bin_y = train_test_split(train_data[train_data_xy_bin], train_data[Target], random_state=0)

# splitting the data with original features plus categorical features encoded using OneHotEncoding
train_data_dummy_x, test_data_dummy_x, train_data_dummy_y, test_data_dummy_y = train_test_split(train_data_dummy[train_data_x_dummy], train_data[Target], random_state=0)

#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),   
]

cv_split = ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0 )

# create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']

# create table to compare MLA predictions
MLA_predict = train_data[Target]

training_data = [
	{
		"file_name": "scores_without_bins.csv",
		"data": train_data[train_data_x_calc]
	},
	{
		"file_name": "scores_with_bins.csv",
		"data": train_data[train_data_x_bin]
	},
	{
		"file_name": "scores_with_dummy.csv",
		"data": train_data_dummy[train_data_x_dummy]
	}
]

# index through MLA and save performance to table
for dataset in training_data:
	MLA_compare = pd.DataFrame(columns = MLA_columns)
	row_index = 0
	for alg in MLA:
	    # set name and parameters
	    MLA_name = alg.__class__.__name__
	    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
	    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
	    
	    # score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
	    cv_results = cross_validate(alg, dataset['data'], train_data[Target], cv=cv_split, return_train_score=True)

	    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
	    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
	    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
	    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
	    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
	    

	    # save MLA predictions - see section 6 for usage
	    alg.fit(train_data[train_data_x_bin], train_data[Target])
	    MLA_predict[MLA_name] = alg.predict(train_data[train_data_x_bin])
	    
	    row_index += 1

	MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

	MLA_compare[['MLA Name', 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean']].to_csv('results/{}'.format(dataset['file_name']))