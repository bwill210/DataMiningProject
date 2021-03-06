'''
Assignment: Project 2
authors: Brock Williams, Roman Mukosieiev (Undergraduates)
Data Mining spring 2022
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

print("---------------------------------------------------------------------\n"
      "Preprocessing adult.input data...\n"
      "---------------------------------------------------------------------\n")

# load dataset
columns = [
    'age',
    'type_employer',
    'fnlwgt',       # must remove
    'education',    # must remove
    'education_num',
    'marital',
    'occupation',
    'relationship', # must remove
    'race',
    'sex',
    'capital_gain',
    'capital_loss',
    'hr_per_week',
    'country',
    'income',
]

train_data = pd.read_csv('adult.input', names=columns)
train_data = pd.DataFrame(train_data)
print("adult.input preview : \n")
print(train_data.head())

# removing unnecessary attributes
to_drop = ['fnlwgt', 'education', 'relationship']
train_data.drop(columns=to_drop, inplace=True)
print("Dropping columns : 'fnlwgt', 'education', 'relationship'\n")

# dropping records with missing data
train_data = train_data.replace('?', np.NaN)
new_data = train_data.dropna(axis=0, how='any')  # new dataset with missing values dropped
print("Dropping records with missing data...")

print("Old data frame length:", len(train_data))
print("New data frame length:", len(new_data))
print("Number of rows with at least 1 NA value: ",
      (len(train_data) - len(new_data)))
train_data = new_data

# numerical columns vs categorical columns
num_cols = train_data.drop('income', axis=1).select_dtypes('number').columns
cat_cols = train_data.drop('income', axis=1).select_dtypes('object').columns

print(f"Number of numerical columns: {len(num_cols)}")
print(f"Number of categorical columns: {len(cat_cols)}\n")

# count number of missing values in each column
# print(dataset.isnull().sum()) # currently not detecting missing values

# Find categorical features
s = (train_data.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print("\n")

# Find numerical features
s = (train_data.dtypes == 'int64')
numeric_cols = list(s[s].index)
print("Numeric variables:")
print(numeric_cols)
print("\n")

# binarization of attributes "capital_gain" "capital_loss" "country" (symmetric)
train_data.loc[train_data["capital_gain"] > 0, "capital_gain"] = 1
train_data.loc[train_data["capital_loss"] > 0, "capital_loss"] = 1
train_data["country"] = (train_data["country"] == "United-States").astype(int)  # 1 if country == United_States, 0 otherwise
train_data["capital_gain>0"] = train_data["capital_gain"]
train_data["capital_loss>0"] = train_data["capital_loss"]
train_data["country=USA"] = train_data["country"]
to_drop = ["country", "capital_gain", "capital_loss"]
train_data.drop(columns=to_drop, inplace=True)


# discretization of continuous attributes "age" and "hr_per_week" (3.2)
# splitting age into four intervals
train_data["young[<=25]"] = train_data["age"]
train_data["adult[26,45]"] = train_data["age"]
train_data["senior[46,65]"] = train_data["age"]
train_data["old[66,90]"] = train_data["age"]
# binarization of age for each interval (asymmetric)
train_data["young[<=25]"] = (train_data["young[<=25]"] <= 25).astype(int)
train_data["adult[26,45]"] = ((train_data["adult[26,45]"] < 46)
                              & (train_data["adult[26,45]"] > 25)).astype(int)
train_data["senior[46,65]"] = ((train_data["senior[46,65]"] < 66)
                               & (train_data["senior[46,65]"] > 45)).astype(int)
train_data["old[66,90]"] = ((train_data["old[66,90]"] < 91)
                            & (train_data["old[66,90]"] > 65)).astype(int)
# drop column 'age'
train_data.drop(columns='age', inplace=True)

# splitting 'hr_per_week' into three intervals
train_data['part_time(<40)'] = train_data['hr_per_week']
train_data['full_time(=40)'] = train_data['hr_per_week']
train_data['over_time(>40)'] = train_data['hr_per_week']
# binarization hr-per-week for each interval (asymmetric)
train_data['part_time(<40)'] = (train_data['part_time(<40)'] < 40).astype(int)
train_data['full_time(=40)'] = (train_data['full_time(=40)'] == 40).astype(int)
train_data['over_time(>40)'] = (train_data['over_time(>40)'] > 40).astype(int)
# drop old column 'hr_per_week'
train_data.drop(columns='hr_per_week', inplace=True)

# merge attributes together and create asymmetric binary values (3.3)
# splitting 'type_employer' into 4 columns
train_data['gov'] = train_data['type_employer']
train_data['not_working'] = train_data['type_employer']
train_data['private'] = train_data['type_employer']
train_data['self_employed'] = train_data['type_employer']
# binarize attributes (asymmetric)
train_data['gov'] = ((train_data['gov'] == 'Federal-gov')
                     | (train_data['gov'] == 'State-gov')
                     | (train_data['gov'] == 'Local-gov')).astype(int)
train_data['not_working'] = ((train_data['not_working'] == 'Without-pay')
                             | (train_data['not_working'] == 'Never-worked')).astype(int)
train_data['private'] = (train_data['private'] == 'Private').astype(int)
train_data['self_employed'] = ((train_data['self_employed'] == 'Self-emp-inc')
                               | (train_data['gov'] == 'Self-emp-not-inc')).astype(int)
# splitting 'marital' into 3 columns
train_data['married'] = train_data['marital']
train_data['never_married'] = train_data['marital']
train_data['not_married'] = train_data['marital']
# binarize attributes (asymmetric)
train_data['married'] = ((train_data['married'] == 'Married-AF-spouse')
                         | (train_data['married'] == 'Married-civ-spouse')).astype(int)
train_data['never_married'] = (train_data['never_married'] == 'Never-married').astype(int)
train_data['not_married'] = ((train_data['not_married'] == 'Married-spouse-absent')
                             | (train_data['not_married'] == 'Separated')
                             | (train_data['not_married'] == 'Divorced')
                             | (train_data['not_married'] == 'Widowed')).astype(int)
# splitting 'occupation' into 5 columns
train_data['exec_managerial'] = train_data['occupation']
train_data['prof-specialty'] = train_data['occupation']
train_data['other'] = train_data['occupation']
train_data['manual_work'] = train_data['occupation']
train_data['sales'] = train_data['occupation']
# binarize attributes (asymmetric)
train_data['exec_managerial'] = (train_data['exec_managerial'] == '').astype(int)
train_data['prof-specialty'] = (train_data['prof-specialty'] == '').astype(int)
train_data['other'] = ((train_data['other'] == 'Tech-support')
                       | (train_data['other'] == 'Adm-clerical')
                       | (train_data['other'] == 'Priv-house-serv')
                       | (train_data['other'] == 'Protective-serv')
                       | (train_data['other'] == 'Armed-Forces')
                       | (train_data['other'] == 'Other-service')).astype(int)
train_data['manual_work'] = ((train_data['manual_work'] == 'Craft-repair')
                             | (train_data['manual_work'] == 'Farming-fishing')
                             | (train_data['manual_work'] == 'Handlers-cleaners')
                             | (train_data['manual_work'] == 'Machine-op-inspct')
                             | (train_data['manual_work'] == 'Transport-moving')).astype(int)
train_data['sales'] = (train_data['sales'] == 'Sales').astype(int)

# dropping old columns
to_drop = ['type_employer', 'marital', 'occupation']
train_data.drop(columns=to_drop, inplace=True)

# min-max scaling for 'education_num' (range 0-1)
minmax_scale = MinMaxScaler()
train_data['education_num'] = minmax_scale.fit_transform(train_data[['education_num']])

# Binarization of race -- if (white or asian) -> 1 else 0
train_data['race=white/asian'] = ((train_data['race'] == 'White')
                                  | (train_data['race'] == 'Asian-Pac-Islander')).astype(int)
train_data.drop(columns='race', inplace=True)

# Binarization of sex -- if male -> 1 else 0
train_data['sex=male'] = (train_data['sex'] == 'Male').astype(int)
train_data.drop(columns='sex', inplace=True)

# Binarization of income -- if >50k -> 1 else 0
train_data['income>50K'] = (train_data['income'] == ">50K").astype(int)
train_data.drop(columns='income', inplace=True)

print("adult.input data after preprocessing : ")
with pd.option_context('display.max_rows', 10,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(train_data)
# print(dataset.describe(include='all'))

# create dependent and independent variable vectors
x = train_data.iloc[:, :-1].values  # independent: all rows and colums except last column
y = train_data.iloc[:, 14].values  # dependent: >50k or <50k
print("\nPredicting attributes : \n")
print(x)
print("\nClass Label : \n")
print(y)
print("\n")

################################################################################################
# REPEATING PROCESS FOR adult.test
################################################################################################
print("---------------------------------------------------------------------\n"
      "Preprocessing adult.test data...\n"
      "---------------------------------------------------------------------\n")

# load dataset
columns = [
    'age',
    'type_employer',
    'fnlwgt',       # must remove
    'education',    # must remove
    'education_num',
    'marital',
    'occupation',
    'relationship', # must remove
    'race',
    'sex',
    'capital_gain',
    'capital_loss',
    'hr_per_week',
    'country',
    'income',
]

test_data = pd.read_csv('adult.test', names=columns)
test_data = pd.DataFrame(test_data)
print("adult.test preview : \n")
print(test_data.head())

# removing unnecessary attributes
to_drop = ['fnlwgt', 'education', 'relationship']
test_data.drop(columns=to_drop, inplace=True)
print("Dropping columns : 'fnlwgt', 'education', 'relationship'\n")

# numerical columns vs categorical columns
num_cols = test_data.drop('income', axis=1).select_dtypes('number').columns
cat_cols = test_data.drop('income', axis=1).select_dtypes('object').columns

print(f"Number of numerical columns: {len(num_cols)}")
print(f"Number of categorical columns: {len(cat_cols)}\n")

# count number of missing values in each column
# print(test_data.isnull().sum()) # currently not detecting missing values

# Find categorical features
s = (test_data.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print("\n")

# Find numerical features
s = (test_data.dtypes == 'int64')
numeric_cols = list(s[s].index)
print("Numeric variables:")
print(numeric_cols)
print("\n")

# binarization of attributes "capital_gain" "capital_loss" "country" (symmetric)
test_data.loc[test_data["capital_gain"] > 0, "capital_gain"] = 1
test_data.loc[test_data["capital_loss"] > 0, "capital_loss"] = 1
test_data["country"] = (test_data["country"] == "United-States").astype(int)  # 1 if country == United_States, 0 otherwise
test_data["capital_gain>0"] = test_data["capital_gain"]
test_data["capital_loss>0"] = test_data["capital_loss"]
test_data["country=USA"] = test_data["country"]
to_drop = ["country", "capital_gain", "capital_loss"]
test_data.drop(columns=to_drop, inplace=True)


# discretization of continuous attributes "age" and "hr_per_week" (3.2)
# splitting age into four intervals
test_data["young[<=25]"] = test_data["age"]
test_data["adult[26,45]"] = test_data["age"]
test_data["senior[46,65]"] = test_data["age"]
test_data["old[66,90]"] = test_data["age"]
# binarization of age for each interval (asymmetric)
test_data["young[<=25]"] = (test_data["young[<=25]"] <= 25).astype(int)
test_data["adult[26,45]"] = ((test_data["adult[26,45]"] < 46)
                             & (test_data["adult[26,45]"] > 25)).astype(int)
test_data["senior[46,65]"] = ((test_data["senior[46,65]"] < 66)
                              & (test_data["senior[46,65]"] > 45)).astype(int)
test_data["old[66,90]"] = ((test_data["old[66,90]"] < 91)
                           & (test_data["old[66,90]"] > 65)).astype(int)
# drop column 'age'
test_data.drop(columns='age', inplace=True)

# splitting 'hr_per_week' into three intervals
test_data['part_time(<40)'] = test_data['hr_per_week']
test_data['full_time(=40)'] = test_data['hr_per_week']
test_data['over_time(>40)'] = test_data['hr_per_week']
# binarization hr-per-week for each interval (asymmetric)
test_data['part_time(<40)'] = (test_data['part_time(<40)'] < 40).astype(int)
test_data['full_time(=40)'] = (test_data['full_time(=40)'] == 40).astype(int)
test_data['over_time(>40)'] = (test_data['over_time(>40)'] > 40).astype(int)
# drop old column 'hr_per_week'
test_data.drop(columns='hr_per_week', inplace=True)

# merge attributes together and create asymmetric binary values (3.3)
# splitting 'type_employer' into 4 columns
test_data['gov'] = test_data['type_employer']
test_data['not_working'] = test_data['type_employer']
test_data['private'] = test_data['type_employer']
test_data['self_employed'] = test_data['type_employer']
# binarize attributes (asymmetric)
test_data['gov'] = ((test_data['gov'] == 'Federal-gov')
                    | (test_data['gov'] == 'State-gov')
                    | (test_data['gov'] == 'Local-gov')).astype(int)
test_data['not_working'] = ((test_data['not_working'] == 'Without-pay')
                            | (test_data['not_working'] == 'Never-worked')).astype(int)
test_data['private'] = (test_data['private'] == 'Private').astype(int)
test_data['self_employed'] = ((test_data['self_employed'] == 'Self-emp-inc')
                              | (test_data['gov'] == 'Self-emp-not-inc')).astype(int)
# splitting 'marital' into 3 columns
test_data['married'] = test_data['marital']
test_data['never_married'] = test_data['marital']
test_data['not_married'] = test_data['marital']
# binarize attributes (asymmetric)
test_data['married'] = ((test_data['married'] == 'Married-AF-spouse')
                        | (test_data['married'] == 'Married-civ-spouse')).astype(int)
test_data['never_married'] = (test_data['never_married'] == 'Never-married').astype(int)
test_data['not_married'] = ((test_data['not_married'] == 'Married-spouse-absent')
                            | (test_data['not_married'] == 'Separated')
                            | (test_data['not_married'] == 'Divorced')
                            | (test_data['not_married'] == 'Widowed')).astype(int)
# splitting 'occupation' into 5 columns
test_data['exec_managerial'] = test_data['occupation']
test_data['prof-specialty'] = test_data['occupation']
test_data['other'] = test_data['occupation']
test_data['manual_work'] = test_data['occupation']
test_data['sales'] = test_data['occupation']
# binarize attributes (asymmetric)
test_data['exec_managerial'] = (test_data['exec_managerial'] == '').astype(int)
test_data['prof-specialty'] = (test_data['prof-specialty'] == '').astype(int)
test_data['other'] = ((test_data['other'] == 'Tech-support')
                      | (test_data['other'] == 'Adm-clerical')
                      | (test_data['other'] == 'Priv-house-serv')
                      | (test_data['other'] == 'Protective-serv')
                      | (test_data['other'] == 'Armed-Forces')
                      | (test_data['other'] == 'Other-service')).astype(int)
test_data['manual_work'] = ((test_data['manual_work'] == 'Craft-repair')
                            | (test_data['manual_work'] == 'Farming-fishing')
                            | (test_data['manual_work'] == 'Handlers-cleaners')
                            | (test_data['manual_work'] == 'Machine-op-inspct')
                            | (test_data['manual_work'] == 'Transport-moving')).astype(int)
test_data['sales'] = (test_data['sales'] == 'Sales').astype(int)

# dropping old columns
to_drop = ['type_employer', 'marital', 'occupation']
test_data.drop(columns=to_drop, inplace=True)

# min-max scaling for 'education_num' (range 0-1)
minmax_scale = MinMaxScaler()
test_data['education_num'] = minmax_scale.fit_transform(test_data[['education_num']])

# Binarization of race -- if (white or asian) -> 1 else 0
test_data['race=white/asian'] = ((test_data['race'] == 'White')
                                 | (test_data['race'] == 'Asian-Pac-Islander')).astype(int)
test_data.drop(columns='race', inplace=True)

# Binarization of sex -- if male -> 1 else 0
test_data['sex=male'] = (test_data['sex'] == 'Male').astype(int)
test_data.drop(columns='sex', inplace=True)

#rearranging dataframe so income>50K is last column
test_data['income>50K'] = test_data['income']
test_data.drop(columns='income', inplace=True)

print("adult.test data after preprocessing : ")
with pd.option_context('display.max_rows', 10,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(test_data)
# print(test_data.describe(include='all'))

##################################################################################
# Training and Model Selection (Perceptron)
##################################################################################

# Splitting training dataset
X = train_data.iloc[:, :-1].values  # independent: all rows and columns used to make predictions
Y = train_data.iloc[:, -1].values  # dependent: >50k or <50k
X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

X.columns = ['education_num', 'capital_gain>0', 'capital_loss>0', 'country=USA',
             'young[<=25]', 'adult[26,45]', 'senior[46,65]', 'old[66,90]',
             'part_time(<40)', 'full_time(=40)', 'over_time(>40)', 'gov',
             'not_working', 'private', 'self_employed', 'married', 'never_married',
             'not_married', 'exec_managerial', 'prof-specialty', 'other',
             'manual_work', 'sales', 'race=white/asian', 'sex=male']
Y.columns = ['income>50K']


# Splitting testing dataset
# Splitting testing dataset
Xtest = test_data.iloc[:, :-1].values  # independent: all rows and columns used to make predictions
Ytest = test_data.iloc[:, -1].values  # dependent: >50k or <50k
Xtest = pd.DataFrame(Xtest)
Ytest = pd.DataFrame(Ytest)

Xtest.columns = ['education_num', 'capital_gain>0', 'capital_loss>0', 'country=USA',
                 'young[<=25]', 'adult[26,45]', 'senior[46,65]', 'old[66,90]',
                 'part_time(<40)', 'full_time(=40)', 'over_time(>40)', 'gov',
                 'not_working', 'private', 'self_employed', 'married', 'never_married',
                 'not_married', 'exec_managerial', 'prof-specialty', 'other',
                 'manual_work', 'sales', 'race=white/asian', 'sex=male']
Ytest.columns = ['income>50K']

test_size = 0.28
random_state = 8
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=test_size,


                                                    random_state=random_state)
# model with 'max_iter=15' and 'eta0=0.2'
perc1 = Perceptron(max_iter=15, eta0=0.2, random_state=random_state)
# fit the model to the training data
perc1.fit(X_train, y_train.values.ravel())
y_pred1 = perc1.predict(X_test)
# measure performance using 'accuracy_score' function
print("model1 accuracy: ")
print(accuracy_score(y_test, y_pred1) * 100)
print("misclassification error:")
print(100 - (accuracy_score(y_test, y_pred1) * 100))

# Create confusion matrix png (max_iter=15)
fig = plt.figure(figsize=(6, 4))
mat = confusion_matrix(y_test, y_pred1)
sns.heatmap(mat.T, cmap="Blues", square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted (>50K)')
plt.ylabel('True (>50K)')
fig.savefig("confusion_matrix_perc1.png")


# model with 'max_iter=40' and 'eta0=0.2'
perc2 = Perceptron(max_iter=40, eta0=0.2, random_state=random_state)
# fit the model to the training data
perc2.fit(X_train, y_train.values.ravel())
y_pred2 = perc2.predict(X_test)
# measure performance using 'accuracy_score' function
print("model2 accuracy: ")
print(accuracy_score(y_test, y_pred2) * 100)
print("misclassification error:")
print(100 - (accuracy_score(y_test, y_pred2) * 100))

# Create confusion matrix png (max_iter=40)
fig = plt.figure(figsize=(6, 4))
mat = confusion_matrix(y_test, y_pred2)
sns.heatmap(mat.T, cmap="Blues", square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted (>50K)')
plt.ylabel('True (>50K)')
fig.savefig("confusion_matrix_perc2.png")


# model with 'max_iter=65' and 'eta0=0.2'
perc3 = Perceptron(max_iter=65, eta0=0.2, random_state=random_state)
# fit the model to the training data
perc3.fit(X_train, y_train.values.ravel())
y_pred3 = perc3.predict(X_test)
# measure performance using 'accuracy_score' function
print("model3 accuracy: ")
print(accuracy_score(y_test, y_pred3) * 100)
print("misclassification error:")
print(100 - (accuracy_score(y_test, y_pred3) * 100))

# Create confusion matrix png (max_iter=65)
fig = plt.figure(figsize=(6, 4))
mat = confusion_matrix(y_test, y_pred3)
sns.heatmap(mat.T, cmap="Blues", square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted (>50K)')
plt.ylabel('True (>50K)')
fig.savefig("confusion_matrix_perc3.png")


# model with 'max_iter=40' and 'eta0=1.2' (Best Model)
perc4 = Perceptron(max_iter=40, eta0=1.2, random_state=random_state)
# fit the model to the training data
perc4.fit(X_train, y_train.values.ravel())
y_pred4 = perc4.predict(X_test)
# measure performance using 'accuracy_score' function
print("model4 accuracy: ")
print(accuracy_score(y_test, y_pred4) * 100)
print("misclassification error:")
print(100 - (accuracy_score(y_test, y_pred4) * 100))
print("\n")

# Create confusion matrix png (max_iter=65)
fig = plt.figure(figsize=(6, 4))
mat = confusion_matrix(y_test, y_pred4)
sns.heatmap(mat.T, cmap="Blues", square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted (>50K)')
plt.ylabel('True (>50K)')
fig.savefig("confusion_matrix_perc4.png")

print("Evaluation Metrics :")
report = classification_report(y_pred4, y_test)
print(report)

# Saving test set predictions of best model (eta0=1.2)
perc = Perceptron(max_iter=40, eta0=1.2, random_state=random_state)
perc = perc.fit(X, Y)
y_predictFinal = perc.predict(Xtest)  # predicting labels for test data
np.savetxt("pred_perc_8.csv", y_predictFinal, delimiter=",", fmt="%d")
print("test set predictions saved to 'pred_perc_8.csv'")