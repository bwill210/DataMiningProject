import pandas as pd
import numpy as np
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt

# load dataset
columns = [
    'age',
    'type_employer',
    'fnlwgt',  # must remove
    'education',  # must remove
    'education_num',
    'marital',
    'occupation',
    'relationship',  # must remove
    'race',
    'sex',
    'capital_gain',
    'capital_loss',
    'hr_per_week',
    'country',
    'income',
]

dataset = pd.read_csv('adult.input', names=columns)
#print(dataset.describe())
dataset = pd.DataFrame(dataset)
print(dataset.head())

# create dependent and independent variable vectors
x = dataset.iloc[:, :-1].values  # independent: all rows and colums except last column
y = dataset.iloc[:, 14].values  # dependent: >50k or <50k
# print(x)
# print(y)

# removing unnecessary attributes
to_drop = ['fnlwgt', 'education', 'relationship']
dataset.drop(columns=to_drop, inplace=True)
print("Dropping columns : 'fnlwgt', 'education', 'relationship'\n")

# dropping records with missing data
dataset = dataset.replace('?', np.NaN)
new_data = dataset.dropna(axis=0, how='any')  # new dataset with missing values dropped
print("Dropping records with missing data...")

print("Old data frame length:", len(dataset))
print("New data frame length:", len(new_data))
print("Number of rows with at least 1 NA value: ",
      (len(dataset) - len(new_data)))
dataset = new_data

# numerical columns vs categorical columns
num_cols = dataset.drop('income', axis=1).select_dtypes('number').columns
cat_cols = dataset.drop('income', axis=1).select_dtypes('object').columns

print(f"Number of numerical columns: {len(num_cols)}")
print(f"Number of categorical columns: {len(cat_cols)}\n")

# count number of missing values in each column
# print(dataset.isnull().sum()) # currently not detecting missing values

# Find categorical features
s = (dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print("\n")

# Display only categorical feature records
features = dataset[object_cols]
print(features.head())
print("\n")

# Find numerical features
s = (dataset.dtypes == 'int64')
numeric_cols = list(s[s].index)
print("Numeric variables:")
print(numeric_cols)
print("\n")

# Display only numerical feature records
features = dataset[numeric_cols]
print(features.head())
print("\n")

# binarizing attributes "capital_gain" "capital_loss" "country"
print("binarizing attributes : 'capital_gain', capital_loss', 'country'")
dataset.loc[dataset["capital_gain"] > 0, "capital_gain"] = 1
dataset.loc[dataset["capital_loss"] > 0, "capital_loss"] = 1
dataset["country"] = (dataset["country"] == "United-States").astype(int)  # 1 if country == United_States, 0 otherwise
dataset["capital_gain>0"] = dataset["capital_gain"]
dataset["capital_loss>0"] = dataset["capital_loss"]
dataset["country=USA"] = dataset["country"]
to_drop = ["country", "capital_gain", "capital_loss"]
dataset.drop(columns=to_drop, inplace=True)
print(dataset)

# discretization of continuous attributes "age" and "hr_per_week" (3.2)
#splitting age into four intervals
dataset["young[<=25]"] = dataset["age"]
dataset["adult[26,45]"] = dataset["age"]
dataset["senior[46,65]"] = dataset["age"]
dataset["old[66,90]"] = dataset["age"]
#binarizing age for each interval
dataset["young[<=25]"] = (dataset["young[<=25]"] <= 25).astype(int)
dataset["adult[26,45]"] = ((dataset["adult[26,45]"] < 46)
                           & (dataset["adult[26,45]"] > 25)).astype(int)
dataset["senior[46,65]"] = ((dataset["senior[46,65]"] < 66)
                            & (dataset["senior[46,65]"] > 45)).astype(int)
dataset["old[66,90]"] = ((dataset["old[66,90]"] < 91)
                         & (dataset["old[66,90]"] > 65)).astype(int)
#drop column 'age'
dataset.drop(columns='age', inplace=True)

#splitting 'hr_per_week' into three intervals
dataset['part_time(<40)'] = dataset['hr_per_week']
dataset['full_time(=40)'] = dataset['hr_per_week']
dataset['over_time(>40)'] = dataset['hr_per_week']
#binarizing hr-per-week for each interval
dataset['part_time(<40)'] = (dataset['part_time(<40)'] < 40).astype(int)
dataset['full_time(=40)'] = (dataset['full_time(=40)'] == 40).astype(int)
dataset['over_time(>40)'] = (dataset['over_time(>40)'] > 40).astype(int)
#drop old column 'hr_per_week'
dataset.drop(columns='hr_per_week', inplace=True)

#merge attributes together and create asymmetric binary values (3.3)
#splitting 'type_employer' into 4 columns
dataset['gov'] = dataset['type_employer']
dataset['not_working'] = dataset['type_employer']
dataset['private'] = dataset['type_employer']
dataset['self_employed'] = dataset['type_employer']
#binarize attributes (asymmetric)
dataset['gov'] = ((dataset['gov'] == 'Federal-gov')
                  | (dataset['gov'] == 'State-gov')
                  | (dataset['gov'] == 'Local-gov')).astype(int)
dataset['not_working'] = ((dataset['not_working'] == 'Without-pay')
                          | (dataset['not_working'] == 'Never-worked')).astype(int)
dataset['private'] = (dataset['private'] == 'Private').astype(int)
dataset['self_employed'] = ((dataset['self_employed'] == 'Self-emp-inc')
                  | (dataset['gov'] == 'Self-emp-not-inc')).astype(int)
#splitting 'marital' into 3 columns
dataset['married'] = dataset['marital']
dataset['never_married'] = dataset['marital']
dataset['not_married'] = dataset['marital']
#binarize attributes (asymmetric)
dataset['married'] = ((dataset['married'] == 'Married-AF-spouse')
                      | (dataset['married'] == 'Married-civ-spouse')).astype(int)
dataset['never_married'] = (dataset['never_married'] == 'Never-married').astype(int)
dataset['not_married'] = ((dataset['not_married'] == 'Married-spouse-absent')
                          | (dataset['not_married'] == 'Separated')
                          | (dataset['not_married'] == 'Divorced')
                          | (dataset['not_married'] == 'Widowed')).astype(int)
#splitting 'occupation' into 5 columns
dataset['exec_managerial'] = dataset['occupation']
dataset['prof-specialty'] = dataset['occupation']
dataset['other'] = dataset['occupation']
dataset['manual_work'] = dataset['occupation']
dataset['sales'] = dataset['occupation']
#binarize attributes (asymmetric)
dataset['exec_managerial'] = (dataset['exec_managerial'] == '').astype(int)
dataset['prof-specialty'] = (dataset['prof-specialty'] == '').astype(int)
dataset['other'] = ((dataset['other'] == 'Tech-support')
                    | (dataset['other'] == 'Adm-clerical')
                    | (dataset['other'] == 'Priv-house-serv')
                    | (dataset['other'] == 'Protective-serv')
                    | (dataset['other'] == 'Armed-Forces')
                    | (dataset['other'] == 'Other-service')).astype(int)
dataset['manual_work'] = ((dataset['manual_work'] == 'Craft-repair')
                          | (dataset['manual_work'] == 'Farming-fishing')
                          | (dataset['manual_work'] == 'Handlers-cleaners')
                          | (dataset['manual_work'] == 'Machine-op-inspct')
                          | (dataset['manual_work'] == 'Transport-moving')).astype(int)
dataset['sales'] = (dataset['sales'] == 'Sales').astype(int)

#dropping old columns
to_drop = ['type_employer', 'marital', 'occupation']
dataset.drop(columns=to_drop, inplace=True)



print(dataset.head())
print(dataset.describe(include='all'))
