import pandas as pd
import numpy as np
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt


#load dataset
columns = [
    'age',
    'type_employer',
    'fnlwgt',           # must remove
    'education',        # must remove
    'education_num',
    'marital',
    'occupation',
    'relationship',     # must remove
    'race',
    'sex',
    'capital_gain',
    'capital_loss',
    'hr_per_week',
    'country',
    'income',
]

dataset = pd.read_csv('adult.input', names = columns)
#print(dataset.describe())
print(dataset.head())

# numerical columns vs categorical columns
num_cols = dataset.drop('income', axis=1).select_dtypes('number').columns
cat_cols = dataset.drop('income', axis=1).select_dtypes('object').columns

print(f"Number of numerical columns: {len(num_cols)}")
print(f"Number of categorical columns: {len(cat_cols)}\n")


#create dependent and independent variable vectors
x = dataset.iloc[:,:-1].values   #independent: all rows and colums except last column
y = dataset.iloc[:,14].values   #dependent: >50k or <50k
#print(x)
#print(y)

#removing unnecessary attributes
to_drop = ['fnlwgt', 'education', 'relationship']
dataset.drop(columns=to_drop, inplace=True)

#dropping records with missing data
new_data = dataset.replace('?', np.NaN)
new_data = new_data.dropna(axis = 0, how ='any')   # new dataset with missing values dropped

print("Old data frame length:", len(dataset))
print("New data frame length:", len(new_data))
print("Number of rows with at least 1 NA value: ",
      (len(dataset)-len(new_data)))

#count number of missing values in each column
#print(dataset.isnull().sum()) # currently not detecting missing values

#Find categorical features
s = (new_data.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)

#Display only categorical feature records
features = new_data[object_cols]
print(features.head())

#binarizing attributes "capital_gain" "capital_loss" "country"
new_data.loc[new_data["capital_gain"] > 0, "capital_gain"] = 1
new_data.loc[new_data["capital_loss"] > 0, "capital_loss"] = 1
new_data["country"] = (new_data["country"] == "United-States").astype(int) # 1 if country == United_States, 0 otherwise
x = new_data.iloc[:,:-1].values
print(x)

#discretization of continuous attributes "age" and "hr_per_week"
