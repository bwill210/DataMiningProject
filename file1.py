import pandas as pd
import numpy as np
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
print(x)
print(y)


#handle missing data

#count number of missing values in each column
print(dataset.isnull().sum()) # currently not detecting missing values
                              # perhaps we need to change '?' for missing values