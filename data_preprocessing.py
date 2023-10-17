import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

# Importing the dataset
dataset = pd.read_csv("data/Data.csv")
x = dataset.iloc[:, :-1].values  # Independent variables
y = dataset.iloc[:, -1].values  # Dependent variables
missing_data = dataset.isnull().sum()
print(missing_data)

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 1:])
x[:, 1:] = imputer.transform(x[:, 1:])

print(x)
print(y)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
