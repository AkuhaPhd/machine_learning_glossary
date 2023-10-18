import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv("data/Data.csv")
x = dataset.iloc[:, :-1].values  # Independent variables
y = dataset.iloc[:, -1].values  # Dependent variables
missing_data = dataset.isnull().sum()

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 1:])
x[:, 1:] = imputer.transform(x[:, 1:])

# Encode independent variable
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
x = np.array(ct.fit_transform(x))

# Encoding dependent variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Feature scaling
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train, "\n")
print(x_test, "\n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
