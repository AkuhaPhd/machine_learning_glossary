import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing dataset
df = pd.read_csv("./data/Salary_Data.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Splitting the data to training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the simple linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test results
y_pred = regressor.predict(X_test)

# Visualising the training results
plt.scatter(X_train, y_train, c="red")
plt.plot(X_train, regressor.predict(X_train), c="blue")
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualising the test results
plt.scatter(X_test, y_test, c="red")
plt.plot(X_train, regressor.predict(X_train), c="blue")
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


