import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load data
df = pd.read_csv("./data/Position_Salaries.csv")
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Training the Polynomial Linear model on the whole dataset
polynomial_regressor = PolynomialFeatures(degree=4)
X_poly = polynomial_regressor.fit_transform(X)
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color="red")
plt.plot(X, linear_regressor.predict(X), color="blue")
plt.title("Truth or Bluff (Linear regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color="red")
plt.plot(X, linear_regressor_2.predict(X_poly), color="blue")
plt.title("Truth or Bluff (Polynomial regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color="red")
plt.plot(X_grid, linear_regressor_2.predict(polynomial_regressor.fit_transform(X_grid)), color="blue")
plt.title("Truth or Bluff (Polynomial regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with Linear Regression
new_salary = linear_regressor.predict([[6.5]])
print(new_salary)

# Predicting a new result with Polynomial Regression
new_salary = linear_regressor_2.predict(polynomial_regressor.fit_transform([[6.5]]))
print(new_salary)
