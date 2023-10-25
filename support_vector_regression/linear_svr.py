import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Load data
df = pd.read_csv("./data/Position_Salaries.csv")
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
y = y.reshape(len(y), 1)

print(X)
print(y)

# Feature scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)

print(X)
print(y)

# Train svr model on entire dataset
regressor = SVR(kernel="rbf")
regressor.fit(X, y)

# Predict a new result
y_pred = regressor.predict(sc_x.transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred.reshape(-1, 1))

print(y_pred)

# Visualising the SVR results
plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color="red")
plt.plot(sc_x.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_x.inverse_transform(X)), max(sc_x.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color="red")
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(X_grid)).reshape(-1, 1)), color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
