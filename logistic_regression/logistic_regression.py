import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# Load data
df = pd.read_csv("./data/Social_Network_Ads.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split data into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the logistic regression model on the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)

# Predict a new result
x_pred = classifier.predict([X_test[0]])
y_true = Y_test[0]
print(x_pred, y_true)

# Predicting the test set results
y_pred = classifier.predict(X_test)
print(
    np.concatenate((y_pred.reshape(len(y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1)
)

# Confusion matrix
cm = confusion_matrix(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
print(cm)
print(acc)
