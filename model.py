# import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import pickle


# Load data into a DataFrame
red_wine_dataset = pd.read_csv(r'datasets/winequality-red.csv', sep=';')

# drop the feature which is having high correlation with other features
red_wine_dataset_v1 = red_wine_dataset.drop(['fixed acidity', 'density', 'citric acid', \
                                          'free sulfur dioxide', 'total sulfur dioxide'], axis=1)

# assign to features X and target Y
X_red_v2 = red_wine_dataset_v1.drop(['quality'], axis=1)
Y_red_v2 = red_wine_dataset_v1['quality']

# Split data into train/test sets
X_train_red_v2, X_test_red_v2, y_train_red_v2, y_test_red_v2 = train_test_split(X_red_v2, Y_red_v2, \
                                                                                test_size=0.20, random_state=42)

# Initialize LR Model
lr = LinearRegression()

# Fit the model
lr.fit(X_train_red_v2, y_train_red_v2)

# Make predictions
predictions_red_v2 = lr.predict(X_test_red_v2)

# print metrics
print("R Squared Score: ", format(r2_score(y_test_red_v2, predictions_red_v2),'.3f'))
print("Root Mean Squared Error: ", format(np.sqrt(mean_squared_error(y_test_red_v2, predictions_red_v2)),'.3f'))
print("Mean Absolute Error: ", format(mean_absolute_error(y_test_red_v2, predictions_red_v2),'.3f'))

# predict the new records
# new_prediction = lr.predict(np.array([0.56, 2.5, 0.114, 3.24, 0.66, 9.6]).reshape(1,-1))
new_prediction = lr.predict(X_test_red_v2.head(1))
print(new_prediction)

# Save the Modle to file in the current working directory
Pkl_Filename = "Pickle_RL_Model.pkl"
with open(Pkl_Filename, 'wb') as file:
    pickle.dump(lr, file)

# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:
    Pickled_LR_Model = pickle.load(file)

print(Pickled_LR_Model)

# prediction
print(Pickled_LR_Model.predict(np.array([0.56, 2.5, 0.114, 3.24, 0.66, 9.6]).reshape(1,-1)))

# print coefficients
print(Pickled_LR_Model.coef_)
print(Pickled_LR_Model.intercept_)