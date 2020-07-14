# LinearRegression
Supervised Learning-Linear Regression problem

Wine quality prediction using linear regression and deploying the model using flask to generate and provide the rest end point for predicting the wine quality.

User can use the rest API endpoint to send the input features and in response gets the quality of the wine.

Input features of the model are:
volatile acidity  
residual sugar  
chlorides    
pH  
sulphates  
alcohol 
sulfur dioxide

and Output is: quality


We have used docker to containerize the model so if you want to make any changes in model or the input, make the changes and rebuild the container.
- docker build --tag my-flask .
- delete the previous container
- docker run --name my-flask -p 5002:5002 my-flask

and use POSTMAN or CURL to send the request for predicting the quality of wine.
