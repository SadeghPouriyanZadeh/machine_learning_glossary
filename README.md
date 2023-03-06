# Machine Learning Glossay

Mastering the Language of Machine Learning

## How to use this markdown?

* **The Fastest Way:** Read the underlined words.
* **Deeper Understaing:** Read the whole meaning and the example.

## The Glossary

* **Linear regression** $\equiv$
a statistical method used to model the relationship between a categorical dependent variable and one or more independent variables. a popular method in the field of machine learning for solving regression problems, where the goal is to  <u>predict a continuous numerical value</u>.  
The linear regression model assumes a linear relationship between the dependent variable and the independent variables, which means that the relationship can be expressed mathematically as a straight line. The model estimates the coefficients of the line, which represent the slope and intercept.  
Once the model has been trained on a set of training data, it can be used to make predictions on new data by plugging in the values of the independent variables and calculating the corresponding predicted value of the dependent variable.  
**example:** Suppose we have a dataset of houses that includes the size of the house in square feet (the independent variable) and the price of the house in dollars (the dependent variable). We can use linear regression to model the relationship between these two variables and make predictions on new data.

1. First, we would split the data into a training set and a testing set. The training set would be used to train the linear regression model, while the testing set would be used to evaluate its performance.  
2. Next, we would fit the linear regression model to the training data by estimating the coefficients of the line that best fits the data.
3. Once the model has been trained, we can use it to make predictions on new data by plugging in the value of the independent variable (house size) and calculating the corresponding predicted value of the dependent variable (house price). For example, if we have a new house that is 1500 square feet in size, we can use the linear regression model to predict its price by calculating the value of the line at x=1500.
4. Finally, we would evaluate the performance of the model on the testing set by comparing the predicted values to the actual values of the dependent variable. We can use metrics such as mean squared error to measure how well the model is able to predict the dependent variable based on the independent variable.

***

* **Logistic regression** $\equiv$
a statistical method used to model the relationship between a categorical dependent variable and one or more independent variables. a popular method in the field of machine learning for solving binary classification problems, where the goal is to predict whether an observation belongs to one of two possible classes.  
The logistic regression model uses a logistic function to transform the output of a linear regression model into a probability value between 0 and 1, which can then be interpreted as the probability of an observation belonging to a certain class. The logistic function, also known as the sigmoid function, is an S-shaped curve that maps any real-valued number to a value between 0 and 1.
**Example:** Suppose you are a credit card company and you want to predict whether a credit card application will be approved or denied based on various factors such as the applicant's income, credit score, and employment status. You have a dataset of past credit card applications and their outcomes, which you can use to train a logistic regression model.

1. First, you would clean and preprocess the data, which might involve filling in missing values, encoding categorical variables, and standardizing numeric variables. You would then split the data into training and testing sets, with the majority of the data used for training the model and a smaller portion used for testing its accuracy.
2. Next, you would fit a logistic regression model to the training data using an appropriate software package, such as scikit-learn in Python. The model would estimate the probability of a credit card application being approved based on the values of the independent variables.
3. Once the model is trained, you can use it to make predictions on new credit card applications. For example, you could input the values of income, credit score, and employment status for a new applicant, and the model would output the predicted probability of the application being approved.
4. Based on the predicted probabilities, you can then set a threshold value (e.g., 0.5) to convert the probabilities into binary class labels (approved or denied). You can evaluate the performance of the model on the test set using metrics such as accuracy, precision, recall, and F1 score.

***

