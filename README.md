# Predicting Customer Churn

# Introduction
Competitive customer-dependent organizations, such as those in the banking industry,
are some of the most affected sectors by the free market economy, which allow service
providers to compete against one another for the attention of customers. Given that
customers are the most valuable assets that have a direct impact on the revenue of the
banking industry, customer churn is a source of major concern for service organizations

## What is Customer Churn?
Churn describes a state where a customer
unsubscribes or leaves an organization for its competitor, thereby leading to a loss in revenue
and profit. It is calculated by the number of customers who leave your company during a given time period, meaning
churn rate shows how your business is doing with keeping customers by your side.

## Why does churn matter?
The financial aspect of churn that causes most trouble.
The cost of acquring a customer is higher than the cost of retination and that is the reason why a bank would try to keep their old customers
And it costs even more to bring a new customer up to the same level as an existing customer
Given the importance of customers and the higher costs of attracting new customers
compared with retaining existing ones, banks and other customer-dependent industries
must be able to automate the process of predicting the behaviors of their customers using
customers’ data in their database.
The advancement of technology in the last few decades has made it possible for
banks and many other service organizations to collect and store data about their customers
and classify them into either the churner or non-churner categories. Data by themselves do
not have much value if they are not studied to reveal the information contained in them.
Our model will forecast customer churn based on patterns that were noted in the past, by analyzing features such as personal details, earnings and spending habits.

Daset is aviable of this link https://drive.google.com/drive/folders/1ogUkw24QmblPaG-P0fOkYtRxjAe7VHd-



![Test picture](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/RocCurves.png)


# Comparison of three best models 
| Ecosystem 	| Enviroment 	| Classifier 	| Accuracy 	| Percision 	| Recall 	| F1 Score 	|
|---	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|
| Python 	| Scikitlearn  	| XGBoost 	| 0.976 	| 0.982 	| 0.990 	| 0.986 	|
| Python 	| Scikitlearn  	| Logistic   Regression 	| 0.976 	| 0.921 	| 0.967 	| 0.944 	|
| Python 	| Scikitlearn  	| Random Forest 	| 0.962 	| 0.968 	| 0.988 	| 0.977 	|
| Python 	| PyCaret 	| CatBoost Classifier 	| 0.973 	| 0.978 	| 0.990 	| 0.984 	|
| Python 	| PyCaret 	| Light Gradient Boosting Machine 	| 0.973 	| 0.980 	| 0.988 	| 0.984 	|
| Python 	| PyCaret 	| Extreme Gradient Boosting 	| 0.972 	| 0.979 	| 0.987 	| 0.983 	|
| R 	| Caret 	| c5.0 	| 0.971 	|   	|   	|   	|
| R 	| Caret 	| Stochastic   Gradient Boosting 	| 0.965 	|   	|   	|   	|
| R 	| Caret 	| Random Forest 	| 0.966 	|   	|   	|   	|










