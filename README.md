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

Daset is aviable of this link https://drive.google.com/drive/folders/1ogUkw24QmblPaG-P0fOkYtRxjAe7VHd-?usp=sharing


# EDA

Our model will forecast customer churn based on patterns that were noted in the past 12 months, by analyzing features such as client's personal details, earnings and spending habits.

## The DATA

The dataset constains 10127 clients with 21 features of which 6 are categorical and 15 are numerical. from the given set we aim to build a supervised learning algorithm to perform a classification task


* The table below describes the dataset futures and conclusions taken from the histograms, boxplots and description table.

<table><tr><td>Click on histogram to zoom in</td></tr></table> 
  
  
|  COLUMN                 | Description, Univariate analysis and Conclusion                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Histogram               |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------               |
| CLIENTNUM:       |  Client number. Unique identifier for the customer holding the account and primary key in our dataset. Since it does not add any value to the data, this column will be dropped                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | \[photo\]              |
| Attrition\_Flag:            | Our target value - whenter our client has attrited or not attrited. We can observe that our target variable is imbalanced - about 84% of the data belongs to the class Existing customers, and only 16% of the clients are churned.<br>Attrition flag distribution as the description said is imbalanced.  An imbalanced classification problem is an example of a classification problem where the distribution of examples across the known classes is biased or skewed. The distribution can vary from a  slight bias to a severe imbalance where there is one example in the minority class for hundreds, thousands, or millions of examples in the majority class or classes. | ![photo](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/arritation%20flag.png?raw=true)|
|Customer_Age:     | We can see a spike in ages at ~25 and no cases of a person younder than 25 is using this banks services, probably this is a age limit for the credit cardthe mean is ~46 and no one is older than 70 so it all makes sense and nothing seems wrong with this coulmn.| ![photo](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/customer_age.png?raw=true)| |
| Gender:          | Demographic variable - M=Male, F=Female.<br>The distribution is even.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |               |
| Dependent\_count:| Demographic variable - Number of dependents. Dependents are mostly 2 or 3, no more than 5. This seems normal                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | ![photo](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/Dependend_count.png?raw=true)|             |
| Education\_Level:| Demographic variable - Educational Qualification of the account holder (example: high school, college graduate, etc.). Most credit card holders are Graduates and most of frequencies are balanced except for Doctorade and Post-graduates. Also, here is an unknown label,<br>Probably these are users that don't want to update their personal information and we will leave it as it is<br>                                                                                                                                                                                                                                                                                     | \[photo\]              |
| <br>Marital\_Status:|  Demographic variable - Married, Single, Divorced, Unknown. most of the users are either Married (43%) or Single (39%), divorcees are least likely to own a credit card or the contly has a low devorce rate.The unknows will be left as they are                                                                                                                                                                                                                                                                                                                                                                                                                                  | \[photo\]              |
| Card\_Category:| Product Variable - Type of Card (Blue, Silver, Gold, Platinum). In the card category most of the users got the Blue Category, probably blue is the basic account<br>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | \[photo\]              |
| Months\_on\_book:| Period of relationship with bank. Months on book has negative values and unussualy high values for the max. Te histogram and the boxplots show that somthing is not right with these values. This column is to be explored further                                                                                                                                                                                                                                                                                                                                                                                                                                                 | ![photo](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/Monts%20on%20book.png?raw=true)|             |
| Total\_Relationship\_Count:| Total no. of products held by the customer. The  Relationship\_count is between 1 and 6, which seems fine                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | ![photo](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/total%20relationship%20count.png?raw=true)|           |
| Months\_Inactive\_12\_mon:| No. of months inactive in the last 12 months, the values are between 0 and 6, while the documentation states that this colums is for clients inactive in the last 12 months. Maybe 6 is the number of months inactive before a client is considered churned.<br>                                                                                                                                                                                                                                                                                                                                                                                                                   |![photo](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/months%20inactive.png?raw=true)|            |
| Contacts\_Count\_12\_mon:| No. of Contacts in the last 12 months                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | ![photo](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/constacts%20count.png?raw=true)|             |
| Credit\_Limit:              | Credit Limit on the Credit Card. This colums seem to have some outliers in the right side, there might be a client with credit line significantly better than the rest .                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | ![photo](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/credit%20limit.png?raw=true)|            |
| Total\_Revolving\_Bal:      | Total Revolving Balance on the Credit Card. The Total revolving balance is low with a lot of customers at 0, which means they are not using the credit card, and at 2500 this might be an upper limit. The distribution is even.                                                                                                                                                                                                                                                                                                                                                                                                                                                   |![photo](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/total%20revolving%20balance.png?raw=true) |           |
| Avg\_Open\_To\_Buy:         | Open to Buy Credit Line (Average of last 12 months) this  indicates the amount of credit available. In this dataset the min is 3 and the max is the same as the max of the credit limit.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | ![photo](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/avg%20open%20to%20buy.png?raw=true)|             |
| Total\_Amt\_Chng\_Q4\_Q1: | This  is the change for the transaction amount in Q4 and Q1, the mean is below 1 which indicates that there are less transactions in Q1 compared to Q4.<br>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | ![photo](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/total%20amt%20q4%20q1.png?raw=true)|             |
| Total\_Trans\_Amt:  | Total Transaction Amount (Last 12 months) indicates that 4.4K is the mean for the year so we have approximately 300  transactions per monts. The max is ~18K and that looks like an outlier.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |![photo](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/total%20trans%20amt.png?raw=true)|            |
| Total\_Trans\_Ct:  | Total Transaction Count (Last 12 months) 64 is the mean for this colums, which is approximately 5.4 per month. The max value is 139. Considering the mean, it seems that we have outlies                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |![photo](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/total%20trans%20ct.png?raw=true)|              |
| Total\_Ct\_Chng\_Q4\_Q1:| Change in Transaction Count (Q4 over Q1)  gives us the same information as The Total\_Amt\_Chng\_Q4\_Q1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | ![photo](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/total%20ct%20q4%20q1.png?raw=true)|              |
| Avg\_Utilization\_Ratio     | Average Card Utilization Ratio is 0.27 but the median is 0.18 which could indicate a positive skew, also there seem to be outliers on the left side. The percentile 25 is 0.02 indicates that 25% are hardly using the product.                                                                                                                                                                                                                                                                                                                                                                                                                                                    | ![photo](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/acg%20utl%20ration.png?raw=true)|              |

# Handling missing values

Handling missing values is an essential part of data cleaning and preparation process because almost all data in real life comes with some missing values.
Our dataset does not have any nulls or blanks. However, there are a few unussual values in the Months_on_book column which can be noticed from the histogram  and the boxplot, 
since there are only 52 cases of unusual value, we decedied that removing these rows will not do any harm to the dataset in this case so we filtered out this column's values between 0 and 500 months.

# Classification models

## 1. Scikit-learn environment

In our manual approach in Scikit-learn environment, we used seven main supervised learning algorithms:  Dummy classifier (DC), Naive bayes (NB), Decision Tree (DT), K-Nearest Neighbors (KNN), eXtreme Gradient Boosting (XGBoost), Logistic regression(LR) and Random Forest (RF).


**Dummy classifier (DC)**.We started with a DC like baseline classifier.This kind of model often is used like a benchmark or in other words if this model gives us better results than other models we don’t need to use other sophisticated models and vice versa. For the purpose of this modeling we used  stratified  DC which generates random predictions by respecting the training set class distribution.Output metrics from this model are: Accuracy 0.7176, Precision  0.8380, Recall 0.8226, F1 Score 0.8303 and FBeta Score 0.8378.Тhese metrics will take as a benchmark and expect the following models give us much better results.

**Naive bayes (NB)**. In statistics, naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naïve) independence assumptions between the features. Output metrics from this model are: Accuracy 0.894293, Precision  0.937315, Recall 0.936761, F1 Score 0.937038 and FBeta Score 0.937310.Тhese metrics are better than metrics from DC, so now we have solid base to continue to examine further with other models.

**Decision Tree (DT)**. Decision tree learning or induction of decision trees is one of the predictive modelling approaches used in statistics, data mining and machine learning. It uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). Output metrics from this model are: Accuracy 0.913151, Precision  0.937315, Recall 0.973995, F1 Score 0.949582 and FBeta Score 0.926812.Тhese metrics are better than metrics from DC and also NB. In order to make better results with this model we applied fine tuning with this algorithm. Output metrics from this model are: Accuracy 0.941935, Precision  0.960257, Recall 0.971040, F1 Score 0.965619 and FBeta Score 0.960363.So here we can conclude that DT with tuning  parameters fitting better than DT  model with default parameters.


**K-Nearest Neighbors (KNN)**. In statistics, the k-nearest neighbors algorithm (k-NN) is a non-parametric classification method first developed by Evelyn Fix and Joseph Hodges in 1951, and later expanded by Thomas Cover. It is used for classification and regression.In this case we used KNN for classification problem.Output metrics from this model are: Accuracy 0.895782 , Precision  0.927336, Recall 0.950355, F1 Score 0.938704 and FBeta Score 0.927558.In order to make better results with this model we applied fine tuning with this algorithm. Output metrics from this model are: Accuracy 0.887841, Precision 0.909497, Recall 0.962175 , F1 Score 0.935095 and FBeta Score 0.909990. So here we can conclude that KNN with default parameters fitting better than KNN model with tuning parameters.

**eXtreme Gradient Boosting (XGBoost)**. Gradient Boosting Machines are one of the most popular boosting machines and were originally devised by Jerome Friedman. Gradient Boosting Machines are considered a ‘gradient buster’ as the algorithm iteratively solves residuals and the boosted characteristics are derived from the use of multiple weak models algorithmically (Gutierrez, 2015, p. 247). Output metrics from this model are: Accuracy 0.976179 , Precision  0.981829 , Recall 0.989953 , F1 Score 0.985874 and FBeta Score 0.981909. In order to make better results with this model we applied fine tuning with this algorithm. Results with this model we applied fine tuning with this algorithm but results are the same as with default parameters.

**Logistic regression (LR)**. Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression(or logit regression) is estimating the parameters of a logistic model (a form of binary regression). Output metrics from this model are: Accuracy 0.976179 , Precision 0.920697 , Recall 0.967494 , F1 Score 0.943516 and FBeta Score 0.921139.

**Random forests (RF)**. These ensemble methods usually combine the classification and prediction results of many different models. The individual models are known as ‘weak learners’ because individually they have poor predictive performance. Random forests are an ensemble classifier that fits a large number of decision trees to a data set, and then combines the predictions from all the trees and for regression, forests are created by averaging over trees.Output metrics from this model are: Accuracy 0.882878, Precision 0.881952, Recall 0.993499 , F1 Score 0.934408 and FBeta Score 0.882933. In order to make better results with this model we applied fine tuning with this algorithm. Output metrics from this model are: Accuracy 0.961787 , Precision 0.967574 , Recall 0.987589 , F1 Score 0.977479   and FBeta Score 0.967768.So here we can conclude that RF with tuned parameters can fit much better than with default. 

A total of seven supervised models were assessed. The best performing model is XGBoost which predicted customs churn with 97,61% accuracy.

In order to evaluate the classifiaction models we used the Receiver Operationg Characteristic, which is a very usful tool when predicting the probability of a binary outcome. THE ROC curve is probbility curve that plots the True positive rate agains the false positive rate. The true positive rate is calculated as the number of true positives divided by the sum of the number of true positives and the number of false negatives and it describes how good the model is at predicting the positive class when the actual outcome is positive. The false positive rate is calculated as the number of false positives divided by the sum of the number of false positives and the number of true negatives and it summarizes how often a positive class is predicted when the actual outcome is negative. The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve. The higher the AUC the he better the performance of the model. When 0.5<AUC<1, there is a high chance that the classifier will be able to distinguish the positive class values from the negative class values.

![Test picture](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/ScikitlearnEnviroment/RocCurves.png)

In our case, beside the Dummy classifier, all the models showed a AUC above 0.5. The top 3 models based on this metric are: XGBClassifier (AUC=0.99), Random Forest Classifier (AUC=0.94) and GridSearchCV(AUC=0.92). Acorrding to this findigs the best model is XGBClassifier. The classifier is able to perfectly distinguish between all the Positive and the Negative class points correctly.


# 2. PyCaret environment      

In our automatic approach in PyCaret environment, we used fifteen  supervised learning algorithms:  Dummy classifier (DC), Naive bayes (NB), Decision Tree (DT), K-Nearest Neighbors (KNN), eXtreme Gradient Boosting (XGBoost), Logistic regression(LR) and Random Forest (RF).



![Test picture](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/Python/CaretEnviroment/FinalResults.PNG)



# 3. R environment

In our approach to machine learning in R environment, we used seven main supervised learning algorithms:  C5.0 algorithm (C5.0 ), Naive bayes(NB), Decision Tree Classifier (DT), KNeighborsClassifier (KNN), eXtreme Gradient Boosting (XGBoost), Logistic regression (LR) and Random Forest Classifier (RF).  

**C5.0.** While there are numerous implementations of decision trees, one of the most well-known is the C5.0 algorithm. This algorithm is an evolution of the C4.5 of Quinlan (1993). The C5.0 algorithm has become the industry standard for producing decision trees, because it does well on foremost types of problems directly out of the box. Compared to more advanced and sophisticated machine learning models the decision trees under the C5.0 algorithm generally perform nearly as well but are much easier to understand and deploy.


**Classification and regression trees (CART).** Originally CART was proposed by Breiman et al. (1984). This form of algorithm is a decision tree and is designed for numerical, continuous predictors and categorical response variables. However, decisions trees are highly prone to overfitting and we must be aware of this weakness. 


**Stochastic Gradient Boosting(GBM)**. In the previous section we already explain this model so here we will focus only on output metrics. Namely this model have : Accuracy 0.894293, Precision  0.937315, Recall 0.936761, F1 Score 0.937038 and FBeta Score 0.937310.Тhese metrics are better than metrics from DC, so now we have solid base to continue to examine further with other models. 




![Test picture](https://github.com/jordans78/PredictingCustomerChurn/blob/main/Documentation/R/AccurancyBoxPlots.PNG)






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










