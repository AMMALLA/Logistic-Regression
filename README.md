# Logistic-Regression
Logistic Regression Description

Contents
1) What is Logistic Regression

2) Why do we use Logistic regression rather than Linear Regression?

3) Logistic Function

How Linear regression is similar to logistic regression?
Derivation of the sigmoid function
What are odds?
4) Cost function in Logistic regression

5) What is the use of MLE in logistic regression?

Derivation of the Cost function
Why do we take the Negative log-likelihood function?
6) Gradient Descent Optimization

Derivative of the Cost function
Derivative of the sigmoid function
7) Endnotes

What is Logistic Regression?
Logistic regression is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary). Like all regression analyses, logistic regression is a predictive analysis. Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.

I found this definition on google and now we’ll try to understand it. Logistic Regression is another statistical analysis method borrowed by Machine Learning. It is used when our dependent variable is dichotomous or binary. It just means a variable that has only 2 outputs, for example, A person will survive this accident or not, The student will pass this exam or not. The outcome can either be yes or no (2 outputs). This regression technique is similar to linear regression and can be used to predict the Probabilities for classification problems.

Why do we use Logistic Regression rather than Linear Regression?
If you have this doubt, then you’re in the right place, my friend. After reading the definition of logistic regression we now know that it is only used when our dependent variable is binary and in linear regression this dependent variable is continuous.

The second problem is that if we add an outlier in our dataset, the best fit line in linear regression shifts to fit that point.

Now, if we use linear regression to find the best fit line which aims at minimizing the distance between the predicted value and actual value, the line will be like this:


Here the threshold value is 0.5, which means if the value of h(x) is greater than 0.5 then we predict malignant tumor (1) and if it is less than 0.5 then we predict benign tumor (0). Everything seems okay here but now let’s change it a bit, we add some outliers in our dataset, now this best fit line will shift to that point. Hence the line will be somewhat like this:


Here the threshold value is 0.5, which means if the value of h(x) is greater than 0.5 then we predict malignant tumor (1) and if it is less than 0.5 then we predict benign tumor (0). Everything seems okay here but now let’s change it a bit, we add some outliers in our dataset, now this best fit line will shift to that point. Hence the line will be somewhat like this:



Logistic Function
You must be wondering how logistic regression squeezes the output of linear regression between 0 and 1. If you haven’t read my article on Linear Regression then please have a look at it for a better understanding.

 Well, there’s a little bit of math included behind this and it is pretty interesting trust me.

Let’s start by mentioning the formula of logistic function:


How similar it is too linear regression? If you haven’t read my article on Linear Regression, then please have a look at it for a better understanding.

 We all know the equation of the best fit line in linear regression is:

 ![image](https://user-images.githubusercontent.com/4158204/152082153-c1ea0d20-8484-4a76-a358-8dabce820b5b.png)

 Let’s say instead of y we are taking probabilities (P). But there is an issue here, the value of (P) will exceed 1 or go below 0 and we know that range of Probability is (0-1). To overcome this issue we take “odds” of P:
 
 ![image](https://user-images.githubusercontent.com/4158204/152082027-4a3dc3ef-490c-4bc3-9a84-5850ced987bc.png)

 
 
