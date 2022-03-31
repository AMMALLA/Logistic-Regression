# Logistic-Regression


Logistic Regression Description



![image](https://user-images.githubusercontent.com/4158204/152091063-70c70f33-4cb8-471c-bc83-77cddc6b16b1.png)


#### 1) What is Logistic Regression

#### 2) Why do we use Logistic regression rather than Linear Regression?

#### 3) Logistic Function


  ##### * How Linear regression is similar to logistic regression?
  
   ##### * Derivation of the sigmoid function
    
   ##### * What are odds?
  
4) Cost function in Logistic regression

5) What is the use of MLE in logistic regression?

    Derivation of the Cost function
    
    Why do we take the Negative log-likelihood function?
  
6) Gradient Descent Optimization

    Derivative of the Cost function
    
    Derivative of the sigmoid function
    
7) Implementation of logistic Regression using Python CODE

## What is Logistic Regression?


Logistic regression is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary). Like all regression analyses, logistic regression is a predictive analysis. Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.

I found this definition on google and now we’ll try to understand it. Logistic Regression is another statistical analysis method borrowed by Machine Learning. It is used when our dependent variable is dichotomous or binary. It just means a variable that has only 2 outputs, for example, A person will survive this accident or not, The student will pass this exam or not. The outcome can either be yes or no (2 outputs). This regression technique is similar to linear regression and can be used to predict the Probabilities for classification problems.

## Why do we use Logistic Regression rather than Linear Regression?


If you have this doubt, then you’re in the right place, my friend. After reading the definition of logistic regression we now know that it is only used when our dependent variable is binary and in linear regression this dependent variable is continuous.

The second problem is that if we add an outlier in our dataset, the best fit line in linear regression shifts to fit that point.

Now, if we use linear regression to find the best fit line which aims at minimizing the distance between the predicted value and actual value, the line will be like this:


Here the threshold value is 0.5, which means if the value of h(x) is greater than 0.5 then we predict malignant tumor (1) and if it is less than 0.5 then we predict benign tumor (0). Everything seems okay here but now let’s change it a bit, we add some outliers in our dataset, now this best fit line will shift to that point. Hence the line will be somewhat like this:

![image](https://user-images.githubusercontent.com/4158204/152091155-0927b702-7d5b-4e84-b60f-f89e42f50b01.png)


Here the threshold value is 0.5, which means if the value of h(x) is greater than 0.5 then we predict malignant tumor (1) and if it is less than 0.5 then we predict benign tumor (0). Everything seems okay here but now let’s change it a bit, we add some outliers in our dataset, now this best fit line will shift to that point. Hence the line will be somewhat like this:
![image](https://user-images.githubusercontent.com/4158204/152091237-b6dc4eb9-c52b-4e6b-b113-c1215f6b974d.png)

## Logistic Function


You must be wondering how logistic regression squeezes the output of linear regression between 0 and 1. If you haven’t read my article on Linear Regression then please have a look at it for a better understanding.

 Well, there’s a little bit of math included behind this and it is pretty interesting trust me.

Let’s start by mentioning the formula of logistic function:
![image](https://user-images.githubusercontent.com/4158204/152091296-95862d5e-aeb9-40bb-bd1d-ef1a5a7c0701.png)


How similar it is too linear regression? If you haven’t read my article on Linear Regression, then please have a look at it for a better understanding.

 We all know the equation of the best fit line in linear regression is:

![image](https://user-images.githubusercontent.com/4158204/152092129-691a479b-44a6-42b9-a31d-85592bf95f8e.png)


 Let’s say instead of y we are taking probabilities (P). But there is an issue here, the value of (P) will exceed 1 or go below 0 and we know that range of Probability is (0-1). To overcome this issue we take “odds” of P:
 
 ![image](https://user-images.githubusercontent.com/4158204/152082027-4a3dc3ef-490c-4bc3-9a84-5850ced987bc.png)

 
 Do you think we are done here? No, we are not. We know that odds can always be positive which means the range will always be (0,+∞ ). Odds are nothing but the ratio of the probability of success and probability of failure. Now the question comes out of so many other options to transform this why did we only take ‘odds’? Because odds are probably the easiest way to do this, that’s it.

The problem here is that the range is restricted and we don’t want a restricted range because if we do so then our correlation will decrease. By restricting the range we are actually decreasing the number of data points and of course, if we decrease our data points, our correlation will decrease. It is difficult to model a variable that has a restricted range. To control this we take the log of odds which has a range from (-∞,+∞).


![image](https://user-images.githubusercontent.com/4158204/152092213-cd2460aa-c957-45b9-8583-0057dcad9a4d.png)
  If you understood what I did here then you have done 80% of the maths. Now we just want a function of P because we want to predict probability right? not log of odds. To do so we will multiply by exponent on both sides and then solve for P.
  
  
  ![image](https://user-images.githubusercontent.com/4158204/152092739-29ecfe85-de04-4f99-a59a-5a03b63a5c4c.png)

Now we have our logistic function, also called a sigmoid function. The graph of a sigmoid function is as shown below. It squeezes a straight line into an S-curve.
![image](https://user-images.githubusercontent.com/4158204/152092813-8c883e26-e469-42ef-a2cc-894b4ba28ef1.png)
Cost Function in Logistic Regression
In linear regression, we use the Mean squared error which was the difference between y_predicted and y_actual and this is derived from the maximum likelihood estimator. The graph of the cost function in linear regression is like this:

![image](https://user-images.githubusercontent.com/4158204/152092881-7f5b50d1-e3e5-41ef-a004-5182f9c9859a.png)

![image](https://user-images.githubusercontent.com/4158204/152092931-e67c86fc-d594-4db7-9650-8f350d2e2d67.png)

In logistic regression Yi is a non-linear function (Ŷ=1​/1+ e-z). If we use this in the above MSE equation then it will give a non-convex graph with many local minima as shown:

![image](https://user-images.githubusercontent.com/4158204/152093040-358347e0-e556-4654-ba01-592c4a721ae0.png)

The problem here is that this cost function will give results with local minima, which is a big problem because then we’ll miss out on our global minima and our error will increase.

In order to solve this problem, we derive a different cost function for logistic regression called log loss which is also derived from the maximum likelihood estimation method.

![image](https://user-images.githubusercontent.com/4158204/152093113-f2483ab3-dca3-4a44-ab85-9b268fcae6d9.png)

In the next section, we’ll talk a little bit about the maximum likelihood estimator and what it is used for. We’ll also try to see the math behind this log loss function.

## What is the use of Maximum Likelihood Estimator?


The main aim of MLE is to find the value of our parameters for which the likelihood function is maximized. The likelihood function is nothing but a joint pdf of our sample observations and joint distribution is the multiplication of the conditional probability for observing each example given the distribution parameters. In other words, we try to find such that plugging these estimates into the model for P(x), yields a number close to one for people who had a malignant tumor and close to 0 for people who had a benign tumor.

Let’s start by defining our likelihood function. We now know that the labels are binary which means they can be either yes/no or pass/fail etc. We can also say we have two outcomes success and failure. This means we can interpret each label as Bernoulli random variable.

A random experiment whose outcomes are of two types, success S and failure F, occurring with probabilities p and q respectively is called a Bernoulli trial. If for this experiment a random variable X is defined such that it takes value 1 when S occurs and 0 if F occurs, then X follows a Bernoulli Distribution.

![image](https://user-images.githubusercontent.com/4158204/152093244-a6589866-e1f1-4c14-b90a-16f23160602d.png)

#####                             Where P is our sigmoid function

![image](https://user-images.githubusercontent.com/4158204/152093670-008d28cf-cfb3-4398-b753-668938c08855.png)

  #####                      where σ(θ^T*x^i) is the sigmoid function. Now for n observations,                          

![image](https://user-images.githubusercontent.com/4158204/152093848-6b573096-e57e-44cf-b2a7-97d2d1a7af61.png)

In machine learning, it is conventional to minimize a loss(error) function via gradient descent, rather than maximize an objective function via gradient ascent. If we maximize this above function then we’ll have to deal with gradient ascent to avoid this we take negative of this log so that we use gradient descent. We’ll talk more about gradient descent in a later section and then you’ll have more clarity. Also, remember,

####                                    max[log(x)] = min[-log(x)]

The negative of this function is our cost function and what do we want with our cost function? That it should have a minimum value. It is common practice to minimize a cost function for optimization problems; therefore, we can invert the function so that we minimize the negative log-likelihood (NLL). So in logistic regression, our cost function is:

![image](https://user-images.githubusercontent.com/4158204/152094072-7fa8add8-62e9-4e1c-8021-b975e92da591.png)

 #####                            Here y represents the actual class and log(σ(θ^T*x^i) ) is the probability of that class.
 
 * p(y) is the probability of 1.

 * 1-p(y) is the probability of 0.

Let’s see what will be the graph of cost function when y=1 and y=0

![image](https://user-images.githubusercontent.com/4158204/152094517-15063030-39af-46b3-96a7-f3a121a71ddd.png)

If we combine both the graphs, we will get a convex graph with only 1 local minimum and now it’ll be easy to use gradient descent here

![image](https://user-images.githubusercontent.com/4158204/152094634-385e6176-5c44-4293-abb4-3698903fc1b1.png)

The red line here represents the 1 class (y=1), the right term of cost function will vanish. Now if the predicted probability is close to 1 then our loss will be less and when probability approaches 0, our loss function reaches infinity.

The black line represents 0 class (y=0), the left term will vanish in our cost function and if the predicted probability is close to 0 then our loss function will be less but if our probability approaches 1 then our loss function reaches infinity.

![image](https://user-images.githubusercontent.com/4158204/152095627-37466743-1cc0-4aa7-ba4f-a352d01db9f7.png)

This cost function is also called log loss. It also ensures that as the probability of the correct answer is maximized, the probability of the incorrect answer is minimized. Lower the value of this cost function higher will be the accuracy.

## Gradient Descent Optimization

In this section, we will try to understand how we can utilize Gradient Descent to compute the minimum cost.

Gradient descent changes the value of our weights in such a way that it always converges to minimum point or we can also say that, it aims at finding the optimal weights which minimize the loss function of our model. It is an iterative method that finds the minimum of a function by figuring out the slope at a random point and then moving in the opposite direction.

![image](https://user-images.githubusercontent.com/4158204/152096025-45fb9012-9e61-4d57-89b9-e967875f7e9e.png)

The intuition is that if you are hiking in a canyon and trying to descend most quickly down to the river at the bottom, you might look around yourself 360 degrees, find the direction where the ground is sloping the steepest, and walk downhill in that direction.

At first gradient descent takes a random value of our parameters from our function. Now we need an algorithm that will tell us whether at the next iteration we should move left or right to reach the minimum point. The gradient descent algorithm finds the slope of the loss function at that particular point and then in the next iteration, it moves in the opposite direction to reach the minima. Since we have a convex graph now we don’t need to worry about local minima. A convex curve will always have only 1 minima.

We can summarize the gradient descent algorithm as:

![image](https://user-images.githubusercontent.com/4158204/152096217-54534220-ba71-42ca-8c1d-3f47f2629906.png)

Here alpha is known as the learning rate. It determines the step size at each iteration while moving towards the minimum point. Usually, a lower value of “alpha” is preferred, because if the learning rate is a big number then we may miss the minimum point and keep on oscillating in the convex curve
![image](https://user-images.githubusercontent.com/4158204/152096306-e966a5ca-0a6e-4301-a512-cf55b1fcf964.png)

Now the question is what is this derivative of cost function? How do we do this? Don’t worry, In the next section we’ll see how we can derive this cost function w.r.t our parameters.

## Derivation of Cost Function:

Before we derive our cost function we’ll first find a derivative for our sigmoid function because it will be used in derivating the cost function.

![image](https://user-images.githubusercontent.com/4158204/152096505-ac29e1a4-e9c6-4931-a426-d156efcce39a.png)

Now, we will derive the cost function with the help of the chain rule as it allows us to calculate complex partial derivatives by breaking them down.

* Step-1: Use chain rule and break the partial derivative of log-likelihood.

![image](https://user-images.githubusercontent.com/4158204/152097142-b94bcf3e-f206-43c8-ad52-2deb0efc1efc.png)           
       
* Step-2: Find derivative of log-likelihood w.r.t p

![image](https://user-images.githubusercontent.com/4158204/152097293-11fd9f41-3830-439c-a503-9f5fe8337a9a.png)


* Step-3: Find derivative of ‘p’ w.r.t ‘z’

![image](https://user-images.githubusercontent.com/4158204/152097461-1bfcb86b-8a56-4c06-bfc6-21425b9e1989.png)

* Step-4: Put all the derivatives in equation 1

![image](https://user-images.githubusercontent.com/4158204/152097567-78cb267a-e2e9-4822-a0e1-7bb922e26014.png)


* Step-5: Find derivate of z w.r.t θ
               
![image](https://user-images.githubusercontent.com/4158204/152097782-35fb4457-5150-45a6-8591-4dc80b983b9e.png)
              
              
              
    Hence the derivative of our cost function is:
                        
![image](https://user-images.githubusercontent.com/4158204/152098005-1a46147d-18ab-4122-9e67-d6eb20ace48d.png)


Now since we have our derivative of the cost function, we can write our gradient descent algorithm as:

 
If the slope is negative (downward slope) then our gradient descent will add some value to our new value of the parameter directing it towards the minimum point of the convex curve. WhereaHence the derivative of our cost function is:s if the slope is positive (upward slope) the gradient descent will minus some value to direct it towards the minimum point.
* 


## Implementation of logistic Regression using Python CODE

When we are implementing Logistic Regression Machine Learning Algorithm using sklearn, we are calling the sklearn’s methods and not implementing the algorithm from scratch.
In this article, I will be implementing a Logistic Regression model without relying on Python’s easy-to-use sklearn library. This post aims to discuss the fundamental mathematics and statistics behind a Logistic Regression model. I hope this will help us fully understand how Logistic Regression works in the background.


 ### Import the required libraries
 
 
##### import numpy as np
##### import pandas as pd
##### import seaborn as sns



### Load the classification data


##### df = pd.read_csv('Logistic-Regression-Data.csv')
##### df.head()

![lg1](https://user-images.githubusercontent.com/4158204/161000837-afde5035-b5fb-4e0c-a12d-997b8b173860.JPG)



### Separate the features and label



##### x = df[['Glucose','BloodPressure']]
##### y = df['Diabetes']
##### x

![lg2](https://user-images.githubusercontent.com/4158204/161003534-44db64c0-a607-46d3-87cc-791c46b97598.JPG)



### Define the sigmoid function


##### def sigmoid(input):    
  ##### output = 1 / (1 + np.exp(-input))
  ##### return output
    
    
    
 ### let us define the optimization function
 
 ##### def optimize(x, y,learning_rate,iterations,parameters): 
   ##### size = x.shape[0]
   ##### weight = parameters["weight"] 
   ##### bias = parameters["bias"]
##### for i in range(iterations): 
   ##### sigma = sigmoid(np.dot(x, weight) + bias)
   ##### loss = -1/size * np.sum(y * np.log(sigma)) + (1 - y) * np.log(1-sigma)
   ##### dW = 1/size * np.dot(x.T, (sigma - y))
   ##### db = 1/size * np.sum(sigma - y)
   ##### weight -= learning_rate * dW
   ##### bias -= learning_rate * db 
    
   ##### parameters["weight"] = weight
   ##### parameters["bias"] = bias
   ##### return parameters
    
### Initialize the weight and bais


##### init_parameters = {} 
##### init_parameters["weight"] = np.zeros(x.shape[1])
##### init_parameters["bias"] = 0


### Define the train function


##### def train(x, y, learning_rate,iterations):
   ##### parameters_out = optimize(x, y, learning_rate, iterations ,init_parameters)
   ##### return parameters_out
   
   
### Train the model

##### parameters_out = train(x, y, learning_rate = 0.02, iterations = 500)
##### parameters_out

### Predict using the trained model
##### output_values = np.dot(x[:10], parameters_out["weight"]) + parameters_out["bias"]
##### predictions = sigmoid(output_values) >= 1/2
##### predictions

![lg3](https://user-images.githubusercontent.com/4158204/161013142-727e9f71-0fae-4e78-9028-70fb0c0e9070.JPG)
