### Linear Model Representation

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function h : X → Y so that h(x) is a “good” predictor for the corresponding value of y. For historical reasons, this function h is called a hypothesis. Seen pictorially, the process is therefore like this:

![linearmodel](https://cloud.githubusercontent.com/assets/8801972/21876561/f2732e74-d8a9-11e6-997f-4c8ca60e858b.png)

where 'h' is a continuous function that maps our prediction variables to the target variable. In other words, it is a curve that is generated to fit in the points (prediction_variables, target) with a minimal error rate.

The most popular algortihm that is used to getting the error rate minimal is **Gradient Descent**. Most of the supervised learning algorithms auto calculates the learning rate that is required for gradient descent from the given training dataset. More information on Gradient Descent can be found [here](https://en.wikipedia.org/wiki/Gradient_descent)

Among python libraries 'sklearn' offers this as 'linear_model' which can be imported and used as follows:

```
import sklearn.linear_model as LinearRegression
alg = LinearRegression()
alg.fit(train_predictors, targets)
predictions = alg.predict(test_predictors)
```

#### Importance of Cross Validation

We want to train the algorithm on different data than we make predictions on. This is critical if we want to avoid overfitting.

Overfitting is what happens when a model fits itself to "noise", not signal. Every dataset has its own quirks that don't exist in the full population. For example, if we were asked to predict the top speed of a car from its horsepower and other characteristics, and are given a dataset that randomly had cars with very high top speeds, we would create a model that overstated speed. The way to figure out if your model is doing this is to evaluate its performance on data it hasn't been trained using.

Every machine learning algorithm can overfit, although some (like linear regression) are much less prone to it. If you evaluate your algorithm on the same dataset that you train it on, it's impossible to know if it's performing well because it overfit itself to the noise, or if it actually is a good algorithm.

Luckily, cross validation is a simple way to avoid overfitting. To cross validate, you split your data into some number of parts (or "folds"). Lets use 3 as an example. You then do this:

* Combine the first two parts, train a model, make predictions on the third.

* Combine the first and third parts, train a model, make predictions on the second.

* Combine the second and third parts, train a model, make predictions on the first.

This way, we generate predictions for the whole dataset without ever evaluating accuracy on the same data we train our model using.

Among python libraries 'sklearn' offers this as 'cross_validation.KFold' which can be imported and used as follows:

```
from sklearn.cross_validation import KFold
kf = KFold(train_data.shape[0], n_folds=3, random_state=1)
for train, test in kf:
 # Do your processeing
```

## [Next](https://github.com/sheikirfanbasha/MachineLearning/tree/master/Excerices/01_Titanic-MachineLearningfromDisaster)
