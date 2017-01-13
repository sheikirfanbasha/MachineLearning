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

#### Note

We can not always fit the given dataset in to Straight line.
In such cases, we can make it a quadratic/cubilc/square root curve by adding more features from the given set of features
Ex. if size is a feature at hand. We can use pow(size, 2) as the second and pow(size, 3) as the third.

One important thing to keep in mind  while trying to fit in non-linear curves is "Scale of the features". They must be ensured to that they fall in the minimum range. This can be achieved by "Feature Scaling"

If Feature Scaling isn't applied then the algorithm (gradient descent) is proned to take a zig zag mannered curve that takes a very long before it converges to global minima. 

#### Normal Equation

Gradient descent gives one way of minimizing our cost function (i.e., minimal error rate). Let’s discuss a second way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm. In the "Normal Equation" method, we will minimize J by explicitly taking its derivatives with respect to the θj ’s, and setting them to zero. This allows us to find the optimum theta without iteration. The normal equation formula is given below:

**θ=(X<sup>T</sup>X)<sup>−1</sup>X<sup>T</sup>y**

There is no need to do feature scaling with the normal equation.

The following is a comparison of gradient descent and the normal equation:

| Gradient Descent | Normal Equation |
| ---------------- | --------------- |
| Need to choose alpha | No need to choose alpha |
| Needs many iterations | No need to iterate |
| O(kn<sup>2</sup>) | O(n<sup>3</sup>), need to calculate inverse of X<sup>T</sup>X |
| Works well when n is large | Slow if n is very large |

With the normal equation, computing the inversion has complexity O(n<sup>3</sup>). So if we have a very large number of features, the normal equation will be slow. In practice, when n exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.

If X<sup>T</sup>X is noninvertible, the common causes might be having :

*Redundant features, where two features are very closely related (i.e. they are linearly dependent)

*Too many features (e.g. m ≤ n). In this case, delete some features or use "regularization". where 'm' is number of training examples or no. of rows in training data and 'n' is number of features or predictors we gonna use to predict our output.

Solutions to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.

## [Next](https://github.com/sheikirfanbasha/MachineLearning/tree/master/Excerices/01_Titanic-MachineLearningfromDisaster)
