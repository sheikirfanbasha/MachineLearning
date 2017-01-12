Linear Model Representation

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function h : X → Y so that h(x) is a “good” predictor for the corresponding value of y. For historical reasons, this function h is called a hypothesis. Seen pictorially, the process is therefore like this:

![linearmodel](https://cloud.githubusercontent.com/assets/8801972/21876561/f2732e74-d8a9-11e6-997f-4c8ca60e858b.png)

where 'h' is a continuous function that maps our prediction variables to the target variable. In other words, it is a curve that is generated to fit in the points (prediction_variables, target) with a minimal error rate.

The most popular algortihm that is used to getting the error rate minimal is **Gradient Descent**. Most of the supervised learning algorithms auto calculates the learning rate that is required for gradient descent from the given training dataset. More information on Gradient Descent can be found ![here](https://en.wikipedia.org/wiki/Gradient_descent)

