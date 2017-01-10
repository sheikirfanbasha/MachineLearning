## Definition

The more well accepted definition of machine learning is that "It is a computer program which is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E"

## Understanding

It is usual in ML space that every task has to deal with a large amount of data at hand. Obviously one doesn't want to let his/her program visit each and every data entity and take years of time to do a simple task.

Thus, it is important to learn few alternatives to represent this huge data in more simple and descriptive ways. This is where **statistics** can play a pivotal role.

Let's start with simple stuff. Say you have a large collection of numbers, may be the age of people across the world. And lets say we are curious to find what is the avergae age that the people are living in the world. If someone needs to look at each and every person's age to come to valid conclusion then definitely it will take a lot of time.

Statistics comes handy in this context. It helps us to represent the whole collection of numbers into a simple concept called "central tendency".

In statistics, the 3 basic and most powerful ways in which "central tendancy" can be expressed are:

* Mean (more specifically Arithmetic Mean)
* Median
* Mode

Most of us often get ourselves into this confusion when to use Mean, Median and Mode. Which is useful over the other?

The answer gets more clear when we understand what each of them does to help us represent the data at our hand more precisely.

### Mean

This term is used with "average" interchangably. There are many different means like Geometric Mean, Harmonic Mean. To keep our discussions simple, we narrow down to just Arithmetic Mean.

It is used to summarize the data by identifying the average of the given data. The definition is as follows:

```
Mean = Sum of the given numbers / Total number of numbers
```

#### When to use:
Lets take an example. Consider we have the following set of data

1 1 2 3 4

Mean = (1 + 1 + 2 + 3 + 4) / 5 = 2.2

Hence, instead of giving all the numbers to the other person we can just tell that the numbers are in the form whose mean is "2.2".

The key problem comes when there is an anamoly or the data that is completely deviated from the others. Ex. Let us consider we have the following set of numbers:

1 1 2 3 100

Mean = (1 + 1 + 2 + 3 + 100) / 5 = 21.4

This mean (21.4) falsely represent the given numbers (eg. 1 is way far from 21.4) because of the existance of the number "100". 

In such cases we will normally be using one the two other alternatives "Median" or "Mode"

### Median

By definition, it is the number present in the middle when the numbers are sorted in either ascending/descending.

Eg:

For the numbers 1 1 2 3 100. The middle number is '2'. Hence the median is '2'. Which represents majority of numbers in the given list. 

Observe that the presence of anamoly has no effect in the median.

*In case if the total number of numbers is even then the mean of the middle two numbers is taken as the median.*

### Mode

By definition it is the number that is most repeated in the given list.

*In case if there is two or more numbers sharing the max count then picking up mode should be strategical*



