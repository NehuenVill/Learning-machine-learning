# Machine learning

## What's machine learning?

It's a subdomain of computer science that focuses on creating algorithms that help computers learn form excisting data to take certain actions without the need of a programmer telling the computer what to do.

## What is AI?

Artificial intelligence, the goal is to make computers able to perform or simulate human behavior.
Machine learning is an AI branch.

## What's Data science?

Field that works toward finding patterns and draw conclusions from data.

# Types of machine learning

## Supervised learning

Here you give labeled inputs to your computer, which have a certain output.
With a huge amount of these inputs you train your computer to predict the next output based on the input.

The inputs info are called feature vectors.

### Types of features:

1. Qualitative: categorical data, there are a certain number of categories or groups (i.e. Eye colors, gender, and so on).

There's _nonimal data_ which means there's no inherent order in the different parts of the categories or groups.

To feed your model with this type of data you normally use the _one-hot encoding_ method.

And there's _ordinal data_ which means there's an order in the different parts of the categories or groups (i.e. age groups, test scores/grades).

To feed your model with this type of data you normally give a number/value ascendingly to each one of the parts of the category.

2. Quantitative: Data which value is numerical (length, temperature), there's continuous and discrete types.

### Supervised learning tasks:

1. Classification: To predict discrete classes.There's multiclass classification (where there are more than one class i.e dog breeds) and binary classification (Where there's only two classes i.e. Spam/not Spam, positive or negative).

2. Regression: To predict continuous values.

### Training the model

#### Clearing out some concepts:

- Sample: Each row of data (think about it like a data frame).

- Feature: Each column of the data set.

- Output label: The output value from a sample of data (which our model will try to predict after it's trained). (this is also the *Target* for our *feature vector*)

- Feature vector: The row of data, without the output, with all its features.

- Features matrix (x): All the feature vectors we're gonna fed our model with in a matrix.

- Target vector (y): The outputs for all the feature vectors. 

#### How we split our data to feed our model:

- Training dataset: 60% would be fine (the most of the data is used here).

- Validation dataset: 20%

- Testing dataset: 20%

#### Training the model with the training dataset:

We feed this dataset to our model which will return a vector of prediction corresponding with each sample from the datase.

Then we must figure out the difference between our prediction and the desired output values, this difference is called *loss*.

Finally we make adjustments to the model based on the previous results.

#### Validation set:

We feed our model with our validation set, and check the loss to determinate whether it can handle unseen data or not.

Adjustments to the model are not made on this stage, thus there's no feedback from the loss.

Here we're gonna choose the model with the lowest loss.

#### Test set:

We take our best performing model and run the test set through our model, to see how generalizable our choosen model is. This is the dinal check.

The loss here will be the final reported performance of our model.

## unsupervised learning

Here we use unlabeled data to discover patterns in it.

## Reinforcement learning

Here the machine learns in an interactive environment, based on rewards and penalties.

# The machine learning cycle

![Machine learning cycle from Python Jedi youtube channel](C:/Users/nehue/Documents/programas_de_python/machine_learning/ml_cycle.png)

# Loss:

- L1 Loss: loss = sum(|y<sub>real</sub> - y<sub>predicted</sub>|)

- L1 Loss: loss = sum((y<sub>real</sub> - y<sub>predicted</sub>)<sup>2</sup>)

# Other metrics of performance:

## Accuracy

The rate to which the model returns the right output for the feature sample.

## Precision

The amount of predicted outputs of a certain label that we know were predicted right over the total amount of predicted outputs of this label (both right and wrong).

## Recall

The amount of predicted outputs of a certain label that we know were predicted right over the total amount of this labels available on our test dataset

# Models

## K-Nearest Neighbors:

In this models you look at different labeled inputs near your new and unknown input (the one to predict) and assign it the output that's repeated the most on the samples around it.

### Steps of the K-NN:

1. Define a distance between the points and classify them (in two dimentional graphs this is known as Euclidean distance).

K = how many neighbors closest to our new input should we based the model prediction on?

### Naive Bayes

#### Bayes' rule

What's the probability of some event *A* happening, given that an even *B* happened.

Formula:

$$ P(A|B) = {P(B|A) . P(A)\over P(B)} $$

P(A|B): The probability of A happening given that B happened.

B: This is our condition, something we know has happened.

#### Naive bayes general formula:

$$ P(C_k|x) = {P(x|C_k) . P(C_k)\over P(x)} $$

Where x is our feature vector.

$$ C_k $$ are the different categories up to *k*.

$$ P(C_k|x) $$ is the probability of X fitting in the category $$ C_k $$ it's called the *posterior*.

$$ P(x|C_k) $$ is the likelihood of x fitting $$ C_k $$ considering that x belongs to that class/category.

$$ P(C_k) $$ is the *prior* which means, what's the probability of the whole class itself, what's the probability of the $$ C_k $$ existing in the general dataset.

P(x) is the *evidence*, the probability of the *x* feature vector existing in the data set.

#### Naive Bayes' rule:

$$ P(C_k|x_1, x_2, ..., x_n) \propto P(C_k) . \prod_{i=1}^{n}P(x_i|C_k) $$

$$ P(C_k|x_1, x_2, ..., x_n) $$ : What's the probability of that we're on some class $$ C_k $$ given that we have up to $$ x_n $$ feature vectors.

In naive bayes we assume that all the different features of our data are independent, the are not linked to each other.

##### Predicted output (ŷ):

$$ ŷ = argmax(P(C_k|x_1, x_2, ..., x_n)) $$ 

argmax returns the argument (k in this case) which maximises the right's expression, which makes it the largest value.

$$ k  \epsilon  \{1, k\}$$ which are all the existing categories/classes in the dataset.

So what we do is to solve the expression for each *k* and find the one that makes that the largest.

According to the naive bayes' rule we can write the formula for ŷ like this:

$$ ŷ = argmax(P(C_k) . \prod_{i=1}^{n}P(x_i|C_k)) $$ 

Finding the *k* value which makes the right's expression the largest is called _MAP_ (Maximum A Posteriori).

