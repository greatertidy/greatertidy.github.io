---
layout: default
usemathjax: true
---

# Linear Regression
  Many types of mathematical functions are used to model or describe phenomena
we observe around us. The simplest and most intuitive function is the
linear function

$$
y = \beta_0 + \beta_1x
$$

which attempts to relate two variables, $$x$$ and $$y$$, by
a linear relationship with parameters $$\beta_0$$ and $$\beta_1$$. Our model
allows us to plug in suitable values for $$x$$ (or $$y$$) and returns a
unique and unambiguous prediction for the value of $$y$$ (or $$x$$). A model
containing this characteristic is what's called a deterministic model.

# Deterministic vs Probablistic Models
When a physicist is interested in calculating the force needed to accelerate
an object with some fixed mass they use Newton's second law

$$
F = m \cdot a
$$


![F=ma](/assets/img/linreg/force-acceleration.png)
which (practically speaking) gives us an exact value for the force given
some value for acceleration we plug into our function (i.e. is a deterministic
model).

  This differs from a probabilistic model which **does** allow some error in
predicting $$y$$. Consider the following plot.
![data with error](/assets/img/linreg/prob-plot.png)
It is clear that as $$x$$ increases, we **expect** $$y$$ to also increase according
to some linear factor. However, there is some **error** in our prediction for $$y$$.
In this scenario we can repeatedly sample a value for some value of $$x$$, say 4,
and for each sample we will get different values of $$y$$. We can model this using a
probabilistic model.

## Probabilistic Models
  To model a probabilistic phenomena we need to incorporate error or deviations into
our formula so that we can better understand how we can go about fitting our model
as well as interpreting it.

$$
Y = \beta_0 + \beta_1x + \epsilon.
$$

  Here $$\epsilon$$ is a random variable denoting our error (note $$y$$ becomes $$Y$$
signifying that it is now being treated not as an ordinary variable but as a *random*
variable). In this situation, instead of modelling $$Y$$ as a linear function of
$$x$$, we can model the expectation of $$Y$$ given some value $$x$$, denoted
$$E[Y|X=x]$$, as a linear function of $$x$$ (because we are "given" $$x$$ we don't
need to consider it as random even if in reality it may be). Under this premise
of trying to model the expectation we can configure our model to be a deterministic
one. That is, we can write

$$
E[Y|X=x] = \beta_0 + \beta_1x
$$

without any error random variable $$\epsilon$$. The interpretation being that
as $$x$$ increases, the expected value of $$Y$$ changes according to a linear
coefficient $$\beta_1$$ without any error (this is a much more abstract way
of interpreting a probabilistic model, since we can not truly measure or observe
the exact value of $$E[Y|X=x]$$ like a physicist could measure force or acceleration).

  With all this in mind we can better understand what our probabilistic model
represents. Consider a simple example like before, but instead let's fix some values.
Assume that, $$\beta_0=2$$, $$\beta_1=1$$ and the error random variable
$$\epsilon$$ is uniform distributed on the interval (-1,1).

$$
Y = 2 + x + \epsilon
$$

![line with y regions](/assets/img/linreg/ydomain.png)

  It is clear that our $$y$$ data points exist in the interval (2, 9), but when we
fix a point, say $$x=4$$, the possible values of y are instead restricted to the
interval (3,5), since the random error variable is bounded in the area (-1,1).
Further, the error is *uniformly* distributed on this interval, meaning that all
regions in the support (provided all these regions are equal size) are equally
probable. So for $$Y_{|X=4}$$ we obtain a density as follows.

![uniform density](/assets/img/linreg/uniformdensity.png)

  This is a rather simple example, usually we assume that our error random variable
$$\epsilon$$ is normally distributed with mean 0 and standard deviation $$\sigma$$.
Thus, the error is unbounded and is more likely to be closer to the center than in
the tails as opposed to equally probable everywhere. Using, the same parameters
before, for $$x=4$$ our density of $$Y_{|X=4}$$ instead becomes the more familiar
bell shaped density.

![normal density](/assets/img/linreg/normaldensity.png)

  Notice the density is still centered around 6 since we always maintain the assumption
that $$E[\epsilon]=0$$.
# Ordinary Least Squares
Now we will derive optimal predictions for our parameters $$\beta_0$$ and
$$\beta_1$$ according to least squares. We will label our predictions with a "hat",
$$\hat{\beta_0}$$ and $$\hat{\beta_1}$$. Then we will use these formulas to
fit a model on some data.

## Simple Linear Regression
  In the simple linear regression case we have that

$$
Y = \beta_0 + \beta_1x + \epsilon.
$$

with $$\epsilon$$ having 0 mean and some variance $$\sigma>0$$. We aim to predict
$$\beta_0$$ and $$\beta_1$$ with values $$\hat{\beta_0}$$ and $$\hat{\beta_1}$$
so that we can
1. better understand the relationship between $$x$$ and $$E[Y]$$
2. make predictions $$\hat{y}$$ for a given value of $$x$$.

  Least squares estimates the values of $$\beta_0$$ and $$\beta_1$$ by trying to
minimize the squared distance between predictions $$\hat{y}$$ and actual values
$$y$$. This is called the residual sum of squares or sum of squares of errors (SSE).

$$
SSE = \sum_i{y_i - \hat{y_i}} = \sum_i{[y_i-(\hat{\beta_0}+\hat{\beta_1}x_i)]^2}
$$

To choose the optimal values for $$\hat{\beta_0}$$ and $$\hat{\beta_1}$$ we can
use calculus to optimize this function with respect to these two parameters.
We have for $$\hat{\beta_0}$$

$$
\frac{\partial SSE}{\partial \hat{\beta_0}} = -2\sum_i{y_i} + 2n\hat{\beta_0} + 2\hat{\beta_1}\sum_i{x_i}
$$

and for $$\hat{\beta_1}$$

$$
\frac{\partial SSE}{\partial \hat{\beta_1}} = -2\sum_i{y_i x_i} + 2\hat{\beta_0} \sum_i{x_i} + 2\hat{\beta_1} \sum_i{x_i^2}.
$$

We want to minimize the total SSE which must happen at a critical point. So we set
the partials equal to 0. For $$\hat{\beta_0}$$

$$
0=\frac{\partial SSE}{\partial \hat{\beta_0}} = -2\sum_i{y_i} + 2n\hat{\beta_0} + 2\hat{\beta_1}\sum_i{x_i}
$$

in which we get the following value for $$\hat{\beta_0}$$

$$
\hat{\beta_0}= \frac{\sum_i{y_i} - \hat{\beta_1}\sum_i{x_i}}{n} = \bar{y} - \hat{\beta_1} \bar{x}
$$

For $$\hat{\beta_1}$$ we can plug in the value we found for $$\hat{\beta_0}$$, and
set to 0 to obtain the following.

$$
0 = \sum_i{y_i x_i} + (\bar{y} - \hat{\beta_1} \bar{x}) \sum_i{x_i} + \hat{\beta_1} \sum_i{x_i^2}
$$

Then with a little work we get our estimate.

$$
\hat{\beta_1} (\sum_i{x_i^2} - \bar{x} \sum{x_i}) = \sum_i{y_i x_i} - \bar{y} \sum_i{x_i} \\
\hat{\beta_1} = \frac{\sum_i{y_i x_i} - \bar{y} \sum_i{x_i}}{\sum_i{x_i^2} - n \bar{x}^2} \\
\hat{\beta_1} = \frac{\sum_i{y_i x_i} - \bar{y} \sum_i{x_i} - \bar{y} \sum{x_i} + \bar{y} \sum_i{x_i}}{\sum_i{x_i^2} - n \bar{x}^2 - n \bar{x}^2 + n \bar{x}}^2 \\
\hat{\beta_1} = \frac{\sum_i{y_i x_i} - \bar{y} \sum_i{x_i} - \bar{x} \sum{y_i} + n \bar{x} \bar{y}}{\sum_i{x_i^2} - 2 \bar{x} \sum_i{x_i} + n \bar{x}^2} \\
\hat{\beta_1} = \frac{\sum_i{(y_i - \bar{y}) (x_i - \bar{x})}}{\sum_i{(x_i - \bar{x})^2}}
$$

Here we have obtained the unique critical point, now we need to ensure that this choice
$$(\hat{\beta_0}, \hat{\beta_1})$$ is a local minimum (and thus a global minimum).
We can do this by checking the second partial derivatives.

$$
\frac{\partial SSE}{\partial^2 \hat{\beta_0}} = 2n \\
\frac{\partial SSE}{\partial^2 \hat{\beta_1}} = 2\sum_i{x_i^2} \\
\frac{\partial SSE}{\partial \hat{\beta_0} \partial \hat{\beta_1}} = 2 \sum_i{x_i}
$$

First we have that $$\frac{\partial SSE}{\partial^2 \hat{\beta_0}}>0$$, then we can show that

$$
(\frac{\partial SSE}{\partial^2 \hat{\beta_0}}) (\frac{\partial SSE}{\partial^2 \hat{\beta_1}}) - (\frac{\partial SSE}{\partial \hat{\beta_0} \partial \hat{\beta_1}})^2= 4n \sum_i{x_i^2} - 4 (\sum_i{x_i^2}) \\
= 4n (\sum_i{x_i^2} - \bar{x} \sum_i{x_i}) = 4n(\sum_i{(x_i - \bar{x})^2}) > 0.
$$

Thus $$(\hat{\beta_0}, \hat{\beta_1})$$ is a global minimum.

### Python Algorithm
  We can create a function in python to test this result.

```python
def simplelr(x,y):
    # naively compute least squares estimator

    # compute mean of x and y
    y_bar = np.mean(y)
    x_bar = np.mean(x)

    # first calculate beta 1
    b_1 = np.sum((y-y_bar) * (x-x_bar)) # numerator first
    b_1 = b_1/np.sum(np.square((x-x_bar)))

    # now beta 0 is given by
    b_0 = y_bar - b_1*x_bar

    return b_0, b_1
```
To test this algorithm I found some simple data [here](https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html) relating the number of insurance claims to the total insurance payments in Sweden.

![simple linear regression](/assets/img/linreg/simplelr.png)

Here, the number of claims and the total payment were square rooted to better fit the assumption of a linear model. The key interpretation here being that the expected payments (square rooted) increase by a factor of 0.45 per claim (square rooted).

## Multiple Linear Regression
   In multiple linear regression the expectation of our random variable $$Y$$ is
modelled according to multiple covariates.

$$
Y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_m x_m + \epsilon= \sum_{j=0}^{n}{\beta_j x_j}  + \epsilon= \vec{x}^T \beta + \epsilon
$$

(Note we put $$x_0=1$$)

Here we represented our sum as the dot product of two vectors,
$$\vec{x}$$ which contains the covariates $$(1, x_1, x_2, ... , x_m)$$ and,
$$\beta$$ which contains the parameters $$(\beta_0, \beta_1, ... , \beta_m)$$. Then we can compute SSE as follows,

$$
SSE = \sum_i{(y_i - \hat{y_i})^2} = \sum_i{(y_i - \vec{x}_i^T \hat{\beta)}}
$$

which is equivalent to,

$$
SSE = (\vec{y} - X \hat{\beta})^T (\vec{y} - X \hat{\beta})
$$

where $$X$$ is an $$n$$  x  $$m$$ matrix with entry $$x_{i,j}$$ equal to the j-th covariate of the the i-th sample. To obtain the least squares estimation of $$\hat{\beta}$$ we first expand our equation.

$$
SSE = \vec{y}^T \vec{y} - (\hat{\beta}^T X^T) \vec{y} - \vec{y}(X \hat{\beta}) + \hat{\beta}^T X^T X \hat{\beta} \\
SSE = \vec{y}^T \vec{y} - 2 \hat{\beta} X^T \vec{y} + \hat{\beta}^T X^T X \hat{\beta} \\
$$

Finally, using calculus we can minimize the SSE function by taking the derivative with respect to the vector $$\hat{\beta}$$.

$$
\frac{\partial SSE}{\partial \hat{\beta}} = -2 X^T \vec{y} + 2 X^T X \hat{\beta}\\
$$

 To obtain the value that minimizes SSE we find the critical point by setting

 $$\frac{\partial SSE}{\partial \hat{\beta}} = 0$$

and then assuming the columns of $$X$$ are not collinear, we can rearrange to obtain

 $$
 \hat{\beta} = (X^T X)^{-1} X^T \vec{y} .
 $$


 Again to check this value achieves a minimum we can use the second derivative test.

 $$
 \frac{\partial^2 SSE}{\partial \hat{\beta} \partial \hat{\beta}^T} = 2 X^T X
 $$

Since $$X^T X $$ is symmetric and $$X$$ has linearly independent columns (by assumption)
we conclude that $$\frac{\partial^2 SSE}{\partial \hat{\beta} \partial \hat{\beta}^T}$$
is positive definite and thus, the least squares solution achieves a minimum.

### Python Algorithm
  Again we can test this by creating a simple function in python.

```python
def multiplelr(x,y):
    # naively compute least squares estimator
    # ((XT)(X))^-1
    beta = np.linalg.inv(np.matmul(np.transpose(x), x))
    # ((XT)(X))^-1 (XT)
    beta = np.matmul(beta, np.transpose(x))
    # ((XT)(X))^-1 (XT) y
    beta = np.matmul(beta, y)
    return beta
```

For this I used, another [insurance dataset]("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv") containing data of insurance charges of customers. Here we tried to capture the linear relationship between the age and BMI of a customer and their insurance charge.

![multiple linear regression](/assets/img/linreg/multiplelr.png)

Since for each observation we obtain a 3-dimensional vector (age, bmi, charge) our model describes a plane mapping the age and BMI of a hypothetical customer to a predicted expected insurance charge. Here, our model claims that the expected payment increases by 242 per increase in age and 333 per increase in BMI.
