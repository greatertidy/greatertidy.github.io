---
layout: default
usemathjax: true
---
This is mostly going to be a collection of things I learn and work on
independently. Storing it mainly for my personal reference.

---

# [Flight Delay Analysis](/posts/flights.html)

I found [this dataset](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018)
on flight delays for US domestic flights from 2009 to 2018. I decided to perform some visual
analysis using tableau. To accomplish this I needed to clean and query the dataset
as the dataset as is was quite large (6M+ rows for each year). To do this I used
SQL and python's pandas library.
[![Flight Overview](/assets/img/flights/flights.JPG)](/posts/flights.html)

---


# [Linear Regression](/posts/linreg.html)

I decided to review some basic material on linear regression to refresh my interpretation and understanding of the model. First I go over what the model is (and how it differs from other types of models) and give an interpretation of what the actual equation and mathematics behind it represent. Then a derivation of the model in the univariate and multivariate case as well as an implementation in python using real datasets.

[![linear regression](/assets/img/linreg/prob-plot.png)](/posts/linreg.html)
