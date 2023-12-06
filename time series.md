# Time Series Notes

## Linear Regression With Time Series

### What is a time series?
A set of observations over time, typically this is split up into periods of days, months, years, etc.

### Linear Regression

Linear Regression learns how to make a weighted sum from its input features.
<br><br>
Example of Two Features:
```
target = weight_1 * feature_one + weight_2 * feature_2 + bias
```
<br>
During training of a linear regression model, values for weight_1, weight_2, and bias that best fit the target are learned.

<br>
<i> maths-y note: the weights are called 'regression coefficients' and the bias is called the 'intercept' because the bias tells you where the graph of this function crosses the y-axis </i><br>
<br>

### Time Series
There are 2 kinds of features unique to time series: time-step features and lag features.

Time Step Features we can derive from the time index (date column that you make the index of a dataframe)
- the most basic is called time-dummy, which associates a number 1 - n with the time steps in the series from beginning to end.

Linear regression with the time-dummy produces the model:
```
target = weight * time + bias
```
<i>
Time-step features let you model time dependence. A series is time dependent if its values can be predicted from the time they occured.</i>
<br><br>

### Lag Features
<i> current time is t </i><br>
To make a lag feature for time t+1, we take values from time t-1 as a feature to forecast for time t+1.

Linear regression with lag feature produces the model:
```
target = weight * lag + bias
```

## Trends

### What is a trend?
A persistent, long-term change in the mean of a series. <br>
<i>Doesn't have to be the mean</i><br><br>

### Moving average plots
To see what kind of trend a time series might have, we can use a moving average plot. To compute a moving average of a time series, we compute the average of the values within a sliding window of some defined width. Each point on the graph represents the average of all the values in the series that fall within the window on either side. The idea is to smooth out any short-term fluctuations in the series so that only long-term changes remain.

### Engineering Trend
Once we have the shape of a trend, we can attempt to model it using a time-step feature. Here is an example for a linear trend.
```
target = a * time + b
```
A slightly different example is if the trend is quadratic.
```
target = a * time^2 + b * time + c
```
<i>Linear regression will learn values for a, b, and c</i>
<br><br>

## Seasonality

