# NOTES

## MODELLING DATA

A dataset has too many variables for one to understand, there are two main ways to pare them down:
- intuition
- statistical analysis (later)

```python
import pandas as pd
fp = "<filepath>"
data = pd.read_csv(fp)
data.columns # gets columns to choose from
```

When going through data, there might be missing values, for now just remove them via this:
```python
data = data.dropna(axis=0)
```

### prediction target
the prediction target is the variable that we are trying to predict with our model.

```python
y = data.<columnName> # prediction target is named y normally
```

### features
The columns inputted into the model to make predictions with are called features.

```python
features = [...]
X = data[features] # features are named X normally
```

## Build a model
There are some steps to building a model: <br>
1. Define - what type of model is it? Some other parameters of the model are decided here aswell
2. Fit - capture patterns from provided data. The "heart" of modelling
3. Predict - predict things
4. Evaluate - see how good the predictions are.

### Example model
```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=1)
# random_state = 1 means that the model wont change no matter how much you run on a dataset, to change the answer you must change the model or the dataset.
model.fit(X, y) # fit the data
```

## Model validation
there are many metrics for model quality, we'll start with a simple one: <br>
- Mean Absolute Error (MAE)
  - for one result: error = actual - predicted
  
- MAE means that on average, the predictions were off by n amount

```python
from sklearn.metrics import mean_absolute_error
predicted_from_data = model.predict(X)
mean_absolute_error(y, predicted_from_data)
```

Note: to validate properly, you need to split the dataset into training and validation sets.

```python
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
model = DecisionTreeRegressor()
model.fit(train_X, train_y)
predictions = model.predict(val_X)
print(mean_absolute_error(val_y, predictions))
```
