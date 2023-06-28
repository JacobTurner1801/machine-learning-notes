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

## Overfitting and Underfitting
- Overfitting is when the model learns on the training data "too well" so that it predicts with high accuracy for the training data, but does terribly with the validation data. <br>
- Underfitting is when the model fails to capture important distinctions / patterns in the data, so it preforms poorly on the training data and also on the validation data.

- in a decision tree model, either one of these happen when the depth of the decision tree is incorrect, for overfitting, it's too deep, and for underfitting it's too shallow.

Here is an example of how to limit the depth.<br>
1. create utility function for mean absolute value.
```python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```

2. Use loop to compare the accuracy of the model for each different depth.
```python
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
"""OUTPUT
Max leaf nodes: 5  		 Mean Absolute Error:  347380
Max leaf nodes: 50  		 Mean Absolute Error:  258171
Max leaf nodes: 500  		 Mean Absolute Error:  243495
Max leaf nodes: 5000  		 Mean Absolute Error:  254983
"""
```

3. decide which depth to use: here it's 500.

4. use in model.
```python
final_model = DecisionTreeRegressor(max_leaf_nodes=500, random_state=1)
```

## Random Forests
- Use many trees, prediction made by averaging each component tree.

```python
# assume data has been split already
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
prediction = forest_model.predict(val_X)
```
