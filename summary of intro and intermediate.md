# Summary
This is a summary of introduction to machine learning and intermediate machine learning.

## Typical setup

```python
# Here is a typical setup
import pandas as pd
from sklearn.model_selection import train_test_split
fp = "<path to dataset>"
data = pd.read_csv(fp)
# y = prediction target
y = data.<prediction target>
X = data.drop(["<target>"], axis=1)
# drop rows where there is nothing in the target
data = data.dropna(subset=["<target>"], axis=0)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
```

## Best model for structured data
XGBoost stands for Extreme Gradient Boosting.<br>
It is the most accurate technique for structured data.
## Running a model "normally"
## Cross validation