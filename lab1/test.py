import random
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

metric = np.mean

houses = pd.read_csv('data/data.csv')
values = houses['AppraisedValue']
houses.drop('AppraisedValue', 1, inplace=True)
houses = (houses - houses.mean()) / (houses.max() - houses.min())
houses = houses[['lat', 'long', 'SqFtLot']]

kdtree = KDTree(houses)


# regresja
def predict(query_point, k):
    _, idx = kdtree.query(query_point, k)
    return metric(values.iloc[idx])


test_rows = random.sample(houses.index.tolist(), int(round(len(houses) * .2)))  # 20%
train_rows = set(range(len(houses))) - set(test_rows)
df_test = houses.loc[test_rows]
df_train = houses.drop(test_rows)
test_values = values.loc[test_rows]
train_values = values.loc[train_rows]

train_predicted_values = []
train_actual_values = []

for _id, row in df_train.iterrows():
    train_predicted_values.append(predict(row, 5))
    train_actual_values.append(train_values[_id])

print(mean_absolute_error(train_predicted_values, train_actual_values))

test_predicted_values = []
test_actual_values = []

for _id, row in df_test.iterrows():
    test_predicted_values.append(predict(row, 5))
    test_actual_values.append(test_values[_id])

print(mean_absolute_error(test_predicted_values, test_actual_values))



