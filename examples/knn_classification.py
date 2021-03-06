import random

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.metrics import log_loss, accuracy_score

houses = pd.read_csv('../lab1/data/data.csv')

values = houses['WaterSystem']
houses.drop('WaterSystem', 1, inplace=True)

houses = (houses - houses.mean()) / (houses.max() - houses.min())
houses = houses[['lat', 'long', 'SqFtLot', 'AppraisedValue']]

kdtree = KDTree(houses)


def classify(query_point, k):
    _, idx = kdtree.query(query_point, k)
    return np.argmax(np.bincount(values.iloc[idx]))


test_rows = random.sample(houses.index.tolist(), int(round(len(houses) * .3)))  # 30%
train_rows = set(range(len(houses))) - set(test_rows)
df_test = houses.loc[test_rows]
df_train = houses.drop(test_rows)
test_values = values.loc[test_rows]
train_values = values.loc[train_rows]

train_predicted_values = []
train_actual_values = []

for _id, row in df_train.iterrows():
    train_predicted_values.append(classify(row, 17))
    train_actual_values.append(train_values[_id])

print(log_loss(train_predicted_values, train_actual_values))
print(accuracy_score(train_predicted_values, train_actual_values))
