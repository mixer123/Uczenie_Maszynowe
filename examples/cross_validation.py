import pandas as pd
import numpy as np
from random import shuffle

from sklearn.metrics import accuracy_score


class Model:
    """
    przykladowa klasa modelu klasyfikatora, zwraca losowe decyzje, model nie jest uczony
    """

    def __init__(self, ds):
        self.ds = ds

    def train(self):
        pass

    def predict(self, obj):
        return np.random.randint(self.ds.iloc[:, -1].max())


df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))  # losowy system decyzyjny

idx = list(range(len(df)))
shuffle(idx)  # tasowanie obiektow w systemie decyzyjnym

folds = 5
chunks = np.array_split(idx, folds)  # podzial systemu decyzyjnego na 5 folds

accs = []

for i in range(folds):
    train_folds = set(range(folds)) - {i}  # folds treningowe
    test_folds = i  # fold testowy

    train_idx = np.take(chunks, list(train_folds), axis=0).flatten()  # indeksy obiektow w systemie treningowym
    test_idx = chunks[test_folds]  # i testowym

    train_df = df.iloc[train_idx, :]  # system treningowy
    test_df = df.iloc[test_idx, :]  # system testowy

    model = Model(train_df)  # utworzenie modelu losowego
    model.train()  # pseudo-trening modelu

    scores = []  # lista decyzji

    for obj in test_df.iterrows():  # wyznaczenie dokladnosci testu
        score = model.predict(obj[:-1])

        scores.append(score)

    accs.append(accuracy_score(scores, test_df.iloc[:, -1].values))

print(f'srednia dokladnosc: {np.mean(accs)}, odch std: {np.std(accs)}')  # wyswietlenie usrednionych wynikow
