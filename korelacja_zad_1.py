# W pliku pracownicy.csv. Oblicz współczynniki korelacji liniowej między zarobkami
# we wszystkich  miesiącach (jedna tabela zmiennych).
# Oblicz współczynniki korelacji między stażem pracy i zarobkami
# w poszczególnych miesiącach a także rocznymi zarobkami. (dwie listy zmiennych)

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as scs
import numpy as np

dane_z_miesiace_df = pd.read_csv('files/pracownicy.csv', sep=';', encoding_errors= 'replace')
# print(list(dane_z_miesiace_df.columns[6:18]))
macierz_df = dane_z_miesiace_df[list(dane_z_miesiace_df.columns[6:18])]
macierz_korelacji_df = macierz_df.corr().round(2)
print(macierz_korelacji_df)
