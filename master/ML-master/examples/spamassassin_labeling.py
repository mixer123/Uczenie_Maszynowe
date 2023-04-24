import glob

import pandas as pd

paths = ('easy_ham', 'spam')  # sciezki do folderow

list_ = []  # lista odczytanych sciezek i ich etykiet

for path in paths:
    for file in glob.glob(f'{path}/*.*'):
        list_.append((path, file))

df = pd.DataFrame(list_)

df.to_csv('all.label', header=False, index=False, sep=' ')  # zapis etykiet do pliku
