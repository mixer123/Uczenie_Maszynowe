import email
import io
import re
from collections import defaultdict
import random

import pandas as pd

from bs4 import BeautifulSoup


class EmailObject:
    def __init__(self, file, category=None):
        self.mail = email.message_from_file(file)
        self.category = category

    def subject(self):
        return self.mail.get('Subject')

    def body(self):
        content_type = self.mail.get_content_type()
        body = self.mail.get_payload(decode=False)

        if content_type == 'text/html':
            return BeautifulSoup(body, 'html.parser').text
        elif content_type == 'text/plain':
            return body
        else:
            return ''


class Tokenizer:
    NULL = u'\u0000'

    @staticmethod
    def tokenize(txt):
        return re.findall('\w+', txt.lower())

    @staticmethod
    def ngram(txt, n=2):
        s = txt.split(' ')
        result = []
        for i in range(1, n + 1):
            result.append([Tokenizer.NULL] * (n - i) + s)
        return list(zip(*result))

    @staticmethod
    def unique_tokenizer(txt):
        tokens = Tokenizer.tokenize(txt)
        return set(tokens)


class SpamTrainer:
    def __init__(self, training_files):
        self.categories = set()

        for category, file in training_files:
            self.categories.add(category)

        self.totals = defaultdict(float)
        self.training = {c: defaultdict(float)
                         for c in self.categories}
        self.to_train = training_files

    def total_for(self, category):
        return self.totals[category]

    def train(self):
        for category, file in self.to_train:
            with open(file, 'r', encoding='latin-1') as f:
                mail = EmailObject(f)
            self.categories.add(category)

            for token in Tokenizer.unique_tokenizer(mail.body()):
                self.training[category][token] += 1
                self.totals['_all'] += 1
                self.totals[category] += 1

        self.to_train = {}

    def score(self, mail):
        self.train()
        cat_totals = self.totals

        aggregates = {c: cat_totals[c] / cat_totals['_all']
                      for c in self.categories}

        for token in Tokenizer.unique_tokenizer(mail.body()):
            for cat in self.categories:
                value = self.training[cat][token]
                r = (value + 1) / (cat_totals[cat] + 1)
                aggregates[cat] *= r

        return aggregates

    def normalized_score(self, mail):
        score = self.score(mail)
        scoresum = sum(score.values())

        normalized = {cat: (agg / scoresum)
                      for cat, agg in score.items()}

        return normalized

    def preference(self):
        return sorted(self.categories, key=lambda cat: self.total_for(cat))

    class Classification:
        def __init__(self, guess, score):
            self.guess = guess
            self.score = score

        def __eq__(self, other):
            return self.guess == other.guess and self.score == other.score

    def classify(self, mail):
        score = self.score(mail)

        max_score = 0.0
        preference = self.preference()
        max_key = preference[-1]

        for k, v in score.items():
            if v > max_score:
                max_key = k
                max_score = v
            elif v == max_score and preference.index(k) > preference.index(max_key):
                max_key = k
                max_score = v
        return self.Classification(max_key, max_score)


training_data_ = (
    ('ham', './data/TRAINING/TRAIN_00002.eml'),
    ('spam', './data/TRAINING/TRAIN_00000.eml'),
    ('ham', './data/TRAINING/TRAIN_00006.eml'),
    ('spam', './data/TRAINING/TRAIN_00003.eml')
)

spam_trainer = SpamTrainer(training_data_)


def label_to_training_data(fold_file):
    """
    Funkcja zwraca wytrenowany model na wyznaczonym podzbiorze wiadomosci
    """

    training_data = []

    with open(fold_file, 'r') as f:
        for line in f:
            target, filepath = line.rstrip().split(' ')
            training_data.append([target, filepath])

    return SpamTrainer(training_data)


def parse_emails(keyfile):
    """
    Funkcja zwraca wyznaczony podzbior wiadomosci w postaci obiektow klasy EmailObject
    """

    emails = []

    with open(keyfile, 'r') as f:
        for line in f:
            label, file = line.rstrip().split(' ')

            with open(file, 'r', encoding='latin-1') as labelfile:
                emails.append(EmailObject(labelfile, category=label))

    return emails


def validate(trainer, set_of_emails):
    """
    Funkcja dokonuje walidacji wytrenowanego modelu (trainer)
    na podstawie zbioru oznaczonych wiadomosci (set_of_emails)
    """

    correct = 0
    false_positives = 0.0
    false_negatives = 0.0
    confidence = 0.0

    for mail in set_of_emails:
        classification = trainer.classify(mail)
        confidence += classification.score

        if classification.guess == 'spam' and mail.category == 'ham':
            false_positives += 1
        elif classification.guess == 'ham' and mail.category == 'spam':
            false_negatives += 1
        else:
            correct += 1

    total = false_positives + false_negatives + correct

    false_positive_rate = false_positives / total
    false_negative_rate = false_negatives / total
    error = (false_positives + false_negatives) / total

    return false_positive_rate, false_negative_rate, error


columns = ('class', 'path')
df = pd.concat(
    (pd.read_csv('../lab2/data/fold1.label', sep=' ', header=None), pd.read_csv('../lab2/data/fold2.label', sep=' ', header=None)),
    ignore_index=True
)

test_rows = random.sample(df.index.tolist(), int(round(len(df) * .3)))  # 30%
train_rows = set(range(len(df))) - set(test_rows)
df_test = df.loc[test_rows]
df_train = df.drop(test_rows)

df_train.to_csv('data/train_70.label', sep=' ', header=False, index=False)
df_test.to_csv('data/test_30.label', sep=' ', header=False, index=False)

del test_rows
del train_rows
del df_train
del df_test
del df

trainer = label_to_training_data('../lab2/data/train_70.label')
emails = parse_emails('data/test_30.label')
fpr, fnr, err = validate(trainer, emails)

print(f'FPR: {fpr}, FNR: {fnr}, error: {err}, accuracy: {1 - err}')
