import os
import csv
import time
import multiprocessing

import pandas as pd
from joblib import Parallel, delayed, wrap_non_picklable_objects
import sklearn

from clean_training_data import preprocess_fasttext
from clean_training_data import preprocess_fasttext


def concat_lists(lists):
    concatd = []
    for list_ in lists:
        concatd += list_
    return concatd


@delayed
@wrap_non_picklable_objects
def func_async_wrapped(i, *args):
    data, step, tokenizer = args
    return [tokenizer(text) for text in data[i:i + step]]


def execute(
    n_cores, end, step, data,
    tokenizer, verbose=0
):
    return \
        Parallel(n_jobs=n_cores, verbose=verbose)(
            func_async_wrapped(i, data, step, tokenizer)
            for i in range(0, end, step))


start = time.time()

df = pd.read_parquet('data/unsupervised_data/train.parquet')
data = df['user.description']

n_cores = max(multiprocessing.cpu_count() - 1, 1)
preprocessed = concat_lists(execute(
    n_cores, len(data), len(data) // n_cores, data,
    preprocess_fasttext, verbose=2))
df = pd.DataFrame()
df['text'] = [i for i in preprocessed if i != '']

train, test = sklearn.model_selection.train_test_split(
    df, test_size=.2, random_state=42, shuffle=True)

train, val = sklearn.model_selection.train_test_split(
    train, test_size=.2, random_state=42, shuffle=True)

train.to_csv(
    os.path.join('data/unsupervised_data/fasttext_unsupervised_train.txt'),
    index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE,
    quotechar='', escapechar=' '
)

val.to_csv(
    os.path.join('data/unsupervised_data/fasttext_unsupervised_val.txt'),
    index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE,
    quotechar='', escapechar=' '
)

test.to_csv(
    os.path.join('data/unsupervised_data/fasttext_unsupervised_test.txt'),
    index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE,
    quotechar='', escapechar=' '
)

print(
    "Time elapsed since start: ",
    time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))
