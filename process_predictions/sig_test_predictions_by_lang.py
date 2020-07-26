"""Analyse language effects of test predictions"""
import os
import pandas as pd
import langdetect 
from functools import lru_cache
import scipy.stats

test_output = os.path.join('..', 'text_classification', 'output', 'test_category_multilang', 'test_output.csv')

def detect_lang(text):
    try:
        return langdetect.detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        return None

@lru_cache(maxsize=1)
def get_data():
    df = pd.read_csv(test_output)
    df['is_correct'] = df['label'] == df['prediction']
    df = df.dropna(subset=['text'])
    df['lang'] = df.text.apply(detect_lang)
    return df

def main():
    df = get_data()
    accuracy = df.is_correct.mean()
    lang_counts = df.lang.value_counts()
    num_correct = df.groupby('lang').is_correct.sum()
    num_total = df.groupby('lang').is_correct.count()
    num_tests = len(num_total)
    for correct, total, lang in zip(num_correct, num_total, df.groupby('lang').groups.keys()):
        test = scipy.stats.binom_test(correct, total, accuracy, alternative='two-sided')
        is_sig = test < (.05/num_tests)
        print(f'Lang {lang}: {test:.3f} (is_sig: {is_sig})')

if __name__ == "__main__":
    main()
