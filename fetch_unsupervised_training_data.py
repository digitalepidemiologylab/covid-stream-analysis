import logging
import random

import pandas as pd
import langdetect

from utils.helpers import get_parsed_data


logger = logging.getLogger(__name__)


def new_detect(s):
    try:
        return langdetect.detect(s)
    except langdetect.lang_detect_exception.LangDetectException:
        return None


def read_preprocess_data(f_name):
    """Reads and preprocesses a single parquet file"""
    if usecols is not None:
        assert 'user.name' in usecols
        assert 'user.description' in usecols
        assert 'lang' in usecols
    # lens = []
    # Read
    df = pd.read_parquet(f_name, columns=usecols)
    # print(f"READ! {random.random()}")
    # lens.append(len(df))
    # Drop nans and duplicates
    df = df.drop_duplicates(subset=['user.name', 'user.description'])
    df = df.dropna(subset=['user.name', 'user.description'])
    # print(f"DROPPED! {random.random()}")
    # lens.append(len(df))
    # Filter lang
    df = df[df.lang == 'en']
    # print(f"FILTERED LANG! {random.random()}")
    # lens.append(len(df))
    # Langdetect
    # df['langdetect'] = df['user.description'].apply(
    #     lambda s: s if new_detect(s) == 'en' else None)
    # df = df.dropna()
    # print(f"LANGDETECTED! {random.random()}")
    # lens.append(len(df))
    # Drop nans
    df = df.dropna(subset=['user.description'])
    # print(f"FINAL DROPNA! {random.random()}")
    # lens.append(len(df))
    # logging.info(
    #     f"\n\nInit: {lens[0]}\n"
    #     f"Dropping nans and dupls: {lens[1]}\n"
    #     f"Lang filtering: {lens[2]}\n"
    #     # f"Langdetect: {lens[3]}\n"
    #     f"Final: {lens[3]}\n"
    # )
    return df


usecols = ['user.name', 'user.screen_name', 'user.description', 'lang']
s_date = '2020-05-07'
e_date = '2020-06-07'
df = get_parsed_data(
    usecols=usecols, s_date=s_date, e_date=e_date,
    read_in_parallel=True, num_files=None,
    read_data_func=read_preprocess_data, verbose=2, final_dropdupl=True)
df = df.drop_duplicates(subset=['user.name', 'user.description'])
df.to_parquet('data/unsupervised_data/train.parquet')
