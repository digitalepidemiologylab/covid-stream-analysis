import logging
import random
import os
import pandas as pd
from tqdm import tqdm, trange
from utils.helpers import get_parsed_data
from utils.process_tweet import ProcessTweet
import datetime
import multiprocessing
import joblib

import sys
try:
    sys.path.append('/mount/SDF/archive/preprocess')
    from utils.helpers import get_parsed_data as get_parsed_data_archive
except:
    print('Archive not found')
    sys.exit()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')

# globals
usecols = ['user.screen_name', 'user.id', 'user.description', 'lang']
verbose = 2
max_examples_per_file = int(1e6)
num_files = None
train_data_path = os.path.join('data', 'annotation_data', 'annotated_users_ten_languages-2.pickle')

def read_covid_data():
    logger.info('Reading covid data')
    s_date = '2020-05-07'
    e_date = '2020-06-07'
    df = get_parsed_data(
        usecols=usecols, s_date=s_date, e_date=e_date,
        read_in_parallel=True, num_files=num_files, verbose=verbose)
    df = df[df.lang == 'en']
    return df

def read_archive_data():
    logger.info('Reading archive data')
    s_date = '2019-05-07'
    e_date = '2019-06-07'
    df = get_parsed_data_archive(
        usecols=usecols, s_date=s_date, e_date=e_date,
        read_in_parallel=True, num_files=num_files, verbose=verbose)
    df = df[df.lang == 'en']
    return df

def write_output_file(df, f_path):
    with open(f_path, 'w') as f:
        num_lines = len(df)
        for i, text in tqdm(df['user.description'].iteritems(), total=num_lines):
            f.write(text + '\n')
    return 1

def get_annotation_data():
    df = pd.read_pickle(train_data_path)
    df = df['screenname']
    return df

def main():
    df = pd.concat([read_covid_data(), read_archive_data()])
    logger.info(f'Read total {len(df):,} records')
    logger.info(f'Drop duplicates')
    df = df.drop_duplicates(subset=['user.id', 'user.description'])
    logger.info(f'Remaining len: {len(df):,}')
    logger.info('Drop NaN')
    df = df.dropna(subset=['user.description'])
    logger.info(f'Remaining len: {len(df):,}')
    logger.info('Min 3 characters')
    df = df[df['user.description'].str.len() > 3]
    logger.info(f'Remaining len: {len(df):,}')
    logger.info('Remove annotation data')
    df_annot = get_annotation_data()
    df = df[~df['user.screen_name'].isin(df_annot)]
    logger.info(f'Remaining len: {len(df):,}')
    logger.info('Standardize text')
    df['user.description'] = df['user.description'].apply(ProcessTweet.normalize_str)

    # write output files
    num_lines = len(df)
    f_out_folder = os.path.join('data', 'unsupervised_data')
    logger.info(f'Collected total of {num_lines:,} examples')
    num_train = max(int(0.8*num_lines), num_lines - int(2e5))
    ts = datetime.datetime.now().strftime('%Y_%m_%d-%H-%M_%s')
    for (_s, _e), _type in zip([(None, num_train), (num_train, None)], ['train', 'dev']):
        _df = df[_s:_e]
        logger.info(f'Writing {len(_df):,} examples for {_type} data...')
        output_folder = os.path.join(f_out_folder, f'run_{ts}', _type)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        # num_cpus = max(multiprocessing.cpu_count() - 1, 1)
        num_cpus = 1
        parallel = joblib.Parallel(n_jobs=num_cpus)
        write_output_file_delayed = joblib.delayed(write_output_file)
        res = parallel((write_output_file_delayed(
            _df.iloc[i:(i+max_examples_per_file)],
            os.path.join(output_folder, f'pretrain_{_type}_{j:03}.txt')
            ) for j, i in enumerate(trange(0, len(_df), max_examples_per_file))))
        logger.info(f'Successfully wrote {len(res):,} file(s) to folder {output_folder}')

if __name__ == "__main__":
    main()
