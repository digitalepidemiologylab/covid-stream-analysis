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
import glob

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')

# globals
usecols = ['user.id', 'user.description', 'lang']
num_files = None
verbose = 0
max_examples_per_file = int(1e6)

def read_data(f_name):
    """Reads single parquet file"""
    df = pd.read_parquet(f_name, columns=usecols)
    return df

def read_covid_data():
    num_cpus = max(multiprocessing.cpu_count() - 1, 1)
    parallel = joblib.Parallel(n_jobs=num_cpus, prefer='threads')
    logger.info('Reading covid data')
    f_names = glob.glob(os.path.join('data', 'extracted', 'tweets', '*.parquet'))
    if isinstance(num_files, int):
        f_names = f_names[:num_files]
    read_data_delayed = joblib.delayed(read_data)
    logger.info('Reading data...')
    df = parallel(read_data_delayed(f_name) for f_name in tqdm(f_names))
    logger.info('Concatenating...')
    df = pd.concat(df)
    return df

def read_archive_data():
    num_cpus = max(multiprocessing.cpu_count() - 1, 1)
    parallel = joblib.Parallel(n_jobs=num_cpus, prefer='threads')
    logger.info('Reading archive data')
    f_names = glob.glob(os.path.join('/', 'mount', 'SDF',  'archive', 'preprocess', 'data', '1_parsed', 'tweets', '*.parquet'))
    if isinstance(num_files, int):
        f_names = f_names[:num_files]
    read_data_delayed = joblib.delayed(read_data)
    logger.info('Reading data...')
    df = parallel(read_data_delayed(f_name) for f_name in tqdm(f_names))
    logger.info('Concatenating...')
    df = pd.concat(df)
    return df

def write_output_file(df, txt_file, parquet_file):
    df.to_parquet(parquet_file)
    with open(txt_file, 'w') as f:
        num_lines = len(df)
        for i, text in tqdm(df['user.description'].iteritems(), total=num_lines):
            f.write(text + '\n')
    return 1

def main():
    df = pd.concat([read_covid_data(), read_archive_data()])
    logger.info(f'Read total {len(df):,} records')
    logger.info(f'Get unique user ids')
    df = df.drop_duplicates(subset=['user.id'])
    logger.info(f'Remaining len: {len(df):,}')
    logger.info('Drop NaN')
    df = df.dropna(subset=['user.description'])
    logger.info(f'Remaining len: {len(df):,}')
    logger.info('Standardize text')
    df['user.description'] = df['user.description'].apply(ProcessTweet.normalize_str)
    logger.info('Drop everything below 3 chars')
    df = df[df['user.description'].str.len() >= 3]
    logger.info(f'Remaining len: {len(df):,}')

    # write output files
    num_lines = len(df)
    f_out_folder = os.path.join('data', 'prediction_data')
    logger.info(f'Collected total of {num_lines:,} examples')
    ts = datetime.datetime.now().strftime('%Y_%m_%d-%H-%M_%s')
    logger.info(f'Writing output files')
    output_folder = os.path.join(f_out_folder, f'run_{ts}')
    for subfolder in ['txt', 'parquet']:
        output_dir = os.path.join(output_folder, subfolder)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    write_output_file_delayed = joblib.delayed(write_output_file)
    num_cpus = max(multiprocessing.cpu_count() - 1, 1)
    parallel = joblib.Parallel(n_jobs=num_cpus)
    res = parallel((write_output_file_delayed(
        df.iloc[i:(i+max_examples_per_file)],
        os.path.join(output_folder, 'txt', f'userbios_{j:03}.txt'),
        os.path.join(output_folder, 'parquet', f'userbios_{j:03}.parquet')
        ) for j, i in enumerate(trange(0, len(df), max_examples_per_file))))
    logger.info(f'Successfully wrote {len(res):,} file(s) to folder {output_folder}')

if __name__ == "__main__":
    main()
