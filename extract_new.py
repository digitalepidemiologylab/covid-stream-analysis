import pandas as pd
import glob
from collections import defaultdict
import gzip
import json
import os
from utils.process_tweet import ProcessTweet
from utils.helpers import get_dtypes
from utils.misc import file_lock
from covid_19_keywords import KEYWORDS
from tqdm import tqdm
import logging
import joblib
import multiprocessing
import sys
from datetime import datetime
import time
import pickle
import shutil
import ray


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)
EXTRACT_DIR = os.path.join('data', 'extracted_new')
OUTPUT_DIR = os.path.join(EXTRACT_DIR, 'tweets')
PRELIM_DIR = os.path.join(EXTRACT_DIR, 'preliminary')
OTHER_DIR = os.path.join(EXTRACT_DIR, 'other')


manager = multiprocessing.Manager()
# shared writable objects between all processes
originals = manager.dict()
retweet_counts = manager.dict()
quote_counts = manager.dict()
replies_counts = manager.dict()

ray.init(num_cpus=10)

def read_used_files():
    f_path = os.path.join(EXTRACT_DIR, f'.used_data')
    if not os.path.isfile(f_path):
        return {}
    with open(f_path, 'r') as f:
        used_files = json.load(f)
    return used_files

def write_used_files(data_files):
    f_path = os.path.join(EXTRACT_DIR, f'.used_data')
    with open(f_path, 'w') as f:
        json.dump(data_files, f, indent=4)

def generate_file_list(by='day'):
    """Group files by interval"""
    f_names = glob.glob('data/raw/**/**/**/**/**')
    f_names_by_interval = defaultdict(list)
    for f_name in f_names:
        if by == 'day':
            key = '_'.join(f_name.split('/')[2:5])
            date = datetime.strptime(key, '%Y_%m_%d')
        elif by == 'hour':
            key = '_'.join(f_name.split('/')[2:6])
            date = datetime.strptime(key, '%Y_%m_%d_%H')
        if date >= datetime(2020, 5, 7, 0, 0, 0):
            f_names_by_interval[key].append(f_name)
    return f_names_by_interval

@ray.remote
def write_parquet_file(f_path_intermediary, interaction_counts):
    # read from json lines
    dtypes = get_dtypes()
    key = f_path_intermediary.split('/')[-1].split('.jsonl')[0]
    df = pd.read_json(f_path_intermediary, lines=True, dtype=dtypes)
    if len(df) > 0:
        # drop duplicates
        df.drop_duplicates(subset=['id'], inplace=True)
        # read_json converts null to stringified 'None', convert manually
        for col in [c for c, v in dtypes.items() if v == str]:
            df.loc[df[col] == 'None', col] = None
        # sanity check, verify uniqueness of IDs
        df = df.drop_duplicates(subset=['id'])
        # merge with interaction counts
        if len(interaction_counts) > 0:
            # subsetting interaction counts to save memory during merge
            interaction_counts = interaction_counts[interaction_counts.id.isin(df.id.unique())]
            df = df.merge(interaction_counts, on='id', how='left')
            for col in ['num_replies', 'num_quotes', 'num_retweets']:
                df[col] = df[col].fillna(0).astype(int)
        else:
            # set default values
            for col in ['num_replies', 'num_quotes', 'num_retweets']:
                df[col] = 0
        # convert columns to datetime
        for datetime_col in ['created_at', 'user.created_at']:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        # convert to categorical types
        # for col in ['country_code', 'region', 'subregion', 'geo_type', 'lang']:
        for col in ['lang']:
            df[col] = df[col].astype('category')
        # sort by created_at
        df.sort_values('created_at', inplace=True, ascending=True)
        df.reset_index(drop=True, inplace=True)
        # write parquet file
        f_out = os.path.join(OUTPUT_DIR, f'covid_stream_{key}.parquet')
        df.to_parquet(f_out)
    return len(df)

def merge_interaction_counts():
    interaction_counts = pd.DataFrame({
        'num_quotes': pd.Series(dict(quote_counts)),
        'num_replies': pd.Series(dict(replies_counts)),
        'num_retweets': pd.Series(dict(retweet_counts))})
    interaction_counts.index.name = 'id'
    for col in ['num_quotes', 'num_replies', 'num_retweets']:
        interaction_counts[col] = interaction_counts[col].fillna(0).astype(int)
    interaction_counts.reset_index(inplace=True)
    return interaction_counts

def dump_interaction_counts(interaction_counts):
    """Cache interaction counts in case something goes wrong"""
    now = datetime.now().isoformat()
    f_name = os.path.join('/', 'tmp', f'interaction_counts_{now}.pkl')
    logger.info(f'Writing interaction counts to temporary file {f_name}...')
    with open(f_name, 'wb') as f:
        pickle.dump(dict(interaction_counts), f)
    return f_name

def main(no_parallel=False, interval='hour', extract_retweets=False, extract_quotes=True):
    def extract_tweets(key, f_names, interval):
        f_out_other_path = os.path.join(OTHER_DIR, f'{key}.jsonl')
        def write_to_file(obj):
            """Write to coresponding preliminary jsonl file"""
            if interval == 'day':
                created_at = obj['created_at'][:10]
            else:
                # by hour
                created_at = obj['created_at'][:13]
                created_at = created_at.replace('T', '-')
            f_path = os.path.join(PRELIM_DIR, f'{created_at}.jsonl')
            with open(f_path, 'a') as f_out:
                with file_lock(f_out):
                    f_out.write(json.dumps(obj) + '\n')
        for f_name in f_names:
            if f_name.endswith('.gz'):
                f = gzip.open(f_name, 'r')
            else:
                f = open(f_name, 'r')
            for i, line in enumerate(f):
                if len(line) <= 1:
                    continue
                try:
                    tweet = json.loads(line)
                except json.decoder.JSONDecodeError:
                    # some files use single quotation, for this we need to use ast.literal_eval
                    tweet = ast.literal_eval(line)
                except:
                    # sometimes parsing completely fails
                    logger.error('Error parsing line:')
                    logger.error(line)
                    continue
                if 'id' not in tweet:
                    continue
                # extract top-level tweet
                pt = ProcessTweet(tweet=tweet)
                if ((extract_retweets and pt.is_retweet)                              # extract retweets (optional)
                        or (extract_quotes and pt.has_quote and not pt.is_retweet)    # extract quotes if not retweet of a quote (optional)
                        or (not pt.is_retweet and not pt.has_quote)):                 # always extract original tweets which are neither retweets nor quotes
                    extracted_tweet = pt.extract()
                    write_to_file(extracted_tweet)
                # extract subtweets
                if pt.has_quote:
                    pt_quote = ProcessTweet(tweet=tweet['quoted_status'])
                    if not pt.is_retweet:
                        # don't count retweeted quotes
                        if pt_quote.id in quote_counts:
                            quote_counts[pt_quote.id] += 1
                        else:
                            quote_counts[pt_quote.id] = 1
                if pt.is_retweet:
                    pt_retweet = ProcessTweet(tweet=tweet['retweeted_status'])
                    if pt_retweet.id in retweet_counts:
                        retweet_counts[pt_retweet.id] += 1
                    else:
                        retweet_counts[pt_retweet.id] = 1
            f.close()

    # setup
    s_time = time.time()

    # create dirs
    for _dir in [OUTPUT_DIR, OTHER_DIR, PRELIM_DIR]:
        if not os.path.isdir(_dir):
            os.makedirs(_dir)

    # set up parallel
    if no_parallel:
        num_cores = 1
    else:
        num_cores = max(multiprocessing.cpu_count() - 1, 1)
    logger.info(f'Using {num_cores} CPUs to parse data...')
    parallel = joblib.Parallel(n_jobs=num_cores)

    # check for already existing files
    all_f_names = generate_file_list(by=interval)
    used_files = read_used_files()
    f_names = all_f_names
    if len(used_files) > 0:
        len_before = len(f_names)
        for key in used_files.keys():
            if key in f_names:
                if set(used_files[key]) == set(f_names[key]):
                    # all files have been used for this key, remove it from list to be computed
                    f_names.pop(key)
        len_after = len(f_names)
        if len_after == 0:
            logger.info(f'Everything is up-to-date.')
            sys.exit(0)
        elif len_before > len_after:
            logger.info(f'Found a total of {len_before:,} interval-keys. {len_before-len_after:,} are already present.')
        elif len_before == len_after:
            logger.info(f'Found a total of {len_before:,} interval-keys. All of which need to be re-computed.')
    else:
        logger.info('Did not find any pre-existing data')

    # run
    logger.info('Extract tweets...')
    extract_tweets_delayed = joblib.delayed(extract_tweets)
    # parallel(extract_tweets_delayed(key, f_names, interval) for key, f_names in tqdm(f_names.items()))

    # merge interaction counts
    logger.info('Merging all interaction counts...')
    # interaction_counts = merge_interaction_counts()
    # interaction_counts_fname = dump_interaction_counts(interaction_counts)

    with open('/tmp/interaction_counts_2020-10-29T04:51:36.445988.pkl', 'rb') as f:
        interaction_counts = pickle.load(f)
    interaction_counts = pd.DataFrame(interaction_counts)

    # store counts as shared memory
    data_id = ray.put(interaction_counts)

    # add interaction data to tweets and write compressed parquet dataframes
    logger.info('Writing parquet files...')
    f_names_intermediary = glob.glob(os.path.join(PRELIM_DIR, '*.jsonl'))
    res = ray.get([write_parquet_file.remote(f_name, data_id) for f_name in tqdm(f_names_intermediary)])
    num_tweets = sum(res)
    logger.info(f'Collected a total of {num_tweets:,} tweets in {len(f_names_intermediary):,} parquet files')

    # write used files
    logger.info('Writing used files...')
    # write_used_files(all_f_names)

    # cleanup
    # if os.path.isdir(PRELIM_DIR):
    #     logger.info('Cleaning up intermediary files...')
    #     shutil.rmtree(PRELIM_DIR)
    e_time = time.time()
    logger.info(f'Finished in {(e_time-s_time)/60:.1f} min')


if __name__ == "__main__":
    main()
