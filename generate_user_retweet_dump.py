import pandas as pd
import joblib
import glob
import logging
import multiprocessing
import os
import time
from datetime import datetime
import gzip
import json
from utils.misc import file_lock
from utils.process_tweet import ProcessTweet
import shutil
from collections import defaultdict
from tqdm import tqdm

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

# dirs
OUTPUT_DIR = os.path.join('data', 'user_retweet_dump')
PRELIM_DIR = os.path.join(OUTPUT_DIR, 'preliminary')
FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'retweet_user_dump.jsonl')

# dictionary of tweet IDs which have already been processed (shared between processes)
manager = multiprocessing.Manager()
used_ids = manager.dict()

def get_f_names(by='day'):
    """Group files by interval"""
    all_f_names = glob.glob('data/raw/**/**/**/**/**')
    f_names = []
    for f_name in all_f_names:
        key = '_'.join(f_name.split('/')[2:5])
        date = datetime.strptime(key, '%Y_%m_%d')
        # only consider tweets betweet May 7 and May 14
        if date >= datetime(2020, 5, 7) and date < datetime(2020, 5, 14):
            f_names.append(f_name)
    return f_names

def extract_user_retweet_info(f_name):
    """Go through raw data and collect user info for certain tweets"""
    def write_to_file(tweet_id, obj):
        """Write to user data to preliminary jsonl file"""
        f_path = os.path.join(PRELIM_DIR, f'{tweet_id}.jsonl')
        with open(f_path, 'a') as f_out:
            with file_lock(f_out):
                f_out.write(json.dumps(obj) + '\n')
    if f_name.endswith('.gz'):
        f = gzip.open(f_name, 'r')
    else:
        f = open(f_name, 'r')
    for i, line in enumerate(f):
        tweet = json.loads(line)
        # ignore logs
        if 'id' not in tweet:
            continue
        # only consider retweets
        if not 'retweeted_status' in tweet:
            continue
        tweet_id = tweet['id_str']
        # ignore duplicates (sometimes the stream collects the same tweets multiple times)
        if tweet_id in used_ids:
            continue
        # flag tweet ID as "used"
        used_ids[tweet_id] = True
        # extract user info
        pt = ProcessTweet(tweet=tweet)
        user_obj = pt.extract_user()
        write_to_file(pt.retweeted_status_id, user_obj)
    f.close()

def merge_file(f_name):
    """Merges preliminary files into a single json file"""
    tweet_id = os.path.basename(f_name).split('.')[0]
    data = {tweet_id: []}
    # read data from single file
    with open(f_name, 'r') as f_in:
        for line in f_in:
            user_obj = json.loads(line)
            data[tweet_id].append(user_obj)
    # write to single output file
    with open(FINAL_OUTPUT_FILE, 'a') as f_out:
        with file_lock(f_out):
            f_out.write(json.dumps(data) + '\n')
    # cleanup
    os.remove(f_name)

def main(no_parallel=False):
    # setup
    s_time = time.time()

    # re-create dirs
    if os.path.isdir(PRELIM_DIR):
        logger.info('Deleting existing preliminary files folder...')
        shutil.rmtree(PRELIM_DIR)
    os.makedirs(PRELIM_DIR)
    if os.path.isfile(FINAL_OUTPUT_FILE):
        logger.info('Deleting existing final output file...')
        os.remove(FINAL_OUTPUT_FILE)

    # set up parallel
    if no_parallel:
        num_cores = 1
    else:
        num_cores = max(multiprocessing.cpu_count() - 1, 1)
    logger.info(f'Using {num_cores} CPUs to parse data...')
    parallel = joblib.Parallel(n_jobs=num_cores)

    f_names = get_f_names()

    logger.info('Extracting user data from raw data...')
    logger.info(f'... found {len(f_names):,} files')
    extract_user_retweet_info_delayed = joblib.delayed(extract_user_retweet_info)
    parallel(extract_user_retweet_info_delayed(f_name) for f_name in tqdm(f_names))

    # merge all files
    logger.info('Merging data...')
    f_names_prelim = glob.glob(os.path.join(PRELIM_DIR, '*.jsonl'))
    logger.info(f'... found {len(f_names_prelim):,} preliminary files')
    merge_file_delayed = joblib.delayed(merge_file)
    parallel(merge_file_delayed(f_name) for f_name in tqdm(f_names_prelim))

    # cleanup
    logger.info('Cleanup...')
    if os.path.isdir(PRELIM_DIR):
        shutil.rmtree(PRELIM_DIR)

    e_time = time.time()
    logger.info(f'Took {(e_time-s_time)/3600:.1f} hours to compute')


if __name__ == "__main__":
    main()
